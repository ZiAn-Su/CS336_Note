"""BPE 训练 (从原始语料 -> vocab + merges)

流水线:
  原始文件
    └─> find_chunk_boundaries  按 special token 大致均匀切片 (子进程独立处理)
    └─> count_pretokens (并行) 用 PAT 抽取 pretoken, 累加频率
    └─> train_BPE              贪心合并出现频率最高的相邻 pair, 直到词表满

关键数据结构:
  token_pair_fre: {pretoken_bytes_tuple: freq}  — pretoken 的字节 id 序列 -> 频次
  vocab        : {int -> bytes}                  — token id -> 字节序列
  merges       : [(bytes, bytes), ...]           — 合并历史 (按训练顺序)
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import BinaryIO
import multiprocessing as mp

import regex as re

# 与 bpe_tokenizer.py 一致的 GPT-2 风格 pretokenization 正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# ============================ 训练核心 ============================

def max_token(counts: dict[tuple[int, int], int],
              vocab: dict[int, bytes]) -> tuple[int, int] | None:
    """选最高频的 pair, 平局时按 (vocab[a], vocab[b]) 字典序最大者 (与 GPT-2 参考实现一致)."""
    if not counts:
        return None
    max_value = max(counts.values())
    if max_value <= 0:
        return None
    return max(
        (k for k, v in counts.items() if v == max_value),
        key=lambda k: (vocab[k[0]], vocab[k[1]]),
    )


def train_BPE(
    token_pair_fre: dict[tuple[int, ...], int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """反复合并最高频 pair, 直到词表大小达到 vocab_size.

    数据结构:
      words[idx]            该词当前的 token id 序列 (会被原地替换)
      word_freqs[idx]       该词的出现频率
      counts[(a, b)]        全语料中相邻对 (a, b) 的总频率
      pair_to_words[(a, b)] 含 pair (a, b) 的词索引集合 (避免每轮全表扫描)

    每轮:
      1) 选 counts 中频率最高的 pair = (index1, index2)
      2) 仅遍历 pair_to_words[pair] 中的词:
         - 减去该词在 counts/pair_to_words 中的全部旧贡献
         - 把词中所有 (index1, index2) 替换为 new_index
         - 加回新词的全部 pair 贡献
      『先减整词旧贡献, 再加整词新贡献』的对称更新, 避免相邻合并 (如 abab→XX)
      时邻居引用错乱 — 这是常见的 off-by-one 来源.
    """
    merges: dict[tuple[bytes, bytes], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    new_index = 255

    words: list[list[int]] = [list(w) for w in token_pair_fre.keys()]
    word_freqs: list[int] = list(token_pair_fre.values())

    counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, word in enumerate(words):
        freq = word_freqs[idx]
        for j in range(len(word) - 1):
            p = (word[j], word[j + 1])
            counts[p] += freq
            pair_to_words[p].add(idx)

    target_size = vocab_size - len(special_tokens)
    while len(vocab) < target_size:
        pair = max_token(counts, vocab)
        if pair is None:
            break
        index1, index2 = pair
        new_index += 1
        merges[(vocab[index1], vocab[index2])] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]

        affected = list(pair_to_words.get(pair, ()))
        for idx in affected:
            word = words[idx]
            freq = word_freqs[idx]

            # 1) 移除旧贡献
            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                counts[p] -= freq
                if counts[p] <= 0:
                    counts.pop(p, None)
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(idx)
                    if not s:
                        pair_to_words.pop(p, None)

            # 2) 词内合并 pair -> new_index
            new_word: list[int] = []
            i, n = 0, len(word)
            while i < n:
                if i + 1 < n and word[i] == index1 and word[i + 1] == index2:
                    new_word.append(new_index)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word

            # 3) 加回新贡献
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                counts[p] += freq
                pair_to_words[p].add(idx)

    # 特殊 token 永远作为单一 token, 训练完后追加到词表
    for token in special_tokens:
        new_index += 1
        vocab[new_index] = token.encode("utf-8")
    return vocab, list(merges.keys())


# ============================ 文件分片 + 预分词 ============================

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """大致均匀切片, 把每个边界对齐到下一个 split_special_token 处.
    保证每个 chunk 不会切到一个文档中间, 可独立 pretoken 统计.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            mini = file.read(mini_chunk_size)
            if not mini:
                boundaries[bi] = file_size
                break
            found = mini.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos + found
                break
            pos += mini_chunk_size

    # 去重并排序 (相邻边界可能撞上, 实际 chunk 数会少于 desired_num_chunks)
    return sorted(set(boundaries))


def count_pretokens(special_pattern: str, chunk: str) -> dict[tuple[int, ...], int]:
    """单进程 worker:
       1) 按 special token 切段 (special token 自身丢弃, 不参与训练)
       2) 每段用 PAT 抽 pretoken
       3) 返回 {pretoken 字节序列 (tuple of int) -> 频次}.
    """
    voc_fre: dict[str, int] = defaultdict(int)
    paragraphs = re.split(special_pattern, chunk) if special_pattern else [chunk]
    for paragraph in paragraphs:
        for word in PAT.findall(paragraph):
            voc_fre[word] += 1
    return {tuple(word.encode("utf-8")): freq for word, freq in voc_fre.items()}


def file_to_tokens_freq(
    input_path: str,
    num_processes: int,
    special_tokens: list[str],
) -> dict[tuple[int, ...], int]:
    """读文件, 多进程并行 pretokenize, 合并出全语料频率表."""
    # 用第一个 special token 作为切片对齐锚点 (一般是 "<|endoftext|>")
    split_token = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        chunks: list[str] = []
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            f.seek(s)
            chunk = f.read(e - s).decode("utf-8", errors="ignore")
            chunks.append(chunk.replace("\r\n", "\n"))

    # 多个 special token 都需要参与切分 (注意 escape, 因为 "<|...|>" 在正则里可能被解释)
    special_pattern = "|".join(re.escape(t) for t in special_tokens) if special_tokens else ""

    with mp.Pool(num_processes) as pool:
        results = pool.starmap(count_pretokens, [(special_pattern, c) for c in chunks])

    token_pair_fre: dict[tuple[int, ...], int] = defaultdict(int)
    for r in results:
        for k, v in r.items():
            token_pair_fre[k] += v
    return token_pair_fre


# ============================ 顶层入口 ============================

def run_train_bpe(
    input_path,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从原始语料训练 BPE tokenizer.

    Args:
        input_path:     训练语料文件路径
        vocab_size:     词表大小目标 (含 special tokens)
        special_tokens: 不参与合并、最后追加到词表的特殊 token 列表
        kwargs:         num_processes 控制并行数, 默认 12
    Returns:
        (vocab, merges)
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    num_processes = kwargs.get("num_processes", 12)

    token_pair_fre = file_to_tokens_freq(input_path, num_processes, special_tokens)
    return train_BPE(token_pair_fre, vocab_size, special_tokens)


def main():
    start = datetime.now()
    vocab, merges = run_train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        num_processes=12,
    )
    print(f"训练耗时: {datetime.now() - start}")
    print(f"vocab 大小: {len(vocab)}, merges 数量: {len(merges)}")

    # 如需保存:
    # import pickle
    # with open("data/vocab.pkl", "wb") as f: pickle.dump(vocab, f)
    # with open("data/merges.pkl", "wb") as f: pickle.dump(merges, f)


if __name__ == "__main__":
    main()
