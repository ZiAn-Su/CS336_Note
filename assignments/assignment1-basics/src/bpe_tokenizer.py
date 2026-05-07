"""BPE Tokenizer

给定 (vocab, merges, special_tokens), 实现编码/解码:
- encode(s)         -> list[int]
- encode_iterable(it) -> Iterator[int]   (流式, 低内存)
- decode(ids)       -> str

GPT-2 风格 BPE 的关键约束:
1) 编码前必须按 PAT 正则做 pretokenization, 合并只能发生在单个 pretoken 内部.
2) special tokens 永远作为单一 token, 不参与 BPE 合并; 多个 special token
   存在前缀关系时 (如 "<|eot|>" 与 "<|eot|><|eot|>"), 必须按长度降序匹配长者优先.
3) 在 pretoken 内部合并时, 每轮选取『当前序列里能匹配上的、训练阶段排名最靠前
   的 pair』作为本轮合并目标 — 即按 merge_rank 选最小, 而非按 pair 频率.
"""

from abc import ABC
from typing import Iterable, Iterator
import regex as re

# GPT-2 风格 pretokenization 正则: 缩写 / 字母簇 / 数字簇 / 其它符号簇 / 空白
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer(ABC):
    """Tokenizer 抽象接口."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # pair -> 训练阶段创建顺序 (越小, 优先级越高)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {p: i for i, p in enumerate(merges)}

        # bytes -> token_id 反查
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # 特殊 token 按长度降序拼成正则, 保证长前缀优先匹配
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = re.compile("|".join(re.escape(t) for t in sorted_specials))
        else:
            self.special_pat = None

    # ----------------------------- 单 pretoken 内部 BPE 合并 -----------------------------

    def _bpe_pretoken(self, pretoken_bytes: bytes) -> list[bytes]:
        """对一个 pretoken 的字节序列做 BPE 合并, 返回最终的字节 token 列表.

        策略: 反复扫描当前 tokens, 找出 merge_rank 最小的 pair, 一次性合并所有
        不重叠出现, 直到没有可合并 pair 为止.
        """
        if len(pretoken_bytes) < 2:
            return [bytes([b]) for b in pretoken_bytes]

        tokens: list[bytes] = [bytes([b]) for b in pretoken_bytes]

        while len(tokens) >= 2:
            # 找出当前 tokens 中 rank 最小 (优先级最高) 的可合并 pair
            best_rank = None
            best_pair = None
            for i in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (tokens[i], tokens[i + 1])

            if best_pair is None:
                break

            # 一次性合并所有不重叠出现, 然后再次搜索下一轮
            a, b = best_pair
            merged = a + b
            new_tokens: list[bytes] = []
            i, n = 0, len(tokens)
            while i < n:
                if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    # ----------------------------- 编码 -----------------------------

    def _encode_no_special(self, string: str) -> list[int]:
        """对不含特殊 token 的纯文本: pretokenize -> 逐 pretoken BPE -> 转 id."""
        ids: list[int] = []
        for m in PAT.finditer(string):
            for tok in self._bpe_pretoken(m.group().encode("utf-8")):
                ids.append(self.byte_to_id[tok])
        return ids

    def encode(self, string: str) -> list[int]:
        if not string:
            return []
        if self.special_pat is None:
            return self._encode_no_special(string)

        # 把字符串按 special token 切成『普通段 / 特殊段』交替序列
        ids: list[int] = []
        last_end = 0
        for m in self.special_pat.finditer(string):
            if m.start() > last_end:
                ids.extend(self._encode_no_special(string[last_end:m.start()]))
            ids.append(self.byte_to_id[m.group().encode("utf-8")])
            last_end = m.end()
        if last_end < len(string):
            ids.extend(self._encode_no_special(string[last_end:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码: 逐行处理, 逐 int 产出, 不会把整文件读入内存."""
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    # ----------------------------- 解码 -----------------------------

    def decode(self, indices: list[int]) -> str:
        # errors="replace": 当 id 序列不构成合法 UTF-8 时用 U+FFFD 占位, 避免抛错
        return b"".join(self.vocab[i] for i in indices).decode("utf-8", errors="replace")
