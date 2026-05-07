"""并行 BPE 推理编码 (大文件 -> token id 二进制数组)

为什么这样组织:
- BPETokenizer 对象较大, 通过 pickle 反复跨进程传递很慢. 因此把 tokenizer 放到
  子进程的全局变量里, 通过 ProcessPoolExecutor 的 initializer 在每个 worker
  启动时加载一次, 之后所有任务复用.
- 文件按字节区间均分给各 worker. 为避免边界处一行被两个 chunk 同时处理或漏掉:
  每个非起始 chunk 先 readline() 跳过半行残端, 后续行属于本 chunk;
  本 chunk 越界后停止 (越界处的半行交给下一 chunk).
- worker 内部把读到的行累积到 buffer, 达到阈值再调用 encode 一次, 减少调用开销.
"""

import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.bpe_tokenizer import BPETokenizer

# 子进程全局 tokenizer (init_worker 中初始化)
_tokenizer: BPETokenizer | None = None


def init_worker(vocab_path: str, merge_path: str) -> None:
    """子进程启动时加载一次 tokenizer."""
    global _tokenizer
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merge_path, "rb") as f:
        merges = pickle.load(f)
    _tokenizer = BPETokenizer(vocab, merges)


def process_chunk(args) -> np.ndarray:
    """编码 [start_byte, end_byte) 范围的文本; 行级对齐边界以避免重叠/遗漏."""
    filename, start_byte, end_byte, dtype = args
    encoded: list[int] = []

    BUFFER_FLUSH = 100_000  # 字符数阈值, 累积到此值再 encode 一次

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_byte)
        # 不是文件起点时, 跳过本行残端 (上一 chunk 已负责处理)
        if start_byte != 0:
            f.readline()

        buffer = ""
        while f.tell() < end_byte:
            line = f.readline()
            if not line:  # EOF
                break
            buffer += line
            if len(buffer) >= BUFFER_FLUSH:
                encoded.extend(_tokenizer.encode(buffer))
                buffer = ""
        if buffer:
            encoded.extend(_tokenizer.encode(buffer))

    return np.array(encoded, dtype=dtype)


def encode_large_file_parallel(
    input_file: str,
    output_file: str,
    vocab_path: str,
    merge_path: str,
    chunk_size_mb: int = 20,
) -> None:
    """大文件并行编码, 输出 token id 二进制数组 + .meta 元信息."""
    with open(vocab_path, "rb") as f:
        vocab_len = len(pickle.load(f))
    dtype = np.uint16 if vocab_len < 65535 else np.uint32
    print(f"Vocab size: {vocab_len}, dtype: {np.dtype(dtype).name}")

    file_size = os.path.getsize(input_file)
    chunk_bytes = chunk_size_mb * 1024 * 1024
    chunks = []
    start = 0
    while start < file_size:
        end = min(start + chunk_bytes, file_size)
        chunks.append((input_file, start, end, dtype))
        start = end
    print(f"Split into {len(chunks)} chunks, using {os.cpu_count()} cores...")

    total_tokens = 0
    with open(output_file, "wb") as f_out, ProcessPoolExecutor(
        initializer=init_worker, initargs=(vocab_path, merge_path)
    ) as executor:
        # executor.map 保证按输入顺序返回结果, 即可保持原始文本顺序
        for chunk_arr in tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Encoding",
        ):
            f_out.write(chunk_arr.tobytes())
            total_tokens += len(chunk_arr)

    print(
        f"Done. Total tokens: {total_tokens}, "
        f"output: {os.path.getsize(output_file) / (1024**3):.2f} GB"
    )

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": vocab_len,
        "total_tokens": total_tokens,
    }
    with open(f"{output_file}.meta", "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    in_path = "data/TinyStoriesV2-GPT4-train.txt"
    out_path = "data/train.bin"
    v_path = "data/TinyStoriesV2-GPT4-train_vocab.pkl"
    m_path = "data/TinyStoriesV2-GPT4-train_merge.pkl"

    # chunk_size_mb 视内存调整, 通常 10MB - 100MB 较合适
    encode_large_file_parallel(in_path, out_path, v_path, m_path, chunk_size_mb=10)
