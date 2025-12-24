import os
import pickle
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import datetime
from src.bpe_tokenizer import BPETokenizer

# --- 全局变量 (用于子进程) ---
# 为了避免将巨大的Tokenizer对象通过Pickle在进程间反复传递（这非常慢），
# 我们使用全局变量在每个子进程初始化时加载一次。
tokenizer = None

def init_worker(vocab_path, merge_path):
    """每个子进程启动时运行一次，加载Tokenizer"""
    global tokenizer
    with open(vocab_path, 'rb') as f: vocab = pickle.load(f)
    with open(merge_path, 'rb') as f: merge = pickle.load(f)
    tokenizer = BPETokenizer(vocab, merge)

def process_chunk(args):
    """
    子进程的工作函数：读取指定字节范围的文本并编码
    """
    filename, start_byte, end_byte, dtype = args
    global tokenizer
    
    encoded_chunk = []
    total_bytes=end_byte-start_byte
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        # 1. 跳转到指定的起始位置
        f.seek(start_byte)
        
        # 2. 如果不是文件的开头，我们要跳过第一行
        # 原因：这一行的开头部分已经被上一个 chunk 处理了（见下文逻辑）
        if start_byte != 0:
            f.readline()
        
        lines=''
        
        # 3. 开始读取并编码
        while True:
            # 记录当前指针位置
            curr_pos = f.tell()
            
            # 如果当前位置已经超过了分配的结束位置，停止处理
            # 下一个 chunk 的进程会负责从这里开始读取
            if curr_pos >= end_byte:
                if lines:
                    ids = tokenizer.encode(text)
                    encoded_chunk.extend(ids)
                break
            
            line = f.readline()
            if not line: # 文件结束
                if lines:
                    ids = tokenizer.encode(text)
                    encoded_chunk.extend(ids)
                break
                
            text = line.strip()
            lines+=text
            if len(lines)>10000:
                ids = tokenizer.encode(lines)
                encoded_chunk.extend(ids)
                lines=''
                print(f'processed: {(curr_pos-start_byte)/total_bytes*100 :.2f}%')
    
    # 将列表转换为紧凑的 numpy 数组返回
    return np.array(encoded_chunk, dtype=dtype)

def encode_large_file_parallel(input_file, output_file, vocab_path, merge_path, chunk_size_mb=20):
    # 1. 确定数据类型
    with open(vocab_path, 'rb') as f: vocab_len = len(pickle.load(f))
    dtype = np.uint16 if vocab_len < 65535 else np.uint32
    dtype_size = 2 if dtype == np.uint16 else 4
    print(f"Vocab size: {vocab_len}, using dtype: {dtype}")

    # 2. 计算文件分块
    file_size = os.path.getsize(input_file)
    chunk_size = chunk_size_mb * 1024 * 1024 # 转换 MB 到 Bytes
    
    # 生成任务列表：每个任务是一个元组 (start, end)
    chunks = []
    start = 0
    while start < file_size:
        end = min(start + chunk_size, file_size)
        chunks.append((input_file, start, end, dtype))
        start = end
    
    print(f"Split file into {len(chunks)} chunks. Processing with {os.cpu_count()} cores...")

    # 3. 并行处理与写入
    # 使用 'wb' 模式打开输出文件
    total_tokens = 0
    
    with open(output_file, 'wb') as f_out:
        # max_workers=None 默认使用 CPU 核心数
        with ProcessPoolExecutor(initializer=init_worker, initargs=(vocab_path, merge_path)) as executor:
            # map 会保证结果按输入 chunks 的顺序返回，这对于保持文本顺序至关重要
            results = executor.map(process_chunk, chunks)
            
            # 使用 tqdm 显示进度
            for chunk_arr in tqdm(results, total=len(chunks), desc="Encoding Parallel"):
                # 写入二进制
                f_out.write(chunk_arr.tobytes())
                total_tokens += len(chunk_arr)

    print(f"Done! Total tokens: {total_tokens}")
    print(f"Output size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

    # 4. 保存元数据
    meta = {
        'dtype': 'uint16' if dtype == np.uint16 else 'uint32',
        'vocab_size': vocab_len,
        'total_tokens': total_tokens
    }
    with open(f'{output_file}.meta', 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    # 配置路径
    in_path = "data/TinyStoriesV2-GPT4-train.txt"
    out_path = "data/train.bin"
    v_path = 'data/TinyStoriesV2-GPT4-train_vocab.pkl'
    m_path = 'data/TinyStoriesV2-GPT4-train_merge.pkl'

    # 运行并行编码
    # chunk_size_mb 可以根据内存调整，通常 10MB - 100MB 比较合适
    encode_large_file_parallel(in_path, out_path, v_path, m_path, chunk_size_mb=1)