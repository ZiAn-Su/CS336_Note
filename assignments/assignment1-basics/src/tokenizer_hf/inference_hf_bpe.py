import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def encode_file_hf(input_file, output_file, model_name="gpt2", batch_size=10000):
    """
    使用 HuggingFace Fast Tokenizer 对大文件进行编码并保存为二进制
    """
    # 1. 加载 Tokenizer
    # use_fast=True 是默认的，但显式指定以确保使用 Rust 后端
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='checkpoints/gpt2',local_files_only=True, use_fast=True)
    
    # 检查是否确实是 Fast Tokenizer
    if not tokenizer.is_fast:
        raise ValueError("This model does not have a Fast Tokenizer available!")

    # 2. 确定存储数据类型
    vocab_size = tokenizer.vocab_size
    if vocab_size < 65535:
        dtype = np.uint16
        print(f"Vocab size {vocab_size} < 65k. Using np.uint16 (Optimized for storage).")
    else:
        dtype = np.uint32
        print(f"Vocab size {vocab_size} > 65k. Using np.uint32.")

    # 3. 统计行数 (用于进度条，可选)
    print("Counting lines...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f)

    # 4. 批处理循环
    # 我们读取 batch_size 行文本，然后一次性喂给 tokenizer
    token_count = 0
    batch_buffer = []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_file, 'wb') as f_out:
        
        # 使用 tqdm 包装文件迭代器
        pbar = tqdm(f_in, total=total_lines, desc="Processing")
        
        for line in pbar:
            text = line.strip()
            if not text:
                continue
            
            batch_buffer.append(text)
            
            # 当缓冲区满时，执行编码
            if len(batch_buffer) >= batch_size:
                # encode_batch 极快，内部并发
                # add_special_tokens=False 取决于你是否需要 <s> 或 <|endoftext|>
                # 这里我们假设只编码纯文本，不加特殊标记
                encoded_batch = tokenizer(batch_buffer, add_special_tokens=False)["input_ids"]
                
                # 展平列表并写入
                for enc in encoded_batch:
                    # 添加 EOT token? (可选)
                    # enc.append(tokenizer.eos_token_id) 
                    
                    arr = np.array(enc, dtype=dtype)
                    f_out.write(arr.tobytes())
                    token_count += len(enc)
                
                # 清空缓冲
                batch_buffer = []

        # 处理剩余的缓冲
        if batch_buffer:
            encoded_batch = tokenizer(batch_buffer, add_special_tokens=False)["input_ids"]
            for enc in encoded_batch:
                arr = np.array(enc, dtype=dtype)
                f_out.write(arr.tobytes())
                token_count += len(enc)

    print(f"\nDone! Total tokens: {token_count}")
    print(f"Saved to: {output_file}")
    
    # 打印元数据建议，方便读取
    print(f"REMEMBER: Read this file using dtype={dtype.__name__}")

if __name__ == "__main__":
    # 1. 准备大文件
    in_path = "/mnt/e/Cousera/CS336_Note/assignments/assignment1-basics/data/owt_train.txt"

    
    # 2. 运行编码 (推荐 gpt2)
    # 第一次运行会自动下载模型
    encode_file_hf(in_path, "data/train_owt.bin", model_name="gpt2", batch_size=5000)