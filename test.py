import regex as re
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict,Counter
from typing import BinaryIO
import random
import os
import pickle
from datetime import datetime
import numpy as np
import threading
import queue
import multiprocessing as mp


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def count_pre_tokens(text:str,voc_fre:defaultdict)->defaultdict:
    '''
    统计词频
    '''
    word_list=PAT.findall(text)
    for word in word_list:
        voc_fre[word.encode()]+=1
    return voc_fre

def count_tokens_paragraph(pattern:str,chunk:str):
    '''
    输入大段原始字符，统计词频
    '''   
    paragraph_list=re.split(pattern,chunk) # 文本中有分隔符，将大段根据分隔符分割为小段
    voc_fre=defaultdict(int)
    for paragraph in paragraph_list:
        voc_fre=count_pre_tokens(paragraph,voc_fre)
    return voc_fre

def run_train_bpe(
    input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    **kwargs,
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    输入：文件路径、最大词汇数量、特殊字符
    输出：vocab词汇表， merges合并结果
    '''
    try:
        start_time = datetime.now()
        chunks=[]
        num_processes = kwargs['num_processes']

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>") # 根据线程数和分隔符，找到文档的分割点
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)

        time1=datetime.now()
        print(f"程序运行时间1：{time1-start_time}")

        pattern = "|".join(map(re.escape, special_tokens)) 
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(count_tokens_paragraph, [(pattern, chunk) for chunk in chunks])



        time2=datetime.now()
        print(f"程序运行时间3：{time2-time1}")
        voc_fre=defaultdict(int)
        for result_fre in results:
            for key,value in result_fre.items():
                voc_fre[key] += value

        # with open('assignments/assignment1-basics/src/temp/voc_fre.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
        #     pickle.dump(voc_fre, f)
        # with open('assignments/assignment1-basics/src/temp/voc_fre.pkl', 'rb') as f:  # 注意是二进制模式 'rb'
        #     voc_fre= pickle.load(f)
        end_time = datetime.now()
        # 计算程序运行时间
        run_time = end_time - start_time
        # 打印运行时间
        print(f"程序运行时间：{run_time}")
        #vocab,merge=train_BPE(voc_fre,vocab_size)

                
    except Exception as e:
        print(f"发生错误: {e}")


def main():
    start_time = datetime.now()
    run_train_bpe('assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt',num_processes=1)
    end_time = datetime.now()
    # 计算程序运行时间
    run_time = end_time - start_time
    # 打印运行时间
    print(f"程序运行时间：{run_time}")



if __name__ == '__main__':
    main()