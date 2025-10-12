import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import pickle
from typing import Dict, Tuple

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

def count_pre_tokens(text:str,PAT,voc_fre:defaultdict)->defaultdict:
    '''
    统计词频
    '''
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_list=re.findall(PAT, text)
    for word in word_list:
        voc_fre[word]+=1
    return voc_fre



def save_with_pickle(data: Dict[Tuple[bytes, ...], int], filename: str) -> None:
    """使用pickle保存数据"""
    with open(filename, 'wb') as f:  # 注意是二进制模式 'wb'
        pickle.dump(data, f)

def load_with_pickle(filename: str) -> Dict[Tuple[bytes, ...], int]:
    """使用pickle加载数据"""
    with open(filename, 'rb') as f:  # 注意是二进制模式 'rb'
        return pickle.load(f)

def find_pos(key:tuple[bytes,...],pattern:tuple[bytes,bytes]):
    '''
    查找位置，快慢指针
    '''
    len_k=len(key)
    len_p=len(pattern)
    pos=[]
    i=0
    while i <len_k-len_p+1:
        j=0
        while j<len_p and key[i+j]==pattern[j]:
            j+=1
        if j>=len_p:
            pos.append(i)
            i+=len_p # 跳过模式串的长度，不需要重复匹配
        else:
            i+=1
    return pos

def replace_tuple(key:tuple[bytes,bytes],pos:list,pattern:tuple[bytes,bytes]):
    pattern_l=b''.join(pattern)
    new_key=[]
    k=0
    while k <len(key):
        if k in pos:
            new_key.append(pattern_l)
            k+=len(pattern)
        else:
            new_key.append(key[k])
            k+=1
    return tuple(new_key)

def merge_vocab(vocab:dict[tuple[bytes,...],int],pattern:tuple[bytes,bytes]):
    '''
    查找模式串，合并词汇表
    '''
    new_vocab={}
    for key,value in vocab.items():
        pos=find_pos(key,pattern)
        new_key=replace_tuple(key,pos,pattern)
        new_vocab[new_key]=value
    return new_vocab


## Usage
with open('assignments/assignment1-basics/src/temp/TinyStoriesV2-GPT4-valid.txt', "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    chunks=[]
    special_token_list=["<|endoftext|>"]
    pattern = "|".join(map(re.escape, special_token_list))
    voc_fre=defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        vocab_size=1000
        # 存储所有需要合并的字节串 list[tuple[bytes, bytes]]
        # 初始化为0-255和特殊字符
        merges=[(bytes([i])) for i in range(256)]
        for token in special_token_list:
            merges.append(token.encode())
        

        # 使用多个分割字符串分割字符串
        split_sen=re.split(pattern,chunk)
        # 分割字符串并统计词频
        for sen in split_sen:
            voc_fre=count_pre_tokens(sen,PAT,voc_fre)
        
        ## 初始化词汇表 vocab：{tuple[bytes,...]:int}
        ### str转tuple[bytes]
        vocab={tuple([c.encode() for c in key]):value for key,value in voc_fre.items()}
        # save_with_pickle(vocab,'voc_fre.pkl') # 用于中间结果测试

        last_best=()
        while True:
            ## 统计词频 输入词汇表，计算相邻词汇的频率
            adjoin_fre=defaultdict(int)
            for key,value in vocab.items():
                ### 如果仅以一个字符如何处理 假设单个字符频率最高，无法合并，最终目的是找到高频重复字符，将已有单词切分
                for i in range(1,len(key)):
                    adjoin_fre[(key[i-1],key[i])]+=value
            ## 选择最高词频，更新词汇表
            max_ele=max(adjoin_fre,key=lambda x:(adjoin_fre[x],x))

            ## 更新词汇表
            vocab=merge_vocab(vocab,max_ele)
            if last_best!=() and b''.join(last_best) in b''.join(max_ele):
                merges[-1]=max_ele
            else:
                merges.append(max_ele)
            last_best=max_ele
            if len(merges)>=vocab_size:
                break
        print(voc_fre)