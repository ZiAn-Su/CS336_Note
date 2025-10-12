
import regex as re
from collections import defaultdict,Counter
from typing import BinaryIO
import os
import pickle
from datetime import datetime
import numpy as np
import multiprocessing as mp



def merge_token(token_pair_fre: dict[tuple[int],int], pair: tuple[int, int], new_index: int) -> dict[bytes,int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_token_pair_fre = {}  
    for key,value in token_pair_fre.items():
        new_key=[]
        i = 0
        while i < len(key):
            if i + 1 < len(key) and key[i] == pair[0] and key[i + 1] == pair[1]:
                 new_key.append(new_index)
                 i += 2
            else:
                new_key.append(key[i])
                i += 1
        new_token_pair_fre[tuple(new_key)]=value
    return new_token_pair_fre

   
def max_token(counts:dict[tuple,int],vocab):
    '''
    返回value最大的元素的key，如果多个元素的value相同，返回key最大的那个元素
    '''
    max_keys=[]
    max_value=0
    for key,value in counts.items():
        if value>max_value:
            max_value=value
            max_keys=[key]
        elif value==max_value:
            max_keys.append(key)
    
    if len(max_keys) > 1:
        max_i=0
        max_ele=(vocab[max_keys[0][0]],vocab[max_keys[0][1]])
        for i in range(1,len(max_keys)):
            ele=(vocab[max_keys[i][0]],vocab[max_keys[i][1]])
            if ele>max_ele:
                max_ele=ele
                max_i=i
        ret_key=max_keys[max_i]
    else:
        ret_key=max_keys[0]
    return ret_key

def train_BPE(token_pair_fre: dict[tuple[int],int], vocab_size: int,special_tokens:list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Start with the list of bytes of string.
    #indices = [item for sublist in voc_fre.keys() for item in sublist]  # @inspect indices
    merges: dict[tuple[bytes, bytes], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    new_index = 255

    # Count the number of occurrences of each pair of tokens
    counts = defaultdict(int)
    for word,value in token_pair_fre.items():
        for index1, index2 in zip(word, word[1:]):  # For each adjacent pair
            counts[(index1, index2)] += value
    len_spec_token=len(special_tokens)
    while len(vocab)+len_spec_token<vocab_size:
        # if len(vocab) == 256:  # 基本词汇表完成后开始显示进度
        #     pbar = tqdm(total=vocab_size-256, desc="Training BPE")
        # Find the most common pair.
        pair = max_token(counts,vocab)
        #pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        
        # Merge that pair.
        new_index +=1  # @inspect new_index
        merges[(vocab[index1],vocab[index2])] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        token_pair_fre = merge_token(token_pair_fre, pair, new_index)  # @inspect indices

        # 查找新token,只更新新token前后组合频率，其他位置不变，删除（index1, index2）被结合了
        # 需要遍历、取元素，空间换时间，
        for key,value in token_pair_fre.items():
            ids = [i for i, x in enumerate(key) if x == new_index]
            len_indices=len(key)
            for id in ids:
                if id>0 and id<len_indices-1:
                    counts[(key[id-1], key[id])] += value
                    counts[(key[id], key[id+1])] += value
                    counts[(key[id-1], index1)] -= value # 不要忘记新的token出现，在新token的前后，与原token的组合是减少的，如they，h和e组成新的he，t h的数量是减少的，e y的数量也是减少的
                    counts[(index2, key[id+1])] -= value
                elif id==0 and len_indices>1:
                    counts[(key[id], key[id+1])] += value
                    counts[(index2, key[id+1])] -= value
                elif id==len_indices-1 and len_indices>1:
                    counts[(key[id-1], key[id])] += value
                    counts[(key[id-1], index1)] -= value
        if (index1, index2) in counts:
            del counts[(index1, index2)]
        
    #     if len(vocab) > 256:
    #         pbar.update(1)
    # if 'pbar' in locals():
    #     pbar.close()
    for token in special_tokens:
        new_index +=1
        vocab[new_index]=token.encode()
    return vocab, list(merges.keys())

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
        voc_fre[word]+=1
    return voc_fre

def count_tokens_paragraph(pattern:str,chunk:str):
    '''
    输入大段原始字符，统计词频
    '''   
    paragraph_list=re.split(pattern,chunk) # 文本中有分隔符，将大段根据分隔符分割为小段
    voc_fre=defaultdict(int)
    for paragraph in paragraph_list:
        voc_fre=count_pre_tokens(paragraph,voc_fre)
    token_pair_fre={}
    for key,value in voc_fre.items():
        token_pair_fre[tuple([k for k in key.encode()])]=value
    return token_pair_fre

def file_to_tokens_freq(input_path:str,num_processes:int,special_tokens:list[str])->defaultdict:
    '''
    读取文件，多进程将文本按照分隔符分割，统计词频
    '''
    chunks=[]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>") # 根据线程数和分隔符，找到文档的分割点
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace('\r\n', '\n')
            chunks.append(chunk)

    #pattern = "|".join(map(re.escape, special_tokens)) 
    #pattern = "|".join(special_tokens)
    pattern = special_tokens[0]
    
    with mp.Pool(num_processes) as pool:
        results=pool.starmap(count_tokens_paragraph,[(pattern,chunk) for chunk in chunks])

    token_pair_fre=defaultdict(int)
    for result_fre in results:
        for key,value in result_fre.items():
            token_pair_fre[key] += value
    return token_pair_fre

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
    #try:
    num_processes = kwargs['num_processes']
    token_pair_fre=file_to_tokens_freq(input_path,num_processes,special_tokens)
    # with open('assignments/assignment1-basics/src/temp/voc_fre.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
    #     pickle.dump(voc_fre, f)

    # with open('src/temp/voc_fre.pkl', 'rb') as f:  # 注意是二进制模式 'rb'
    #     voc_fre= pickle.load(f)

    vocab,merge=train_BPE(token_pair_fre,vocab_size,special_tokens)

    return  vocab,merge          
    # except Exception as e:
    #     print(f"发生错误: {e}")

def main():

    start_time = datetime.now()
    vocab,merge=run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt',vocab_size=10000,num_processes=12)
    end_time = datetime.now()
    # 计算程序运行时间
    run_time = end_time - start_time
    # 打印运行时间
    print(f"TinyStoriesV2-GPT4-train_程序运行时间：{run_time}")

    # start_time = datetime.now()
    # vocab,merge=run_train_bpe('data/TinyStoriesV2-GPT4-train.txt',vocab_size=10000,num_processes=12)
    # end_time = datetime.now()
    # # 计算程序运行时间
    # run_time = end_time - start_time
    # # 打印运行时间
    # print(f"TinyStoriesV2-GPT4-train_程序运行时间：{run_time}")
    # with open('data/TinyStoriesV2-GPT4-train_vocab.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
    #     pickle.dump(vocab, f)
    # with open('data/TinyStoriesV2-GPT4-train_merge.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
    #     pickle.dump(merge, f)
    

    # start_time = datetime.now()
    # vocab,merge=run_train_bpe('data/owt_train.txt',vocab_size=32000,num_processes=12)
    # end_time = datetime.now()
    # # 计算程序运行时间
    # run_time = end_time - start_time
    # # 打印运行时间
    # print(f"程序运行时间：{run_time}")
    # with open('data/owt_train_vocab.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
    #     pickle.dump(vocab, f)
    # with open('data/owt_train_merge.pkl', 'wb') as f:  # 注意是二进制模式 'wb'
    #     pickle.dump(merge, f)

    # # Training the tokenizer
    # string = "the cat in the hat"  # @inspect string
    # params = train_bpe(string, num_merges=3)

    # # Using the tokenizer
    # tokenizer = BPETokenizer(params)
    # string = "the quick brown fox"  # @inspect string
    # indices = tokenizer.encode(string)  # @inspect indices
    # reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    # assert string == reconstructed_string

if __name__=="__main__":
    main()