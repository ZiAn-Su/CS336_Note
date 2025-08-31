import re, collections
import statistics
def get_stats(vocab): 
    '''
    获取所有相邻字符/串的频率
    输入：词汇及词汇频率
    输出：各相邻字符/串及出现的频率
    '''
    pairs = collections.defaultdict(int) #创建默认初始化为0的字典
    for word, freq in vocab.items(): 
        symbols = word.split() 
        for i in range(len(symbols)-1): 
            pairs[symbols[i],symbols[i+1]] += freq 
    return pairs

def merge_vocab(pair, v_in): 
    '''
    根据子词合并词汇
    输入：子词、词汇表
    输出：词汇表
    '''
    v_out = {} 
    bigram = re.escape(' '.join(pair)) # re.escape替换字符串中的特殊字符为原始字符，如.替换为\.
    # re.compile编译一个正则表达式，
    # (?<!\S): 这是一个“负向回顾断言”。它要求 bigram ('e s') 的前面不能是(\S)一个非空白字符。换句话说，它的前面必须是字符串的开头或一个空白字符。
    # (?!\S): 这是一个“负向先行断言”。它要求 bigram ('e s') 的后面不能是(\S)一个非空白字符。换句话说，它的后面必须是字符串的结尾或一个空白字符。
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
    for word in v_in: 
        w_out = p.sub(''.join(pair), word) # 用''.join(pair)替换word中匹配到的内容
        v_out[w_out] = v_in[word] 
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2, 'n e w e s t </w>':6, 'w i d e s t </w>':3} 
# 获取原始词汇表
raw_vocab=[re.sub(r'\s+', '', key) for key in vocab.keys()]

sub_voc=[] #存储子词
last_best='' #上一次的最高频子词
median_value=statistics.median_low(vocab.values()) #计算下中位数频率
num_merges = 100 # 合并次数
for i in range(num_merges): 
    pairs = get_stats(vocab) 
    best = max(pairs, key=pairs.get) # 比较字典的值，key=pairs.get是自定义函数，比较这个函数的结果
    best_value=pairs[best]
    if best_value <= median_value: #中位数判断，防止一直合并，最终得到的是初始词汇表vocab的情况
        break
    vocab = merge_vocab(best, vocab) 

    # 记录子词，用于后续拆分文本
    best_concat=''.join(best)
    if best_concat not in raw_vocab:
        if last_best in best_concat and len(sub_voc) >0:
            sub_voc[-1]=best_concat
        else:
            sub_voc.append(best_concat)
    last_best=best_concat
    print(best)
print(vocab)