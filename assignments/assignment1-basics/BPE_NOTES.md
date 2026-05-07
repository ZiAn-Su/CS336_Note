# BPE 原理与实现

## 一、BPE 干了什么

把任意 UTF-8 文本切成有限词表里的 subword token。

- 起点: 256 个单字节就能表达任何文本, 但每条文本变得很长。
- 做法: 在语料上反复合并**最高频的相邻字节对**, 把它当作一个新 token 加入词表; 重复直到词表满。
- 产物: `vocab: dict[int, bytes]` (id → 字节序列) 和 `merges: list[tuple[bytes, bytes]]` (按训练顺序记录的合并历史)。

为什么 merges 要按顺序: 训练时它记录"哪一对在第几步被合并", 编码时这个**顺序就是优先级** — 越早合并的越通用。

## 术语: pretoken

**pretoken** = 用 GPT-2 风格正则 `PAT` 从文本中切出的"词级单位": 一个英文单词、一段数字串、一段标点串、或一段空白。BPE 的合并**只能发生在 pretoken 内部**, 不跨 pretoken — 否则 "the cat" 就可能被并成一个 token, 失去语义边界。

```python
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# 例: "Hello, world!" → ["Hello", ",", " world", "!"]
```

训练和编码都以 pretoken 为最小作用域。

## 二、训练 ([src/train_bpe.py](src/train_bpe.py))

**原理**: 在语料中反复挑出**当前最高频的相邻 token 对**, 把它合并成一个新 token; 重复, 直到词表达到目标大小。

```
原始文件
  │
  ▼ 1. 文件分片: 按 b"<|endoftext|>" 把文件大致均分成 N 片 (供多进程并行)
  │
  ▼ 2. 切段: 每片用 special token 切成段落 (special token 自身丢弃, 不参与训练)
  │
  ▼ 3. 预分词 + 词频统计: 每段用 PAT 抽出 pretoken, 累加得到
  │       {pretoken 的 UTF-8 字节序列 : 出现频次}
  │
  ▼ 4. 初态: 每个 pretoken 表示为单字节 id 序列; vocab 初始化为 256 个单字节
  │
  ▼ 5. 合并循环, 直到 |vocab| 达到目标:
  │     a. 统计全语料相邻 pair 的总频率 (按 pretoken 频次加权)
  │     b. 选最高频 pair (a, b)            ← 同频时按 (vocab[a], vocab[b]) 字典序最大
  │     c. 分配新 id X = vocab[a] + vocab[b], 记录到 merges
  │     d. 把所有 pretoken 内的 (a, b) 替换成 X
  │
  ▼ 6. 追加 special tokens (永不参与合并)
  │
  ▼ 输出 (vocab, merges)
```

实现上 `train_BPE` 用 `counts[(a,b)]` (频率) 和 `pair_to_words[(a,b)]` (倒排索引) 把 step 5 摊到 O(受影响词长度), 而不是每轮重扫全语料。

## 三、编码 ([src/bpe_tokenizer.py](src/bpe_tokenizer.py))

**原理**: 把训练阶段记录的 `merges` 当成一份"合并优先级表" — 训练越早创建的 pair 优先级越高。在每个 pretoken 内, 按这个优先级反复合并, 直到序列里没有任何 `merges` 中的 pair。

```
输入字符串
  │
  ▼ 1. 切段: 按 special token 把输入切成『普通段 / 特殊段』交替序列
  │
  ▼ 2. 处理每一段:
  │     ├─ 特殊段: 整段作为一个 token, 直接查 vocab → id
  │     └─ 普通段:
  │          a. PAT 切出 pretoken
  │          b. 每个 pretoken → 单字节 id 序列
  │          c. 在该序列内反复找 rank 最小的可合并 pair (即训练时最早记录的那条 merge),
  │             一次性合并所有不重叠出现, 直到序列中再无 merges 中的 pair
  │          d. 查 vocab → id
  │
  ▼ 3. 串接所有段的 id, 返回
```

关键对称: 训练时**按频率**选择哪一对要合并; 编码时**按训练时的合并顺序 (rank)** 决定先合哪一对。这样保证编码出的 token 序列正是训练阶段"假如这段文本出现在语料中, 它会被切成的样子"。

`encode_iterable` 用 `yield from self.encode(line)` 逐 int 流式产出, 配合按行读文件即可在 1MB 内存内完成 GB 级文件编码。

## 四、推理时大文件并行编码 ([src/inference_bpe.py](src/inference_bpe.py))

GB 级语料离线编码用; 不参与课程测试。

- `ProcessPoolExecutor + initializer`: 每个 worker 启动时加载一次 tokenizer 到全局变量, 之后任务复用 (避免反复 pickle 大对象)。
- 文件按字节区间切给各 worker, 行级对齐边界 (非起点 chunk 先 `readline()` 跳半行, 越界后停; 既不重叠也不漏)。
- 输出 `.bin` (token id 二进制数组) + `.meta` (dtype / vocab_size / total_tokens)。

## 五、三件套职责对照

| 文件 | 职责 | 对外入口 |
|---|---|---|
| [src/train_bpe.py](src/train_bpe.py) | 语料 → `(vocab, merges)` | `run_train_bpe(input_path, vocab_size, special_tokens)` |
| [src/bpe_tokenizer.py](src/bpe_tokenizer.py) | `(vocab, merges)` → tokenizer 对象 | `BPETokenizer.encode / decode / encode_iterable` |
| [src/inference_bpe.py](src/inference_bpe.py) | 并行批编码大文件 | `encode_large_file_parallel(...)` |
