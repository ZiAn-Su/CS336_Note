# CS336 Spring 2025 Assignment 1: Basics

## 文档说明 (从这里开始读)
本仓库的所有说明文档及其用途, 按推荐阅读顺序排列:

| 文档 | 类型 | 内容 | 何时读 |
|---|---|---|---|
| [README.md](README.md) | 项目入口 | 文档导航、源码结构、测试与数据下载命令 | **入门第一步** |
| [Note.md](Note.md) | 作业知识点笔记 | 沿作业题号 (2.1, 2.2, ...) 记录的 Unicode/BPE/Transformer 等概念笔记, 含原题与解答 | 想理解题目背景与理论时 |
| [BPE_NOTES.md](BPE_NOTES.md) | BPE 实现总结 | BPE 训练/编码的整体设计、关键数据结构、复杂度、踩过的坑 (含修复记录) | 想理解 `src/*bpe*.py` 三件套的实现细节时 |

> 维护约定: 新增独立文档时, 在上表追加一行说明; 不要让说明文档无依据地散落, 否则两周后自己也找不到。

## 源码结构
```
# tokenizer 相关
src/bpe_tokenizer.py   BPE tokenizer 类 (encode / decode / encode_iterable)
src/train_bpe.py       从原始语料训练 BPE → (vocab, merges)
src/inference_bpe.py   并行编码大文件 → token id 二进制数组 (.bin + .meta)
src/tokenizer_hf/*     HuggingFace 的 GPT-2 tokenizer (参考实现)

# 语言模型相关
src/utils.py           通用函数 (loss / 优化器 / 学习率调度 / checkpoint 等)
src/transformer.py     模型架构定义 (Linear / RMSNorm / RoPE / MHA / FFN / Block / LM)
src/train.py           训练模型
src/inference.py       模型推理 (生成)
```

数据类型约定:
- `vocab: dict[int, bytes]`  — token id → 字节序列
- `merges: list[tuple[bytes, bytes]]` — 按训练顺序记录的合并历史 (编码阶段决定合并优先级)

## 单元测试
```
#单元测试某个文件
uv run pytest tests/test_train_bpe.py
#单元测试某个文件的某个函数
uv run pytest path/to/test_file.py::test_function_name
# 或
uv run pytest -k test_function_name
```
## Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

