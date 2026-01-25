# CS336 Spring 2025 Assignment 1: Basics
## 文件说明
```
# tokenizer相关
src/bpe_tokenizer.py：bpe_tokenizer的定义
src/train_bpe.py：训练bpe tokenizer，vocab: dict[int, bytes]，词汇表，映射token到bytes； merges: list[tuple[bytes, bytes]]：按顺序记录哪些bytes与哪些bytes结合
src/inference_bpe.py：bpe tokenizer推理
src/tokenizer_hf/*：huggginface的gpt2 tokenizer

# 语言模型相关
src/utils.py: 通用函数
src/transformer.py： 模型架构定义
src/train.py：训练模型
src/inference.py：模型推理
```

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

