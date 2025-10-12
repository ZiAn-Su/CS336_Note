from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="stanford-cs336/owt-sample",
    filename="owt_valid.txt.gz",
    repo_type='dataset',
    local_dir='assignments/assignment1-basics/data',
    local_dir_use_symlinks=False
    # token="your_token" # 如果需要下载私有模型，同样可以加token
)
print(f"文件已通过镜像站下载到：{file_path}")