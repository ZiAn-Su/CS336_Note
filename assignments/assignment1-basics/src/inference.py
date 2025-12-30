import os
import time
from contextlib import nullcontext
from dataclasses import dataclass,asdict
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
from transformer import *
from utils import *
from train import TrainingConfig

@dataclass
class InferConfig:
    vocab_size: int = 50257      # 词汇表数量
    context_length: int = 512     # 一次处理的最大token数
    d_model: int = 256            # 特征维度（嵌入模型维度及其他层）
    num_layers: int = 3        # Transformer层的数量
    num_heads: int = 8       # 多头注意力头数
    d_ff: int = 512              # 前馈神经网络的维度
    rope_theta: float = 10000.0         # rope旋转位置编码的theta值
    device: str = 'cuda'
def main():
    # 读取模型
    cfg=InferConfig()
    checkpoint_path='checkpoints/run_20251230_080621/ckpt.pt'
    model = TransformerLM(vocab_size=cfg.vocab_size,context_length=cfg.context_length,d_model=cfg.d_model,num_layers=cfg.num_layers,num_heads=cfg.num_heads,d_ff=cfg.d_ff,rope_theta=cfg.rope_theta,device=cfg.device)
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)

    # tokenizer
    prompt='Once upon a time there was a little boy named Ben. Ben loved to explore'
    tokenizer = AutoTokenizer.from_pretrained("gpt2",cache_dir='checkpoints/gpt2',local_files_only=True, use_fast=True)
    encoded_token = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens_tensor = torch.tensor([encoded_token], dtype=torch.long, device=cfg.device)
    model.eval()
    with torch.no_grad():
        pred_t = model(tokens_tensor)
        print(pred_t)


if __name__=="__main__":
    main()