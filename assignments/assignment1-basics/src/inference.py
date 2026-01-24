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
from train import ModelConfig,TrainingConfig
import torch.nn.functional as F

@dataclass
class InferConfig(ModelConfig):
    device: str = 'cuda'
    # 新增生成配置
    max_new_tokens: int = 200  # 最大生成长度
    temperature: float = 1.0    # 温度采样 (0-1之间，越小越保守，越大越随机)
    top_p: float = 0.8  # 新增：Top-P 采样阈值 (通常在 0.8-0.95 之间)
    top_k: int = None             # Top-K 采样过滤

def generate(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # --- Top-K 过滤 ---
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # --- Top-P (Nucleus) 过滤 ---
            if top_p is not None and top_p < 1.0:
                # 对 logits 按降序排列
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算累计概率
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累计概率超过 top_p 的 token
                # 我们希望保留累计和刚刚达到 top_p 的那组 token，所以掩码要往右移一位
                sorted_indices_to_remove = cumulative_probs > top_p
                # 确保第一个 token 永远不被移除（防止空集）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将需要移除的 token 对应原始 logits 设置为负无穷
                # scatter 能够根据排序后的索引把掩码映射回原始位置
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # --- 采样 ---
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
                
    return idx

def main():
    # 读取模型
    cfg=InferConfig()
    checkpoint_path='checkpoints/run_20260124_183019/ckpt.pt'
    model = TransformerLM(vocab_size=cfg.vocab_size,context_length=cfg.context_length,d_model=cfg.d_model,num_layers=cfg.num_layers,num_heads=cfg.num_heads,d_ff=cfg.d_ff,rope_theta=cfg.rope_theta,device=cfg.device)
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)

    # tokenizer
    prompt='Once'
    tokenizer = AutoTokenizer.from_pretrained("gpt2",cache_dir='checkpoints/gpt2',local_files_only=True, use_fast=True)
    eos_id = tokenizer.eos_token_id
    encoded_token = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens_tensor = torch.tensor([encoded_token], dtype=torch.long, device=cfg.device)
    
    start_time = time.time()
    generated_indices = generate(
        model, 
        tokens_tensor, 
        max_new_tokens=cfg.max_new_tokens,
        context_length=cfg.context_length,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        eos_token_id=eos_id
    )
    end_time = time.time()

    # 5. 解码并打印结果
    output_text = tokenizer.decode(generated_indices[0], skip_special_tokens=False)
    
    print(f"\n\n{'='*50}\nGenerated Text:\n{'='*50}")
    print(output_text)
    print(f"\n{'='*50}")
    print(f"Time taken: {end_time - start_time:.2f}s")

if __name__=="__main__":
    main()