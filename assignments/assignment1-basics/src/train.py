import os
import time
import argparse
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformer import *
from utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 1. 参数配置 (Configuration) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表数量')
    parser.add_argument('--context_length', type=int, default=16, help='一次处理的最大token数')
    parser.add_argument('--d_model', type=int, default=64, help='特征维度（嵌入模型维度及其他层）')
    parser.add_argument('--num_layers', type=int, default=3, help='Transformer层的数量')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--d_ff', type=int, default=128, help='前馈神经网络的维度')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='rope旋转位置编码的theta值')

    # 训练和优化器参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=600000, help='总迭代次数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Max learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='一阶矩系数')
    parser.add_argument('--beta2', type=float, default=0.99, help='二阶矩系数')
    parser.add_argument('--eps', type=float, default=1e-8, help='小数防止数据无效')
    parser.add_argument('--max_norm', type=float, default=0.01, help='梯度裁剪的最大二范数')
    
    # LR Scheduler
    parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=100, help='the number of iterations to linearly warm-up')
    parser.add_argument('--cosine_cycle_iters', type=int, default=1000, help='the number of cosine annealing iterations')
    
    # 数据和路径
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing train.bin and val.bin')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=f'run_{time.strftime("%Y%m%d_%H%M%S")}', help='A name for this training run')

    # 日志和评估
    parser.add_argument('--eval_interval', type=int, default=2000, help='How often to evaluate')
    parser.add_argument('--log_interval', type=int, default=10, help='每隔多少步打印一次日志')
    parser.add_argument('--eval_iters', type=int, default=200, help='验证时跑多少个 batch 来估算 loss')
    parser.add_argument('--save_iters', type=int, default=10, help='保存间隔')
    
    # 运行环境
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', help='Data type to use for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    return parser.parse_args()

@dataclass
class TrainingConfig:
    # --- 模型参数 ---
    vocab_size: int = 10000      # 词汇表数量
    context_length: int = 16     # 一次处理的最大token数
    d_model: int = 64            # 特征维度（嵌入模型维度及其他层）
    num_layers: int = 3        # Transformer层的数量
    num_heads: int = 4       # 多头注意力头数
    d_ff: int = 128              # 前馈神经网络的维度
    rope_theta: float = 10000.0        # rope旋转位置编码的theta值

    # --- 训练和优化器参数 ---
    batch_size: int = 16        # 训练 batch size
    max_iters: int = 600000     # 总迭代次数
    learning_rate: float = 0.001 # max learning rate
    weight_decay: float = 0.01
    beta1: float = 0.9      # 一阶矩系数
    beta2: float = 0.99     # 二阶矩系数
    eps: float= 1e-8
    max_norm: float = 0.01      # 梯度裁剪的最大二范数
    
    # --- 学习率调度 ---
    min_lr: float = 0.0001      # 最小学习率 (通常是 lr 的 10%)
    warmup_iters: int = 100    # 预热步数
    cosine_cycle_iters: int = 1000   # cos步数
    
    # --- 数据和路径 ---
    data_dir: str = 'data' # 数据路径
    checkpoint_dir: str = 'checkpoints' # 检查点路径
    run_name: str = f'run_{time.strftime("%Y%m%d_%H%M%S")}' # 训练任务的名称

    # --- 日志和评估 ---
    eval_interval: int = 2000   # 每隔多少步验证一次
    log_interval: int = 100     # 每隔多少步打印一次日志
    eval_iters: int = 200       # 验证时跑多少个 batch 来估算 loss
    always_save_checkpoint: bool = True # 是否总是保存最好的模型

    # --- 运行环境 ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    resume: bool = False 

# --- 2. 高效数据加载 (Data Loading) ---
def get_batch(split, data, block_size, batch_size, device):
    d = data[split]
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(d[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if 'cuda' in device:
        # pin_memory makes transferring to GPU faster
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        return x.to(device), y.to(device)

# --- 3. 评估函数 (Evaluation Function) ---
@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters):
    out = {}
    model.eval() # 设置为评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 重新设置为训练模式
    return out

def main():
    args = get_args()
    args = TrainingConfig()
    # --- 设置 ---
    torch.manual_seed(1337)
    os.makedirs(os.path.join(args.checkpoint_dir, args.run_name), exist_ok=True)
    
    # --- 数据加载 ---
    print("Loading data...")
    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    data = {'train': train_data, 'val': val_data}
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    # --- 模型初始化 ---
    model = TransformerLM(vocab_size=args.vocab_size,context_length=args.context_length,d_model=args.d_model,num_layers=args.num_layers,num_heads=args.num_heads,d_ff=args.d_ff,rope_theta=args.rope_theta)
    model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # --- 优化器和学习率调度器 ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),eps=args.eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.lr_decay_iters, eta_min=args.min_lr)

    # --- Checkpoint 加载 ---
    checkpoint_path = os.path.join(args.checkpoint_dir, args.run_name, 'ckpt.pt')
    iter_num = 0
    best_val_loss = 1e9

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # --- 训练循环 ---
    X, Y = get_batch('train', data, args.block_size, args.batch_size, args.device) # 预取第一批数据
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # --- 评估和保存Checkpoint ---
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, data, args.block_size, args.batch_size, args.device, args.eval_iters)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"New best val loss: {best_val_loss:.4f}. Saving checkpoint...")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'args': args,
                }
                torch.save(checkpoint, checkpoint_path)

        # --- 前向和后向传播 ---
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # --- 梯度裁剪 ---
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # --- 更新学习率 ---
        scheduler.step()

        # --- 日志记录 ---
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lr = scheduler.get_last_lr()[0]
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

        iter_num += 1
        X, Y = get_batch('train', data, args.block_size, args.batch_size, args.device) # 预取下一批数据

if __name__ == '__main__':
    main()