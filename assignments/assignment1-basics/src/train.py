import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 假设您的模型定义在 model.py 中
# from model import TransformerLM, ModelConfig

# --- 为了让这个脚本可以独立运行，我们先定义一个简单的占位模型 ---
# 请将其替换为您自己的模型实现
class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 一个简单的示例模型
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)
        print(f"Model initialized with vocab size: {config.vocab_size}")

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        tok_emb = self.embedding(idx) # (B, T, n_embd)
        logits = self.linear(tok_emb) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
# --- 占位模型结束 ---


# --- 1. 参数配置 (Configuration) ---
def get_args():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    # 数据和路径
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing train.bin and val.bin')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=f'run_{time.strftime("%Y%m%d_%H%M%S")}', help='A name for this training run')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocabulary size (GPT-2 has 50257, but we use a larger one for padding)')
    parser.add_argument('--block_size', type=int, default=1024, help='Context length')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')

    # 训练和优化器参数
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=600000, help='Total number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Max learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # LR Scheduler
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='Iterations for learning rate decay (usually same as max_iters)')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')

    # 日志和评估
    parser.add_argument('--eval_interval', type=int, default=2000, help='How often to evaluate')
    parser.add_argument('--log_interval', type=int, default=10, help='How often to log training status')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of batches for evaluation')
    
    # 运行环境
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    return parser.parse_args()

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
        out[split = losses.mean()
    model.train() # 重新设置为训练模式
    return out

def main():
    args = get_args()
    
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
    model_config = ModelConfig(
        vocab_size=args.vocab_size, 
        block_size=args.block_size, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd
    )
    model = TransformerLM(model_config)
    model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # --- 优化器和学习率调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
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

            if losses['val' < best_val_loss:
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
            lr = scheduler.get_last_lr()[0
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

        iter_num += 1
        X, Y = get_batch('train', data, args.block_size, args.batch_size, args.device) # 预取下一批数据

if __name__ == '__main__':
    main()