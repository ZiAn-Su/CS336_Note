import os
import time
import math
import pickle
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from transformer import *
from utils import *
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

def train(model, optimizer, config):
    # 1. 准备工作
    os.makedirs(config.out_dir, exist_ok=True)
    model.to(config.device)
    
    # 用于混合精度训练的 Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # 记录状态
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()
    
    print(f"Starting training on {config.device}...")
    
    while iter_num < config.max_iters:
        
        # --- A. 动态调整学习率 ---
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # --- B. 获取数据 ---
        X, Y = get_batch('train', config)
        
        # --- C. 前向传播 (Forward) 与 Loss 计算 ---
        # 使用混合精度上下文 (如果支持)
        with torch.cuda.amp.autocast(dtype=torch.float16 if config.dtype == 'float16' else torch.bfloat16):
            logits, loss = model(X, Y)
        
        # --- D. 反向传播 (Backward) ---
        # 初始化梯度为 None (比 zero_grad() 稍微高效一点)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # --- E. 梯度裁剪 (Gradient Clipping) ---
        scaler.unscale_(optimizer) # 裁剪前必须 unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # --- F. 优化器步进 (Step) ---
        scaler.step(optimizer)
        scaler.update()
        
        # --- G. 日志记录 (Logging) ---
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
            
        # --- H. 验证与 Checkpoint 保存 ---
        if iter_num > 0 and (iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1):
            losses = estimate_loss(model, config)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': config, # 保存配置以便稍后恢复模型结构
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    save_path = os.path.join(config.out_dir, 'ckpt.pt')
                    print(f"Saving checkpoint to {save_path}")
                    torch.save(checkpoint, save_path)
        
        iter_num += 1

    print("Training finished!")


if __name__ == '__main__':
    # 1. 实例化配置
    cfg = TrainingConfig()
    
    # 2. 实例化模型
    model = TransformerLM(vocab_size=cfg.vocab_size,context_length=cfg.context_length,d_model=cfg.d_model,num_layers=cfg.num_layers,num_heads=cfg.num_heads,d_ff=cfg.d_ff,rope_theta=cfg.rope_theta)
    
    # 3. 实例化 Optimizer (AdamW)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2),eps=cfg.eps)

    # 4. 检查是否需要从 Checkpoint 恢复 (Resume)
    if os.path.exists(os.path.join(cfg.out_dir, 'ckpt.pt')):
        print("Found checkpoint, resuming...")
        iter_num = load_checkpoint(os.path.join(cfg.out_dir, 'ckpt.pt'),model,optimizer)
   
    # 5. 开始训练
    # 为了演示，你需要先生成假的 train.bin 和 val.bin
    if not os.path.exists(cfg.data_dir): os.makedirs(cfg.data_dir)
    # 创建假数据供测试
    with open(os.path.join(cfg.data_dir, 'train.bin'), 'wb') as f:
        np.array([1]*(cfg.block_size*100), dtype=np.uint16).tofile(f)
    with open(os.path.join(cfg.data_dir, 'val.bin'), 'wb') as f:
        np.array([1]*(cfg.block_size*50), dtype=np.uint16).tofile(f)

    try:
        train(model, optimizer, cfg)
    except KeyboardInterrupt:
        print("Training interrupted manually.")