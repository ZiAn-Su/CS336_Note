import torch
from torch import Tensor
import math 
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
import warnings

def softmax(in_features: Tensor, dim: int) -> Tensor:
    max_ele=torch.max(in_features)
    in_features=in_features-max_ele
    exp_ele=torch.exp(in_features)
    sum_dim=exp_ele.sum(dim).unsqueeze(dim)
    return exp_ele/sum_dim

def scaled_dot_product_attention(Q:Tensor,K:Tensor,V:Tensor,mask:Tensor=None):
    '''
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    '''
    d_k=Q.shape[-1]
    qk=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        qk=qk.masked_fill(mask==0,-1e9)
    softmax_qk=softmax(qk,-1)
    return softmax_qk.matmul(V)

def convert_transformer_lm_weights(state_dict):
    """手动映射预训练权重"""
    new_state_dict = {}
    for key, value in state_dict.items():
        # 替换键名
        if key.startswith('token_embeddings.weight'):
            new_key = key.replace('token_embeddings.weight', 'embedding.weights')
            new_state_dict[new_key] = value
        elif key.startswith('layers.'):
            new_key = key.replace('layers.', 'tfm_block_list.')
            if 'q_proj.weight' in  new_key:
                new_key = new_key.replace('q_proj.weight', 'q_proj_weight.weights')
            elif 'k_proj.weight' in  new_key:
                new_key = new_key.replace('k_proj.weight', 'k_proj_weight.weights')
            elif 'v_proj.weight' in  new_key:
                new_key = new_key.replace('v_proj.weight', 'v_proj_weight.weights')
            elif 'output_proj.weight' in  new_key:
                new_key = new_key.replace('output_proj.weight', 'o_proj_weight.weights')
            
            elif 'ln1.weight' in  new_key:
                new_key = new_key.replace('ln1.weight', 'norm1.weights')
            elif 'ln2.weight' in  new_key:
                new_key = new_key.replace('ln2.weight', 'norm2.weights')

            elif 'ffn.w1.weight' in  new_key:
                new_key = new_key.replace('ffn.w1.weight', 'ffn.linear1.weights')
            elif 'ffn.w2.weight' in  new_key:
                new_key = new_key.replace('ffn.w2.weight', 'ffn.linear2.weights')
            elif 'ffn.w3.weight' in  new_key:
                new_key = new_key.replace('ffn.w3.weight', 'ffn.linear3.weights')
            new_state_dict[new_key] = value
        elif key.startswith('ln_final.weight'):
            new_key = key.replace('ln_final.weight', 'norm.weights')
            new_state_dict[new_key] = value
        elif key.startswith('lm_head.weight'):
            new_key = key.replace('lm_head.weight', 'linear.weights')
            new_state_dict[new_key] = value
    
    return new_state_dict

def cross_entropy(inputs:Tensor, targets:Tensor):
    max_ele=torch.max(inputs, dim=-1, keepdim=True)[0]
    in_features=inputs-max_ele
    exp_ele=torch.exp(in_features)
    sum_dim=exp_ele.sum(-1)
    tar_score=in_features[torch.arange(targets.shape[0]),targets]
    c_e=torch.log(sum_dim)-tar_score
    clean_data = torch.where(torch.isinf(c_e), torch.tensor(float('nan')), c_e)
    return clean_data.nanmean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
# 测试SGD
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,weight_decay=0.01,betas=(0.9, 0.999),eps=1e-8,):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,"weight_decay":weight_decay,"betas":betas,"eps":eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay=group["weight_decay"]
            (beta1,beta2)=group["betas"]
            eps=group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                h_m = state.get("m", torch.zeros_like(p)) # 过去的一阶矩
                h_v = state.get("v", torch.zeros_like(p)) # 过去的二阶矩
                t = state.get("t", 1) # 迭代次数
                
                m=beta1*h_m+(1-beta1)*grad
                v=beta2*h_v+(1-beta2)*torch.pow(grad,2)
                lr_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)

                p.data = p.data - lr*weight_decay*p.data  # 注意，此处与教材上不同，先进行权重衰减，后进行权重更新；参考了torch.optim.adamw.AdamW的说明
                p.data -= lr_t * m / (v.sqrt()+eps)                
                state["t"] = t + 1 
                state["m"] = m
                state["v"] = v
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif it<= cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    '''
    梯度裁剪，当所有梯度的二范数大于最大值时，按照最大值/二范数进行缩放
    '''
    # 计算二范数，判断是否大于max_l2_norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2范数
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5  # 开方得到总L2范数
    
    # 如果梯度范数超过最大值，按比例缩放
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    return

def get_batch(
    dataset, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 1. 随机生成 batch_size 个起始索引
    # 最高有效起始索引为 len(dataset) - context_length - 1
    ix = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))

    # 2. 使用列表推导式从数据集中提取输入 (x) 和标签 (y) 的数据块
    x_list = [dataset[i : i + context_length] for i in ix]
    y_list = [dataset[i + 1 : i + 1 + context_length] for i in ix]

    # 3. 将数据块列表堆叠成 NumPy 数组
    x_np = np.stack(x_list)
    y_np = np.stack(y_list)

    # 4. 将 NumPy 数组转换为 PyTorch 张量，并移动到指定设备
    #    注意：PyTorch 的 LongTensor 对应于 np.int64
    x = torch.from_numpy(x_np.astype(np.int64)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)

    return x, y

def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    
    # 1. 创建一个字典来保存所有需要的信息
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 如果需要，还可以添加其他信息，例如：
        # 'loss': last_loss,
        # 'scheduler_state_dict': scheduler.state_dict(),
    }

    # 2. 使用 torch.save 将字典保存到指定位置
    # torch.save 可以智能地处理文件路径或文件对象
    torch.save(checkpoint, out)

class CosineAnnealingScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int ,    # 预热步数
        cosine_cycle_iters: int ,  # cos步数
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose="deprecated",
    ):  # noqa: D107
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters=cosine_cycle_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""
        return [
            get_lr_cosine_schedule(self.last_epoch,base_lr,self.min_lr,self.warmup_iters,self.cosine_cycle_iters)
            for base_lr in self.base_lrs
        ]
