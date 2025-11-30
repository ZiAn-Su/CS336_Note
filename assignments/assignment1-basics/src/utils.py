import torch
from torch import Tensor
import math 
from collections.abc import Callable, Iterable
from typing import Optional

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