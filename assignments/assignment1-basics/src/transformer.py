import torch
import torch.nn as nn
from torch import Tensor
import math
class Linear(nn.Module):
    def __init__(self,in_features: int,out_features: int,device=None,dtype=None,):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weights=nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weights,0,std,-3*std,3*std)
    
    def forward(self, input: Tensor) -> Tensor:
        # 等价于input.matmul(self.weights.T)
        return input@self.weights.T

class Embedding(nn.Module):   
    def __init__(self,num_embeddings: int,embedding_dim: int,device=None,dtype=None,):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weights=nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weights,0,1,-3,3)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        # return self.weights[token_ids] 或：
        return self.weights.index_select(0,token_ids.reshape(-1)).reshape(*token_ids.shape, self.embedding_dim)
    
class RMSNorm(nn.Module):
    def __init__(self,d_model: int,eps: float=1e-5,device=None,dtype=None,):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weights=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    
    def forward(self, input: Tensor) -> Tensor:
        in_dtype = input.dtype
        input = input.to(torch.float32)
        deno=torch.sqrt(torch.sum(input**2,dim=2)/self.d_model+self.eps).unsqueeze(-1)
        result=input.div(deno).mul(self.weights)
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.linear1=Linear(d_model,d_ff,device,dtype)
        self.linear2=Linear(d_ff,d_model,device,dtype)
        self.linear3=Linear(d_model,d_ff,device,dtype)
    def forward(self,input: Tensor):
        ret1=self.linear1.forward(input)
        silu_t=ret1.mul(torch.sigmoid(ret1))
        ret2=self.linear3.forward(input)
        return self.linear2.forward(silu_t.mul(ret2))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        seq_l = torch.arange(max_seq_len,device=device, dtype=torch.float32)  # shape: [n]
        d_l = torch.arange(0,d_k,2, device=device,dtype=torch.float32)  # shape: [m]
        ret = d_l/d_k
        power_result = theta ** ret
        # 使用广播计算所有组合的角度：L1[i] / L2[j]
        angles = seq_l.unsqueeze(1) / power_result.unsqueeze(0)  # shape: [n, m]
        
        # 计算三角函数
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        
        # 构建矩阵序列 [n, m, 2, 2]
        matrices = torch.stack([
            torch.stack([cos_vals,-sin_vals], dim=-1),
            torch.stack([sin_vals, cos_vals], dim=-1)
        ], dim=-2)

        self.register_buffer('rope',matrices,persistent=False)
    def forward(self,x: torch.Tensor, token_positions: torch.Tensor):

        batch_size, seq_len, d_k = x.shape
        
        # 重塑x为复数形式 [batch_size, seq_len, d_k/2, 2]
        x_complex = x.reshape(batch_size, seq_len, d_k//2, 2)
        
        # 选择对应位置的旋转矩阵
        rope_selected = self.rope[token_positions]  # [seq_len, d_k/2, 2, 2]
        
        # 为batch维度添加广播维度
        rope_selected = rope_selected.unsqueeze(0)  # [1, seq_len, d_k/2, 2, 2]

        # 应用旋转：矩阵乘法
        # 需要调整维度以进行批量矩阵乘法
        x_rotated = torch.einsum('bsdij,bsdj->bsdi', rope_selected, x_complex)
        
        # 重塑回原始形状
        return x_rotated.reshape(batch_size, seq_len, d_k)
