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