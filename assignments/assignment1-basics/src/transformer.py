import torch
import torch.nn as nn
from torch import Tensor
import math

nn.Linear
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
