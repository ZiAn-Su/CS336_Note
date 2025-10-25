import torch
from torch import Tensor
import math 
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
