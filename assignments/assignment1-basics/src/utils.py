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