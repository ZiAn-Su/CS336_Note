## Linear
- 线性层：实现了y=xW-t的操作，其中x y为行向量；线性代数中y=Wx，xy为列向量。
- 权重初始化：参数使用随机的截断正态分布进行初始化，nn.init.trunc_normal_
![0](./resources/linear0.jpg)
- nn.Parameter是torch.Tensor子类，有一些特殊属性，比如默认梯度跟踪、参数能被一些函数识别，模型中需要训练用Parameter
- torch.matmul矩阵乘法 等价于@
- torch.empty创建空张量

## Embedding
- 嵌入层：实现了token到特征向量的转换，转换方式为查表，输入token，查找表为在vocab size x 特征向量维度的表中，查找第token个特征向量
- torch.reshape()将张量变形，比如reshape(-1)变为一维、reshape(4，12，1)
- torch.index_select 选择张量中的某些元素
- self.weights[token_ids]等价于self.weights.index_select(0,token_ids.reshape(-1)).reshape(*token_ids.shape, self.embedding_dim)
## RMSNorm
- 原始transformer使用post-norm，即归一化在残差MLP或多头注意力的后面，后续工作发现pre-norm提升了模型训练的稳定性，被现代模型作为标准采用，pre-norm即把归一化放在每一层的输入上，在所有block的后面再加一个归一化
<img src="./resources/rmsnorm.jpg" style="width: auto; height: 300;">
> Toan Q. Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. 2019    
> Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan,
Liwei Wang, and Tie-Yan Liu. On layer normalization in the Transformer architecture.2020
- 原始transformer使用layer normalization，参考2023年Llama的工作，使用RMS norm(Root mean square layer normalization)
> Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016.    
> Hugo Touvron, Thibaut Lavril, ..., Llama: Open and efficient foundation language models, 2023.    
> Biao Zhang and Rico Sennrich. Root mean square layer normalization.2019