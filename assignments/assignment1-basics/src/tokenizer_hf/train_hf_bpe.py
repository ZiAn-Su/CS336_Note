from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def train_custom_tokenizer(files):
    # 1. 初始化 BPE
    tokenizer = Tokenizer(models.BPE())
    
    # 2. 预处理 (按字节)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 3. 定义训练器
    trainer = trainers.BpeTrainer(
        vocab_size=32000, 
        special_tokens=["<|endoftext|>", "<pad>"]
    )
    
    # 4. 训练
    # 这一步非常快，几百 MB 的文本几秒钟就能训练完
    tokenizer.train(files, trainer)
    
    # 5. 保存
    tokenizer.save("data/my_custom_tokenizer.json")
    
    # 之后你可以用 AutoTokenizer.from_file("data/my_custom_tokenizer.json") 加载它