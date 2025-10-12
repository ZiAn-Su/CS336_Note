### 0913
在tiny数据集上训练tokenizer
    最小可行单元：并行读取文档、统计词频（预标记）、合并次数
    在一段中执行如下内容：
        统计字频
            移除特殊字符
        合并
            合并字典是可以提速的


### 2.5 BPE tokenzier训练实践
> 参考代码，将初始文档分块，用于后续并行处理；输入：文件、分段数、分隔符，实现：按照分段数计算参考分割位置，然后在每一段后找最近的分隔符作为真正的分割位置，输出：真正的分割位置
> https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py


对于tinystory 最大词汇表为10000 词汇表是什么类型呢


效果可以 速度太慢

