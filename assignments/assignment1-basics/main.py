from tests.test_train_bpe import *
import pickle
import numpy as np
from src.bpe_tokenizer import BPETokenizer
from src.transformer import *
from src.utils import *
import torch
#from fvcore.nn import FlopCountAnalysis, parameter_count

def max_token(counts:dict[tuple,int]):
    '''
    返回value最大的元素的key，如果多个元素的value相同，返回key最大的那个元素
    '''
    max_keys=[]
    max_value=0
    for key,value in counts.items():
        if value>max_value:
            max_value=value
            max_keys=[key]
        elif value==max_value:
            max_keys.append(key)
    return max(max_keys)

def test_tokenizer():
    #test_train_bpe()
    # with open('tests/_snapshots/test_train_bpe_special_tokens.pkl','rb') as f:
    #     snapshot=pickle.load(f)
    # test_train_bpe_special_tokens(snapshot)

    with open('data/TinyStoriesV2-GPT4-train_vocab.pkl','rb') as f:
        vocab=pickle.load(f)
    with open('data/TinyStoriesV2-GPT4-train_merge.pkl','rb') as f:
        merge=pickle.load(f) 
    tokenizer=BPETokenizer(vocab,merge,['<|endoftext|>'])
    text='''
Home at a hotel <|endoftext|>

The hotel’s high-ceilinged lobby resembles that of a ski resort, with an elaborate chandelier, a grand piano and extensive wood paneling. It’s a busy crossroads of guests and long-term residents — plus thrifty diners lured by the $5 soul food-buffet in the adjacent restaurant. I first visited Hotel Louisville in June and recently returned for a five-day stay. After checking in at the front desk, I took a well-worn elevator to my room on the ninth floor, one of five levels open to the general public. It was a standard, budget-rate hotel room: clean and air-conditioned, with a TV and small desk free of flourish.

Cassie Lintz settles down for the night with her daughters Kendal, age 4, and Chloe, age 6, on right. Pat McDonogh for Al Jazeera America I’d soon learn that the residents’ quarters, on floors four, five, six and eleven, are homey and personalized, more like tiny apartments than hotel rooms sanitized with Smells BeGone. To date, between 96 and 162 people at a time — mostly women in recovery, but also some children and a few men — have lived in the hotel. Cassie Lintz and her daughters, preschooler Kendal and first-grader Chloe, know the place better than most. Several months ago, they moved in for a second time. “I had almost two years clean when I relapsed. So I came back here to do this again and get back on track,” Lintz said of her struggle with prescription painkillers.

She and the two round-faced, sandy-haired girls were sharing a room on the hotel’s family floor, home to several mostly single-parent units. Generally, substance-abuse centers require parents to come alone, leaving their children behind. Hotel Louisville is a rarity even among family-based recovery programs, giving clients with children a free, private room, with child care and activities included. Upstairs, on one of the singles floors, Yolanda Thomas wore her reading glasses to study the Bible in bed. She’d arrived at Hotel Louisville in July, having slept off her last high on a bus from Virginia. She was living in a tidy double with another “girl in the program,” splitting a bathroom, nightstand, TV and table. Thomas’s shoes — including several pairs of pointy high heels she has little occasion to wear — were lined up along the wall.

Recovering through work?

At early-morning meditation on a recent Friday, Thomas and about two dozen other women — in varying degrees of wakefulness, young and old, white and African-American — sat in a circle of chairs in the first-floor chapel. Everyone clutched a Bible and The Big Book from Alcoholics Anonymous. A few brought young children, who played in the back or snoozed in their strollers. (The day care wasn’t yet open.) Hotel manager Virginia Taylor, or “Miss V,” entered the room around 8 a.m. Short and stocky with freckled brown skin and oiled hair, she’s a former addict with 22 years clean and nearly as much peer-counseling experience. Taylor oversees Wayside’s recovery program and leads the gospel choir. In a thick Southern accent full of tapering R’s, she speaks with the air of a preacher or prison warden. “The disease does not promise you it’ll be easy. But you have a daily choice to stay sober. Choose to live,” she said. Taylor led the community meeting, a chance for residents to express grievances with the facility and with one another. Women clapped for every submission of “consequences” — 10,000-word reflections written on handfuls of notebook paper — the punishment for minor infractions, like missing a work shift or neglecting to sign out. The meeting concluded with a collective Lord’s Prayer, and the women then split up for work. Work therapy — performing housekeeping, food service, security and laundry for public hotel guests — is a key component of Wayside’s recovery program.

Recovery Manager Virginia Taylor, a former addict, leads early-morning meditation, part of the substance-abuse curriculum at Hotel Louisville. Pat McDonogh for Al Jazeera America “Many of these ladies didn’t know anything about housekeeping, but here, once you get to second phase (of recovery), they can choose to go to another hotel and get a job,” Cheri Hartwill, a recovering addict who leads the cleaning team, explained. “It prepares you — your work ethic, how you treat people — because a lot of us come from the streets.” Since the work is part of their recovery, and because lodging, food and other basic needs are provided free of charge, hotel residents are given an hourly “gift” of 50 cents to $1.50 per hour in lieu of pay. Residents sign “contracts acknowledging that they’re not employees or workers, but trainees,” said Wayside Chief Operating Officer Nina Moseley. The work assignments at Hotel Louisville are a more elaborate version of what Wayside and other charities have always asked of their residents. Since long before the hotel opened, Wayside has operated two donation-based Louisville thrift stores, with revenues supporting the organization’s central work. It based this model on that of larger groups like Goodwill and the Salvation Army, which require “beneficiaries” to unload, sort, stock and inventory donated goods in vast warehouses and shops. But Hotel Louisville residents are split on the value of work therapy. “It’s part of my penance,” Kim, a former nurse, said of her time in the hotel laundry room. Before coming to Wayside in September, she’d lost her home and car and had to sleep on the porch of an abandoned house. Others in the program say that their long hours and the strenuous nature of work therapy sometimes make it difficult to attend meetings and classes. One resident, who asked to remain anonymous, called these work assignments "unpaid labor," plain and simple. “This isn’t training. This is a business,” she said. “These are actual jobs you’re performing. You’re not doing it to strengthen yourself for a job outside. This is a job inside.” Karen Garnett, district director of the federal Department of Labor in Louisville, questions whether Hotel Louisville’s work-therapy program complies with U.S. labor laws. In some cases, such “trainees” are in fact employees entitled to the minimum wage and overtime, she said. “For the part of the hotel that’s a business entity, there may be some part that could be considered training, but it takes a minimum amount of training for some of this work.”

Manager Linda Stith went from being a homeless addict to helping run Hotel Louisville. Pat McDonogh for Al Jazeera America Wayside stands by its “training hotel” model, which its directors say teaches critical occupational and life skills to clients long removed from the labor force. “We have a very good rate of success for people who complete their programs,” Nina Moseley said. While Wayside does not track information on recovery participants (data collection has not been a priority), many graduates have gone on to work in local hotels, restaurants and hospitals. The hotel cannot afford to pay minimum wage to work-therapy participants, said Linda Stith, one of three general managers. The commercial side of the hotel has subsidized Wayside’s charitable programs since the summer of 2011. Last year, net proceeds of approximately $258,000 — from hotel guests, banquets and restaurant income — funded not only the residential recovery program but also Wayside’s other facility: a traditional homeless shelter and soup kitchen on nearby Jefferson Street. Altogether, the organization’s annual budget for homeless services and recovery is about $3.4 million, much of which comes from government contracts and private donations.

Business or charity or both
    '''
    tokens=tokenizer.encode(text)
    text1=tokenizer.decode(tokens)
    assert text==text1
    print(tokens)

def test_Linear():
    d_in=2
    d_out=3
    weights=torch.ones(3,2).float()
    in_features=torch.tensor([1,2]).float()
    linear_model=Linear(d_in,d_out)
    # 保存权重到文件
    #torch.save(linear_model.state_dict(), 'linear_weights.pth')
    #weights1=torch.load('linear_weights.pth')
    linear_model.load_state_dict({'weights':weights})
    result=linear_model.forward(in_features)
    return result

def test_swiglu():
    swiglu=FFNSwiGLU(2,8)
    torch.save(swiglu.state_dict(),'swiglu_weights.pth') 
    weights1=torch.load('swiglu_weights.pth')
    swiglu.load_state_dict(weights1)
    print('ok')
def test_rope():
    rope = RoPE(0.1,4,8)

def test_einsum():
    # 设置随机种子以便复现结果
    torch.manual_seed(42)
    
    # 创建简单的测试数据
    batch_size = 1
    seq_len = 1
    dim = 2
    
    # 创建输入张量 x_complex (bsd, num_heads, dim//2, 2)
    # 这里我们假设 dim=4，所以每个头的复数维度是 2
    x_complex = torch.randn(batch_size, seq_len,  dim//2, 2)
    print("输入张量 x_complex 形状:", x_complex.shape)
    print("x_complex[0,0,0]:", x_complex[0,0,0])
    
    # 创建旋转位置编码 rope_selected (bsd, num_heads, dim//2, 2, 2)
    # 旋转矩阵的形式为 [[cosθ, -sinθ], [sinθ, cosθ]]
    rope_selected = torch.zeros(batch_size, seq_len,  dim//2, 2, 2)
    
    # 为每个位置设置不同的旋转角度
    for b in range(batch_size):
        for s in range(seq_len):
            for d in range(dim//2):
                angle = (s + 1) * 0.1  # 简单的角度计算
                cos_theta = torch.cos(torch.tensor(angle))
                sin_theta = torch.sin(torch.tensor(angle))
                rope_selected[b, s, d] = torch.tensor([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
    
    print("旋转矩阵 rope_selected 形状:", rope_selected.shape)
    print("rope_selected[0,0,0,0]:", rope_selected[0,0,0,0])
    
    # 使用 einsum 计算旋转后的结果
    x_rotated = torch.einsum('bsdij,bsdj->bsdi', rope_selected, x_complex)
    print("旋转后结果 x_rotated 形状:", x_rotated.shape)
    print("x_rotated[0,0,0]:", x_rotated[0,0,0])
    
    # 手动验证第一个元素的计算
    print("\n=== 手动验证第一个元素的计算 ===")
    rope_mat = rope_selected[0,0,0]  # 形状 (2, 2)
    x_vec = x_complex[0,0,0]         # 形状 (2,)
    print("旋转矩阵:\n", rope_mat)
    print("输入向量:", x_vec)
    
    # 手动矩阵乘法
    manual_result = torch.matmul(rope_mat, x_vec)
    print("手动计算结果:", manual_result)
    print("einsum 计算结果:", x_rotated[0,0,0])
    
    # 验证两者是否相等
    print("结果是否一致:", torch.allclose(manual_result, x_rotated[0,0,0,0]))
    
    return x_rotated

def test_softmax():
    input = torch.randn(2,3,4)
    output = softmax(input,2)
    print('x')


def test_npz():
    # 加载 .npz 文件
    data = np.load('tests/_snapshots/test_multihead_self_attention_with_rope.npz')
    # 查看每个数组的详细信息
    for key in data.files:
        array = data[key]
        print(f"\n数组 '{key}':")
        print(f"  形状: {array.shape}")
        print(f"  数据类型: {array.dtype}")
        print(f"  数值范围: [{np.min(array):.6f}, {np.max(array):.6f}]")
        print(f"  前几个元素: {array.flatten()[:5]}")  # 显示前5个元素

    # 关闭文件
    data.close()

def test_count_para():
    vocab_size=50257
    context_length=1024
    d_model=1600
    num_layers=48
    num_heads=25
    d_ff=6400
    rope_theta=0.2
    model = TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # 只显示有参数的模块
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params

            print(f"{name:40} | {num_params:>10,} parameters")


def profile_with_torch_profiler(model, input):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        output = model(input)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return prof


def test_count_flops():
    vocab_size=50257
    context_length=1024
    d_model=1600
    num_layers=48
    num_heads=25
    d_ff=6400
    rope_theta=0.2
    model = TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta)
    
    # 使用PyTorch profiler
    input = torch.randint(1, 1000, (1, 12))
    prof = profile_with_torch_profiler(model, input)

    #input = torch.randint(1, 1000, (1, 12)).to(next(model.parameters()).device)
    flops = FlopCountAnalysis(model, input)
    params = parameter_count(model)

    print(f"FLOPs: {flops.total():,}")
    print(f"Parameters: {params['']:,}")

if __name__=="__main__":
    pred=torch.load('pred.pt')
    Y_train=torch.load('X_train.pt')
    print('ok')
    test_count_flops()
    test_count_para()
    # test_npz()
    # test_softmax()
    # test_einsum()
    # test_rope()
    # test_Linear()
    # test_swiglu()