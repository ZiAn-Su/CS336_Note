from tests.test_train_bpe import *
import pickle

from src.bpe_tokenizer import BPETokenizer
from src.transformer import Linear
import torch
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




if __name__=="__main__":
    test_Linear()