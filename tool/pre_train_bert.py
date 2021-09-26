#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/6/9 18:11
import transformers
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from keras.utils import to_categorical
from tools import data_reader



#-------------------------------------------------------------------------------
def english_segment_text(texts):
    from nltk import word_tokenize #分词
    filtered_tweets = []
    str=''
    for tweet in texts["tweet"]:
        tweet_tokens = word_tokenize(tweet)  # 分词
        for w in tweet_tokens:
            if w != 'url' and w != 'user' and w != '@' and w != ',' and w != "'" and w != '.' and w != '#' :
                str=str+' '+w
        filtered_tweets.append(str)
        str=''
    return filtered_tweets
#-------------------------------------------------------------------------------
def get_data_bert(args):
    from transformers import BertTokenizer
    path='/data1/yangyong/junjie/offensive/Multilingual/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(path)

    # 这是训练集
    English_train_data = pd.read_csv(args.datav2_train)
    English_test = pd.read_csv(args.datav2_test)

    # English_train_data = pd.read_csv(args.English_train_data,sep='\t')
    # English_test = pd.read_csv(args.English_test,sep='\t')

    English_train_data.tweet = English_train_data.tweet.str.lower()  # 小写化
    English_test.tweet = English_test.tweet.str.lower()  # 测试集也小写化

    English_train_data.tweet = data_reader.EmojiToText(English_train_data.tweet)  # 将表情文本化
    English_test.tweet = data_reader.EmojiToText(English_test.tweet)  # 测试集也同样如此

    filtered_train_tweets=english_segment_text(English_train_data)
    filtered_test_tweets=english_segment_text(English_test)

    encoded_inputs=tokenizer(filtered_train_tweets, return_tensors='tf',padding=True,truncation=True,max_length=128)
    encoded_test=tokenizer(filtered_test_tweets, return_tensors='tf',padding=True,truncation=True,max_length=128)

    input_train_ids=encoded_inputs.get("input_ids")
    input_train_mask=encoded_inputs.get("attention_mask")
    input_train_tokentype=encoded_inputs.get("token_type_ids")

    input_test_ids=encoded_test.get("input_ids")
    input_test_mask=encoded_test.get("attention_mask")
    input_test_tokentype=encoded_test.get("token_type_ids")

    return input_train_ids,input_train_mask,input_train_tokentype,input_test_ids,input_test_mask,input_test_tokentype


#-------------------------------------------------------------------------------


# -------------------------------------------------------------------------------


def english_segment_text(texts):
    from nltk import word_tokenize #分词
    filtered_tweets = []
    str=''
    for tweet in texts["tweet"]:
        tweet_tokens = word_tokenize(tweet)  # 分词
        for w in tweet_tokens:
            if w != 'url' and w != 'user' and w != '@' and w != ',' and w != "'" and w != '.' and w != '#' :
                str=str+' '+w
        filtered_tweets.append(str)
        str=''
    return filtered_tweets

if __name__ == '__main__':
    from tools import config
    config=config.ModelConfig()
    get_data_bert(config)