#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/10 18:05
 
class ModelConfig:
    out_dir = 'output_dir'  # out_dir是输出目录

    English_vocab_path='output_dir/English_vocab.pkl'  #词汇表目录


    # datav2_train='data/datav2.0/English_data_segment.csv'
    # datav2_test='data/datav2.0/Olid_testdata.csv'

    datav2_train='data/datav2.0/DV_augmentation.csv'
    datav2_test='data/datav2.0/Davision_test.csv'

    English_word_features='output_dir/English_word_features.pkl'
    Turkish_vocab_path='output_dir/Turkish_vocab.pkl'


    wordNgram=4  #提取字符级特征的长度
    #单词典
    sentence_length=128  #句子长度，因为添加了ngram特征，所以这个要大一些

    # #双词典
    # word_length=100
    # ngram_length=150

    emb_dim=300  #词向量维度300

    model_type='xiaorong1'

    algorithm='nadam'

    loss='categorical_crossentropy'  #  binary_crossentropy

    metric='accuracy'
    epochs=150 #有早停机制
    batch_size=16

