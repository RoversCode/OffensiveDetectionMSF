#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/11 13:51
import logging,sys
import pandas as pd
import pickle as pk


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

BOW='<'
COW='>'
#---------------------------------------------------------------#

#---------------------------------------------------------------#
#加载词典
def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab
#---------------------------------------------------------------#
def load_features(features_path):
    with open(features_path, 'rb') as vocab_file:
        features = pk.load(vocab_file)
    return features

#---------------------------------------------------------------#
'''
参数：word对应词
    ngram_feature用来装word的ngram特征
    dictionary装ngram的词典
    wordngram: ngram的取值
'''
def computer_subwords(word:str,ngram_feaure:list,dictionary=None,wordngram=4):
    #先把原本的单词赋给ngram_feaure
    ngram_feaure.append(word)
    for i in range(len(word)):
        ngram='' #清空
        if (ord(word[i]) & 0xC0)==0x80:
            continue
        j = i  #python的for不能加条件真是太麻烦了
        for n in range(1,len(word)+1):
            if j >= len(word):
                break
            ngram=ngram+word[j]
            j=j+1
            while(j<len(word) and (ord(word[i]) & 0xC0)==0x80):
                ngram=ngram+word[j]
                j+=1
            if(n==wordngram and not(n==1 and (i==0 or j==len(word)))):   #原样是n>=minx
                if dictionary==None:
                    ngram_feaure.append(ngram)
                else:
                    try:
                        dictionary[ngram]+=1
                    except KeyError:
                        dictionary[ngram] =1
                    ngram_feaure.append(ngram)


#---------------------------------------------------------------#
#将表情转换为text
def EmojiToText(texts):
    import emoji
    filtered_tweets=[]
    for tweet in texts:
        filtered_sentences=emoji.demojize(tweet)
        # print(filtered_sentences)
        filtered_tweets.append(filtered_sentences)

    return  filtered_tweets

#---------------------------------------------------------------#
#找包含word ngram特征对应的list
def look_feature(word,word_features):
    for i in range(len(word_features)):
        if word==word_features[i][0]:
            break
    return word_features[i]
#---------------------------------------------------------------#

#分词
def english_segment_text(texts,way):
    from nltk import word_tokenize #分词
    from nltk.corpus import stopwords #停止词
    from krovetzstemmer import Stemmer
    print("Lowercase operation is done ! ")
    if way==1:
        # For train data
        filtered_tweets = []
        for tweet in texts["tweet"]:
            tweet_tokens = word_tokenize(tweet)  # 分词

            filtered_sentence = [w for w in tweet_tokens if (
                        w.isalpha() == True and w != 'url' and w != 'user' and w != '@' and w != ',' and w != "'" and w != '.' and w != '#' and w != '?')]
            filtered_tweets.append(filtered_sentence)

        texts["tweet_initial"] = filtered_tweets
    elif way==2:
        # 2-Stop word removal
        print("Stop word  and punctuation removal begins !")
        stop_words = set(stopwords.words('english'))
        filtered_tweets = []
        for tweet in texts["tweet"]:
            tweet_tokens = word_tokenize(tweet)

            filtered_sentence = [w for w in tweet_tokens if ((not w in stop_words) and w.isalpha() == True and w != 'url' and w != 'user' and w != '@' and w != ',' and w != "'" and w != '.' and w != '#' and w != '?')]

            filtered_tweets.append(filtered_sentence)
        texts["tweet_after_stopword"] = filtered_tweets
        print("Stop word and punctuation removal is done !")

    #词干
    stemmer = Stemmer()
    stemmer.stem('utilities')  # got: 'utility'
    # For stemming tranining data
    dum = filtered_tweets
    stemmed_tweets = []
    for tweet in dum:
        tweet_arr = []
        for word in tweet:
            # print(stemmer.stem(word))
            tweet_arr.append(stemmer.stem(word))
        stemmed_tweets.append(tweet_arr)
    texts["tweet_after_stemmed"] = stemmed_tweets

    print("Stemming Done")
    print("Reprocessing Done!")
    return texts

#---------------------------------------------------------------#

#创建词汇表
def create_vocab(texts,ngram=4):
    logger.info('Creating vocabulary.........')
    total_words, unique_words = 0, 0

    dictionary = {}  #词典
    word_feature=[]
    ngram_feature = []

    # #vocab.txt是bert的词典文件
    # with open("data/vocab.txt", "r", encoding='utf-8') as f:
    #     for line in f.readlines():
    #         word = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         try:
    #             dictionary[word] += 1
    #         except KeyError:
    #             unique_words += 1
    #             dictionary[word] = 1
    #             #提取词的ngram特征
    #             computer_subwords(word, ngram_feature, dictionary,ngram)
    #             if len(ngram_feature)==2:
    #                 word_feature.append(ngram_feature[0])
    #             else:
    #                 word_feature.append(ngram_feature)
    #             ngram_feature=[]
    #     total_words += 1

    # 这个并不计算word的ngram特征

    for i in range(len(texts)):
        sentence=texts[i]
        for word in sentence:
            try:
                dictionary[word] +=1
                ngram_feature=[]
            except KeyError: #如果在sentences中的词，词典中没有，就发生这个异常
                unique_words +=1
                dictionary[word]=1
                #提取词的ngram特征
                computer_subwords(word, ngram_feature, dictionary,ngram)
                if len(ngram_feature)==2:
                    word_feature.append(ngram_feature[0])
                else:
                    word_feature.append(ngram_feature)
                word_feature.append(ngram_feature)
                ngram_feature=[]
            total_words +=1  #这个并不计算word的ngram特征

    logger.info('  %i total words, %i unique words' % (total_words, unique_words))  #这个信息并没有计算ngram

    import operator
    #对词典进行了排序
    sorted_word_freqs = sorted(list(dictionary.items()), key=operator.itemgetter(1), reverse=True)

    #统计词汇数
    vocab_size=0

    for word,freq in sorted_word_freqs:
        if freq >=1:
            vocab_size +=1

    vocab={'<pad>':0,'<unk>':1,'<word>':2,'<noword>':3}

    vcb_len=len(vocab)
    index=vcb_len

    for word,_ in sorted_word_freqs[:vocab_size]:
        vocab[word] = index
        index += 1

    return vocab,word_feature #返回词典

#---------------------------------------------------------------#
#创造双词典
def create_two_vocab(texts,ngram):
    import operator
    logger.info('Creating Two vocabulary.........')
    total_words, unique_words = 0, 0
    word_dictionary = {}  #词典
    feature_dictionary = {}  #ngram特征词典
    word_feature=[]
    ngram_feature = []
    for i in range(len(texts)):
        sentence = texts[i]
        for word in sentence: #取到词
            # word=BOW+word+COW  #双词典就不要这个了
            try:
                word_dictionary[word] +=1
                #即便是词已经有了，也得把词频加上去，所以越高频的词，暗含的信息量越少
                computer_subwords(word,ngram_feature,feature_dictionary,ngram)
                ngram_feature=[]
            except KeyError: ##如果在sentences中的词，词典中没有，就发生这个异常
                unique_words +=1
                word_dictionary[word]=1
                #提取词的ngram特征
                computer_subwords(word, ngram_feature, feature_dictionary,ngram)
                word_feature.append(ngram_feature)
                ngram_feature=[]
            total_words +=1  #这个并不计算word的ngram特征
    #看独特词占据总词量的大小
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))  #这个信息并没有计算ngram
    #对词典进行了排序
    sorted_word_freqs = sorted(list(word_dictionary.items()), key=operator.itemgetter(1), reverse=True)
    sorted_ngram_freqs=sorted(list(feature_dictionary.items()), key=operator.itemgetter(1), reverse=True)

    word_size=0 #词典大小
    ngram_size=0
    for word, freq in sorted_word_freqs:
        if freq >= 1:
            word_size += 1
    for word, freq in sorted_ngram_freqs:
        if freq >= 1:
            ngram_size += 1


    word_vocab={'<pad>':0,'<unk>':1}
    vcb_len=len(word_vocab)
    index=vcb_len
    for word,_ in sorted_word_freqs[:word_size]:
        word_vocab[word] = index
        index += 1

    ngram_vocab={'<pad>':0,'<unk>':1}
    vcb_len=len(ngram_vocab)
    index=vcb_len

    for word, _ in sorted_ngram_freqs[:ngram_size]:
        ngram_vocab[word] = index
        index += 1

    return word_vocab,ngram_vocab,word_feature #返回词典和特征
#得到词索引
def get_indices(texts, vocab,word_features,wordngram=4):

    # #读取一下脏词表
    # with open('data/DirtyWordDict/word_all.txt', 'r') as f:  # 得到词表
    #     word_list = f.read().split('\n')

    # word_list = [s.lower() for s in word_list]  #小写化


    data_x=[]
    unk_hit,total=0.,0.

    for sentence in texts:
        indice=[]
        feature=[]
        for word in sentence:
            # word=BOW+word+COW  #添加<>
            # if word in word_list: #脏词2  不是脏词3
            #     category=[2]
            # else:
            #     category=[3]
            if word in vocab:
                #这里应该寻找下word对应word_features的位置,返回对应的list
                feature=look_feature(word,word_features)
                for j in feature:
                    indice.append(vocab[j]) #把句子中的每一个token(包含ngram特征)，映射成索引
                feature=[]
            else:
                #如果这个词不在词典当中,那就计算这个词的subwords
                computer_subwords(word,feature,vocab,wordngram)  #此时不应放入词典了。得到该陌生词的ngram
                for j in range(1,len(feature)):
                    if feature[j] in vocab:
                        indice.append(vocab[feature[j]]) #把句子中的每一个token(包含ngram特征)，映射成索引
                    else:#没有就用unk替代
                        indice.append(vocab['<unk>'])
                feature=[]
                unk_hit +=1  #统计找不到词找不到索引的次数
            total +=1

        #最后indice要加上是否有不良言论的标志
        # indice+=category

        data_x.append(indice) #把映射完的sentence，都赋给data_x

    #输出词找不到索引的概率
    logger.info('<unk> hit rate: %.2f%%' % (100 * unk_hit / total))  # 输出找不到词的几率

    return data_x

#---------------------------------------------------------------#
#得到双索引
def get_two_indices(texts,word_vocab,ngram_vocab,word_featurs,wordngram=4):
    word_x=[]
    ngram_x=[]
    unk_hit,total=0.,0.
    ngram=[]
    for sentence in texts:
        indice=[] #word
        feature=[] #ngram
        for word in sentence:
            if word in word_vocab: #词在词典中
                indice.append(word_vocab[word])
                ngram=look_feature(word,word_featurs)
                for j in range(1,len(ngram)):
                    feature.append(ngram_vocab[ngram[j]]) #把句子中的每一个token(包含ngram特征)，映射成索引
                ngram=[]
            else: #词不在词典中
                indice.append(word_vocab['<unk>'])
                computer_subwords(word,ngram,wordngram=wordngram)  # 此时不应放入词典了。得到该陌生词的ngram
                for j in range(1,len(ngram)):
                    if ngram[j] in ngram_vocab:
                        feature.append(ngram_vocab[ngram[j]]) #把句子中的每一个token(包含ngram特征)，映射成索引
                    else:#没有就用unk替代
                        feature.append(ngram_vocab['<unk>'])
                ngram = []
                unk_hit+=1
            total+=1
        word_x.append(indice)
        ngram_x.append(feature)
    #输出词找不到索引的概率
    logger.info('<unk> hit rate: %.2f%%' % (100 * unk_hit / total))  # 输出找不到词的几率
    return word_x,ngram_x

#---------------------------------------------------------------#
from keras.preprocessing import sequence

def read_dataset(args,num_vocab=1):
    from keras.utils import to_categorical
    import os
    # #这是训练集
    # English_train_data=pd.read_csv(args.English_train_data,sep='\t')
    # English_test=pd.read_csv(args.English_test,sep='\t')
    # English_test_label=pd.read_csv(args.English_test_label)
    
    # 将标注转变为0和1  1为OFF ，0为NOT
    # y_train = [1 if i == "OFF" else 0 for i in English_train_data["subtask_a"]]
    # y_test = [1 if i == "OFF" else 0 for i in English_test_label['label']]

    English_train_data=pd.read_csv(args.datav2_train)  #数据集v2.0
    English_test=pd.read_csv(args.datav2_test)

    y_train = English_train_data["label"]
    y_test = English_test['label']


    # #测试
    # y_dirty=y_train

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    English_train_data.tweet=English_train_data.tweet.str.lower() #小写化
    English_test.tweet=English_test.tweet.str.lower() #测试集也小写化

    English_train_data.tweet=EmojiToText(English_train_data.tweet) #将表情文本化
    English_test.tweet=EmojiToText(English_test.tweet)   #测试集也同样如此

    English_train_data=english_segment_text(English_train_data,1)#三种处理方式:tweet_initial 1, stopword 2 ，提取词干一定有
    English_test=english_segment_text(English_test,1)

    train_data=English_train_data.tweet_after_stemmed #这里选用词干
    test_data = English_test.tweet_after_stemmed

    #检测要单词典还是双词典
    if num_vocab==1:
        # 检查是否已经有了相应得词汇表
        if not os.path.exists(args.English_vocab_path):
            #这里指定哪种预处理方式进行创建词典

            English_vocab,word_features = create_vocab(train_data,ngram=args.wordNgram)

            #Dump vocab 将词汇表以二进制流的方式，存入vocab.pkl
            if not os.path.exists(args.English_vocab_path):
                with open(args.out_dir + '/English_vocab.pkl', 'wb') as vocab_file:
                    pk.dump(English_vocab, vocab_file)
            #将word_features存储
            if not os.path.exists(args.English_word_features):
                with open(args.out_dir + '/English_word_features.pkl', 'wb') as vocab:
                    pk.dump(word_features, vocab)

        else:
            English_vocab = load_vocab(args.English_vocab_path)
            word_features=load_features(args.English_word_features)


        X_train = get_indices(train_data, English_vocab, word_features,wordngram=args.wordNgram)  # 得到词、字符的词频
        X_test = get_indices(test_data, English_vocab, word_features,wordngram=args.wordNgram)

        # 对序列进行截断和补充，默认从开头开始，这里从后边开始截断或者补充，补充值为0
        X_train = sequence.pad_sequences(X_train, padding='post', maxlen=args.sentence_length)
        X_test = sequence.pad_sequences(X_test, padding='post', maxlen=args.sentence_length)


        return X_train, X_test, y_train, y_test, len(English_vocab)

    else:#双词典
        # 检查是否已经有了相应得词汇表
        if not os.path.exists(args.English_vocab_path):
            word_vocab,ngram_vocab,word_features=create_two_vocab(train_data,ngram=args.wordNgram)
            # Dump vocab 将词汇表以二进制流的方式，存入vocab.pkl
            if not os.path.exists(args.English_vocab_path):
                with open(args.out_dir + '/English_vocab.pkl', 'wb') as vocab_file:
                    pk.dump(word_vocab, vocab_file)
                with open(args.out_dir + '/ngram_vocab.pkl', 'wb') as vocab_file:
                    pk.dump(ngram_vocab, vocab_file)
            #将word_features存储
            if not os.path.exists(args.English_word_features):
                with open(args.out_dir + '/English_word_features.pkl', 'wb') as vocab_file:
                    pk.dump(word_features, vocab_file)
        else:
            word_vocab = load_vocab(args.English_vocab_path)
            ngram_vocab=load_vocab('output_dir/ngram_vocab.pkl')
            word_features=load_features(args.English_word_features)

        # 在这里做出决断要用原始处理的哪种方式
        train_word,train_ngram = get_two_indices(train_data,word_vocab,ngram_vocab,word_features,args.wordNgram)
        test_word, test_ngram = get_two_indices(test_data, word_vocab, ngram_vocab, word_features,
                                                  args.wordNgram)

        # 对序列进行截断和补充，默认从开头开始，这里从后边开始截断或者补充，补充值为0
        X_train_word = sequence.pad_sequences(train_word, padding='post', maxlen=args.word_length)
        X_train_ngram = sequence.pad_sequences(train_ngram, padding='post', maxlen=args.ngram_length)
        X_test_word = sequence.pad_sequences(test_word, padding='post', maxlen=args.word_length)
        X_test_ngram = sequence.pad_sequences(test_ngram, padding='post', maxlen=args.ngram_length)

        return X_train_word,X_train_ngram,X_test_word,X_test_ngram,y_train,y_test,word_vocab,ngram_vocab

#---------------------------------------------------------------#

#两个参数，一个需要得语言数量，默认为1，一个是模型得参数类，最后一个是数据预处理出来是要两个词典还是一个词典
def get_data(args,num_vocab=1):

    if num_vocab==1:
        X_train, X_test, y_train, y_test, vocab_length= read_dataset(args,num_vocab)
        return X_train, X_test, y_train, y_test, vocab_length
    else:
        X_train_word, X_train_ngram, X_test_word, X_test_ngram, y_train, y_test, word_vocab, ngram_vocab=read_dataset(args,args.sentence_length,num_vocab)
        return X_train_word, X_train_ngram, X_test_word, X_test_ngram, y_train, y_test, word_vocab, ngram_vocab



def get_y(args):
    from tensorflow.keras.utils import to_categorical
    English_train_data = pd.read_csv(args.datav2_train)  # 数据集v2.0
    English_test = pd.read_csv(args.datav2_test)

    y_train = English_train_data["label"]
    y_test = English_test['label']

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return y_train,y_test

    

def xiaorong(args):
    y_train,y_test=get_y(args)
    return y_train,y_test
