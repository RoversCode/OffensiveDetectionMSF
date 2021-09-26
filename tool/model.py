#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/10 18:05
import numpy as np
import logging
logger = logging.getLogger(__name__)
from tensorflow.keras import backend as K
from tensorflow.keras import Model
def expand_dim(x): #变成列向量
	return K.expand_dims(x, 1)

def create_model(config, num_class, overal_maxlen, word_vocab,ngram_vocab=None):
    from keras.layers import Input,Dense,Embedding,GlobalAveragePooling1D,GlobalMaxPooling1D,Concatenate,Convolution1D,Dropout,MaxPool1D\
        ,LSTM,LeakyReLU,concatenate,Flatten
    from tools.mylayers import SelfAttention,MultiHeadAttention,GateModule_word
    #还是用if把，以免要搭多个模型测试
    if config.model_type=='mymodel':  #词级和字符级参杂在一块
        logger.info('Building a mymodel Model')

        word_input=Input(shape=(config.sentence_length,),dtype='int32')
        category_input=Input(shape=(100,),dtype='int32')

        word_embedding=Embedding(input_dim=len(word_vocab),output_dim=config.emb_dim)(word_input)
        category_embedding=Embedding(input_dim=len(word_vocab),output_dim=100)(category_input)

        emb_output=concatenate([word_embedding,category_embedding],axis=-1)

        merge=Dense(400)(emb_output) #融合
        # attention=MultiHeadAttention(4,100)(merge)
        #
        # forward=Dense(400,activation='relu')(attention)
        #
        # attention=MultiHeadAttention(4,100)(forward)
        # forward = Dense(400, activation='relu')(attention)


        #卷积
        conv=Convolution1D(filters=128,kernel_size=5,activation='relu')(merge)
        conv=Convolution1D(filters=128,kernel_size=3,activation='relu')(conv)#94,128

        #池化
        pool=MaxPool1D(pool_size=3,strides=2)(conv) #47,128

        conv=Convolution1D(filters=64,kernel_size=3,activation='relu')(pool) #45*64

        pool=MaxPool1D(pool_size=3,strides=2)(conv) #23*64

        conv = Convolution1D(filters=16, kernel_size=3, activation='relu')(pool)  # 21*16

        pool=MaxPool1D(pool_size=2,strides=2)(conv) #10*16
        flat=Flatten()(pool)
        x=Dense(128,activation='relu')(flat)

        #全连接层
        #输出层
        predictions=Dense(units=num_class,activation='softmax')(x)
        #Bulid and compile model
        model = Model(inputs=[word_input,category_input], outputs=predictions)


        print("不良言论检测模型建立: ")
        model.summary()
    elif config.model_type=='model2':
        logger.info('Building a mymodel Model')
        input=Input(shape=(overal_maxlen,),dtype='int32')  #(200,)
        embedding1=Embedding(len(word_vocab),config.emb_dim)(input) #(  batch,200,300)
        embedding1=Dropout(0.2)(embedding1)
        conv=Convolution1D(128,5,activation='relu')(embedding1)
        pool_max=MaxPool1D(pool_size=4)(conv)
        conv=Convolution1D(128,2,activation='relu')(pool_max)
        pool_max=GlobalMaxPooling1D()(conv)
        # lstm=LSTM(100,dropout=0.2,recurrent_dropout=0.2)(pool_max)
        x=Dense(100,activation='relu')(pool_max)
        #输出层
        predictions=Dense(units=num_class,activation='softmax')(x)
        #Bulid and compile model
        model = Model(inputs=input, outputs=predictions)
        print("不良言论检测模型建立: ")
        model.summary()
    elif config.model_type=='model3':

        logger.info('Building a model3 Model')
        input1 = Input(shape=(config.word_length,), dtype='int32')  # (60,)
        embedding1 = Embedding(len(word_vocab), config.emb_dim)(input1)  # (batch,60,300)
        attention_out= MultiHeadAttention(3,100)(embedding1)
        forward=Dense(300,kernel_initializer="he_normal")(attention_out)
        forward=LeakyReLU(alpha=0.2)(forward)

        forward=MultiHeadAttention(3,100)(forward)

        pool_max=GlobalMaxPooling1D()(forward)
        pool_avg=GlobalAveragePooling1D()(forward)
        pool=Concatenate()([pool_max,pool_avg])
        # pool=Dropout(0.1)(pool)

        x=Dense(300,kernel_initializer="he_normal")(pool)
        x=LeakyReLU(alpha=0.2)(x)


        input2= Input(shape=(config.ngram_length,), dtype='int32')  # (60,)

        embedding2 = Embedding(len(ngram_vocab), config.emb_dim)(input2)  # (batch,100,300)
        conv_layers = [[128, 5], [128, 3]]
        conv=embedding2
        for num_filters, filter_width in conv_layers:
            conv=Convolution1D(num_filters,filter_width)(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
                                                        #(94,128)
        max=GlobalMaxPooling1D()(conv)
        avg=GlobalAveragePooling1D()(conv)
        concat=Concatenate()([max,avg])
        concat=Dropout(0.1)(concat)

        # b=Dense(256,kernel_initializer="he_normal")(concat)
        # b=LeakyReLU(alpha=0.2)(b)

        concat=Concatenate()([x,concat])


        x=Dense(512,kernel_initializer="he_normal")(concat)
        x=LeakyReLU(alpha=0.2)(x)
        # x = Dense(256, kernel_initializer="he_normal")(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(64, kernel_initializer="he_normal")(x)
        x = LeakyReLU(alpha=0.2)(x)
        #输出层
        predictions=Dense(units=num_class,activation='softmax')(x)
        #Bulid and compile model
        model = Model(inputs=[input1,input2], outputs=predictions)
        print("不良言论检测模型建立: ")
        model.summary()
    elif config.model_type=='bingge':
        import tensorflow as tf
        from tensorflow.keras.layers import concatenate,Lambda
        logger.info('Building a CharCNN')
        input1 = Input(shape=(overal_maxlen,), dtype='int32')

        input2 = Input(shape=(overal_maxlen,), dtype='int32')



        input1_emb = Embedding(len(word_vocab), 300)(input1)

        input2_emb = Embedding(len(word_vocab), config.emb_dim, name='emb')(input2)

        # Convolution layers
        conv_layers = [[128, 5], [128, 3]]
        convolution_output = []
        conv = input1_emb
        for num_filters, filter_width in conv_layers:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='relu',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(conv)

        pool_max = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
        convolution_output.append(pool_max)
        pool_avg = GlobalAveragePooling1D(name='GlobalAveragePooling1D{}_{}'.format(num_filters, filter_width))(conv)
        convolution_output.append(pool_avg)
        x_1 = Concatenate()(convolution_output)
        x_1 = Dropout(0.1)(x_1)
        x_1 = Dense(128, activation='relu')(x_1)
        x_1 = Dense(64, activation='relu')(x_1)
        x_1 = Dropout(0.1)(x_1)
        # conv_output = Dense(256, activation='relu')(x)

        # SWEM
        mlp_output = SelfAttention(300)(input2_emb)
        # mlp_output = Dense(300, activation='relu')(input2_emb)
        avg = GlobalAveragePooling1D()(mlp_output)
        max1 = GlobalMaxPooling1D()(mlp_output)
        x_2 = concatenate([max1, avg], axis=-1)
        x_2 = Dropout(0.1)(x_2)
        x_2 = Dense(128, activation='relu')(x_2)
        x_2 = Dense(64, activation='relu')(x_2)
        x_2 = Dropout(0.1)(x_2)

        x_3 = concatenate([max1, avg, pool_max, pool_avg], axis=-1)
        # x_3 = Dropout(0.2)(x_3)
        x_3 = Dense(256, activation='relu')(x_3)
        x_3 = Dense(64, activation='relu')(x_3)
        x_3 = Dropout(0.1)(x_3)

        gate_output = GateModule_word(units=[128, 3])(input1_emb)
        # gate_output2 = GateModule2(units=[128, 2])(input1_emb)
        gate_output = Lambda(expand_dim)(gate_output)
        # gate_output2 = Lambda(expand_dim)(gate_output2)

        stack = K.stack([x_1, x_2, x_3], axis=1) #在axis堆叠

        x = tf.matmul(gate_output, stack)  #相乘
        # x2 = tf.matmul(gate_output2, stack)
        x = tf.squeeze(x, axis=1)  #Remove dim 1

        fully_connected_layers = [16]
        for fl in fully_connected_layers:
            x = Dense(fl, activation='relu')(x)
        x = Dropout(0.1)(x)

        # Output layer
        predictions = Dense(units=num_class, activation='softmax')(x)
        model = tf.keras.Model(inputs=[input1, input2], outputs=predictions)
        model.emb_index = 1

        # model.compile(optimizer='adam', loss=self.loss)
        print("CharCNNKim model built: ")
        # print(model.layers)
        model.summary()
    return model


def pre_bert_model(config,length_vocab=1):  #

    from tensorflow.keras.layers import Input,Dense,Embedding,GlobalAveragePooling1D,GlobalMaxPooling1D,Concatenate,Convolution1D,Dropout,MaxPool1D\
        ,LSTM,LeakyReLU,concatenate,Flatten
    from transformers import BertTokenizer, TFBertModel
    #用下载在本地的预训练模型

    path='/data1/yangyong/junjie/offensive/Multilingual/bert-base-uncased'
    bert_model = TFBertModel.from_pretrained(path)
    if config.model_type=='pre_bert':

        input_ids=Input(shape=(128,),dtype='int32')
        input_mask=Input(shape=(128,),dtype='int32')
        input_tokentype=Input(shape=(128,),dtype='int32')

        #第二个输入
        second_input = Input(shape=(128,), dtype='int32')

        dim_out=bert_model({"input_ids":input_ids,"token_type_ids":input_tokentype,"attention_mask":input_mask})

        #第二输入的embedding
        emd_dim=Embedding(length_vocab,300)(second_input)


        #第二个输入的卷积池化操作

        conv=Convolution1D(128,5)(emd_dim)
        conv=Convolution1D(64,3)(conv)

        max_pool=GlobalMaxPooling1D()(conv) #300



        #Merge
        concat= concatenate([dim_out.pooler_output, max_pool], axis=-1)

        #金字塔操作
        one_merge=Dense(512,activation='elu',kernel_initializer='he_normal')(concat)

        two_merge=Dense(256,activation='elu',kernel_initializer='he_normal')(one_merge)

        three_merge=Dense(64,activation='elu',kernel_initializer='he_normal')(two_merge)

        #concate
        concat = concatenate([one_merge,two_merge,three_merge], axis=-1)
        

        x=Dense(256,activation='elu',kernel_initializer='he_normal')(concat)
        x=Dropout(0.2)(x)

        predict=Dense(2,activation='softmax')(x)

        model=Model(inputs=[input_ids,input_mask,input_tokentype,second_input],outputs=predict)

        model.summary()

    elif config.model_type=='xiaorong1':

        input_ids=Input(shape=(128,),dtype='int32')
        input_mask=Input(shape=(128,),dtype='int32')
        input_tokentype=Input(shape=(128,),dtype='int32')
        
        dim_out=bert_model({"input_ids":input_ids,"token_type_ids":input_tokentype,"attention_mask":input_mask})
        #金字塔操作
        one_merge=Dense(512,activation='elu',kernel_initializer='he_normal')(dim_out.pooler_output)
        two_merge=Dense(256,activation='elu',kernel_initializer='he_normal')(one_merge)
        three_merge=Dense(64,activation='elu',kernel_initializer='he_normal')(two_merge)

        #concate
        concat = concatenate([one_merge,two_merge,three_merge], axis=-1)

        x=Dense(256,activation='elu',kernel_initializer='he_normal')(concat)
        x=Dropout(0.2)(x)

        #输出层
        predict=Dense(2,activation='softmax')(x)

        model=Model(inputs=[input_ids,input_mask,input_tokentype],outputs=predict)

        model.summary()


    elif config.model_type=='bert_xiaorong':
        #第二个输入
        second_input = Input(shape=(128,), dtype='int32')
        #第二个输入得embedding
        emd_dim=Embedding(length_vocab,300)(second_input)

        #第二个输入得卷积操作
        conv=Convolution1D(128,5)(emd_dim)
        conv=Convolution1D(64,3)(conv)

        #第二个输入得池化操作
        max_pool=GlobalMaxPooling1D()(conv) #300

        #金字塔操作
        one_merge=Dense(256,activation='elu',kernel_initializer='he_normal')(max_pool)
        two_merge=Dense(128,activation='elu',kernel_initializer='he_normal')(one_merge)
        three_merge=Dense(64,activation='elu',kernel_initializer='he_normal')(two_merge)

        #concate
        concat = concatenate([one_merge,two_merge,three_merge], axis=-1)
        x=Dense(256,activation='elu',kernel_initializer='he_normal')(concat)
        x=Dropout(0.2)(x)
        #输出层
        predict=Dense(2,activation='softmax')(x)
        model=Model(inputs=second_input,outputs=predict)
        model.summary()


    return model