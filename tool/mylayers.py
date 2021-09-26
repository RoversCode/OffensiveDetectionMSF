#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/16 17:43
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dense, Convolution1D, Dropout,\
							 GlobalAveragePooling1D, Concatenate, Add
from keras.layers import Concatenate
class SelfAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape): #为SelfAttention层定义权重
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      trainable=True)
        super(SelfAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self,x): #这里是编写层的功能逻辑
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (300 ** 0.5)  #这个300是 q和k的维度(这里是小写的q,k 是列向量的)
        QK = K.softmax(QK)

        B = K.batch_dot(QK, WV)

        return B

    def compute_output_shape(self, input_shape):
        # inputs.shape = (batch_size, time_steps, seq_len) 这里的time_stpes是指sentences的长度，seq_len是嵌入维度
        return(input_shape[0], input_shape[1], self.output_dim)

class MultiHeadAttention(Layer):
    """
        多头注意力机制
    """
    def __init__(self, heads, head_size, output_dim=None, **kwargs):
        self.heads = heads
        self.head_size = head_size
        self.output_dim = output_dim or heads * head_size
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.head_size),
                                      initializer='uniform',
                                      trainable=True)
        self.dense = self.add_weight(name='dense',
                                     shape=(input_shape[2], self.output_dim),
                                     initializer='uniform',
                                     trainable=True)

        super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        out = []
        for i in range(self.heads):
            WQ = K.dot(x, self.kernel[0])
            WK = K.dot(x, self.kernel[1])
            WV = K.dot(x, self.kernel[2])

            # print("WQ.shape",WQ.shape)
            # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

            QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
            QK = QK / (100 ** 0.5)
            QK = K.softmax(QK)

            # print("QK.shape",QK.shape)

            V = K.batch_dot(QK, WV)
            out.append(V)
        out = Concatenate(axis=-1)(out)
        out = K.dot(out, self.dense)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)



class BaseLayer(Layer):
    def build_layers(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape=layer.compute_output_shape(shape)


class GateModule_word(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers=[]
        self.layers = []
        super(GateModule_word, self).__init__(**kwargs)

    def build(self, input_shape):
        # conv_layer = Convolution1D(filters=64,kernel_size=10,padding='same',activation='relu')
        # conv_layer.build(input_shape)
        # shape = conv_layer.compute_output_shape(input_shape)
        # self.conv_layers.append(conv_layer)
        # pooling_layer = GlobalMaxPooling1D()
        # shape = pooling_layer.compute_output_shape(shape)
        # self.pooling_layers.append(pooling_layer)
        self.layers.append(Convolution1D(filters=128,kernel_size=5,activation='relu'))
        self.layers.append(Convolution1D(filters=128,kernel_size=3,activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        # self.layers.append(Dropout(0.2))
        self.layers.append(Dense(128, activation='relu'))
        self.layers.append(Dense(128, activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[1], activation='softmax'))
        # self.build_layers(shape)

        super(GateModule_word,self).build(input_shape)

    def call(self, inputs):
        ## 2. 主要是改这里的结构！！！
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs_max = self.layers[2](xs)
        xs_avg = self.layers[3](xs)
        xs = self.layers[4]([xs_max, xs_avg])
        # xs = self.layers[5](xs)
        for layer in self.layers[5:]:
            # xs=Dropout(0.1)(xs)
            xs=layer(xs)
        # xs = tf.nn.softmax(xs, axis=-1)
        return xs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units[-1]]
