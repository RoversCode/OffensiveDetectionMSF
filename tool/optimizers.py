#-*- coding:utf-8 -*-
#Description TODO
#author Rover  Email:1059885524@qq.com
#version 1.0
#date 2021/5/16 16:57
 
import tensorflow.keras.optimizers as opt

def get_optimizer(config):
    # clipvalue=0
    # clipnorm=10
    if config.algorithm =='adam':
        #是否可以做更改，还犹未可知
        #参数beta_1和beta_2是Adam论文中的衰减因子，然后epsilon是平滑项。beta_1对应借鉴的是动量优化中的动量beta
        #beta_2借鉴的是，RMSprop中的衰减率。所以Adam是动量优化和RMSProp两者思想的结合
        optimizer = opt.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    elif config.algorithm=='nadam':
        optimizer=opt.Nadam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-7)
    return optimizer