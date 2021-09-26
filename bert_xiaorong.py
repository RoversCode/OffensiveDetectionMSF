# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2021/8/5 13:56
from tools import  pre_train_bert,config,data_reader
from time import time  #该模块提供了各种与时间相关的函数
import sys
import logging #日志包
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



#指定GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



#配置
config=config.ModelConfig()

#得到第二组输入
X_train, X_test, y_train, y_test, vocab_length=data_reader.get_data(config)


###############################################################################################################################
## Optimizaer algorithm
#

from tools.optimizers import get_optimizer
optimizer = get_optimizer(config)

#创建模型
from   tools import  model
from tools  import  bert_evl as evaluator


model=model.pre_bert_model(config,vocab_length)

# #冻结bert层
# for layer in model.layers[:-1]:
#     if layer.name=="tf_bert_model":
#         layer.trainable=False



evl=evaluator.Evaluator(config,config.out_dir,config.batch_size,X_test,y_test)




model.compile(optimizer=optimizer,loss=config.loss,metrics='accuracy')


total_train_time = 0
total_eval_time = 0
t1 = time()
f1=0
for ii in range(config.epochs):
    t0 = time()
    train_history=model.fit(X_train,y_train,batch_size=config.batch_size,epochs=1,verbose=1)

    tr_time = time() - t0
    total_train_time += tr_time

    #评估
    t0 = time()
    evl.evaluate(model, ii,print_info=True)
    evl_f1=evl.info_f1()
    evl_time = time() - t0
    total_eval_time += evl_time
    total_time = time()-t1

    # Print information
    train_loss = train_history.history['loss'][0]
    train_metric = train_history.history['accuracy'][0]

    logger.info('Epoch %d, train: %is, evaluation: %is, toaal_time: %is' % (ii, tr_time, evl_time, total_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
    if evl_f1>f1:
        f1=evl_f1
    else:
        break  #不训练了


evl.print_final_info()
