# -*- coding:utf-8 -*-
# Description TODO
# author Rover  Email:1059885524@qq.com
# version 1.0
# date 2021/8/5 13:57
import logging
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Evaluator:
    def __init__(self, args, out_dir,batch_size,*kwargs):
        X_test, y_test=kwargs
        self.model_type = args.model_type
        self.out_dir = out_dir

        self.test_x=X_test
        #标签
        self.test_y=y_test


        self.batch_size = batch_size

        self.best_test_f1 = 0
        self.best_acc = 0
        self.best_report = None
        self.best_test_epoch = -1

    def evaluate(self, model, epoch, print_info=False):

        self.test_pred = model.predict(self.test_x, batch_size=self.batch_size)

        self.test_pred_label = np.argmax(self.test_pred, axis=1)

        self.test_y_label = np.argmax(self.test_y, axis=1)

        self.acc = accuracy_score(self.test_y_label, self.test_pred_label)

        #要宏观f1
        self.f1_all = f1_score(self.test_y_label, self.test_pred_label, average='macro')

        self.report = classification_report(self.test_y_label, self.test_pred_label,digits=4)


        if self.f1_all >= self.best_test_f1:
            self.best_test_f1 = self.f1_all
            self.best_acc = self.acc
            self.best_test_epoch = epoch
            self.best_report = self.report
            #将最好的权重保存下来
            model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)

        if print_info:
            logger.info("Evaluation on test data: acc = %0.4f f1 = %0.4f%%" % (self.acc, self.f1_all) )

    def info_f1(self):
        return self.f1_all

    def print_final_info(self):
        logger.info('--------------------------------------------------------------------------------------------------------------------------')
        logger.info('Best @ Epoch %i:' % self.best_test_epoch)
        # logger.info('BestF1 %f ' % self.best_test_f1)
        logger.info('  [TEST] report %s' % self.best_report)