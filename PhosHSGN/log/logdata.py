import os
from datetime import datetime
import csv
import random
import string
import numpy as np


class Log():
    def __init__(self,model_name,msg,train_data_pos_size,train_data_neg_size,test_data_pos_size,test_data_neg_size):
        self.model_name = model_name
        self.log_data = []
        self.train_data_pos_size=int(train_data_pos_size)
        self.train_data_neg_size=int(train_data_neg_size)
        self.test_data_pos_size = int(test_data_pos_size)
        self.test_data_neg_size = int(test_data_neg_size)
        self.msg=msg

    def add_epoch_data(self, epoch, task_type, loss, acc, precision, recall, f1, sensitivity, specificity,roc_poslist,reallab,TP,FP,TN,FN,MCC,AUC):
        data = epoch_data(datetime.now(),epoch, task_type, loss, acc, precision, recall, f1, sensitivity, specificity,roc_poslist,reallab,TP,FP,TN,FN,MCC,AUC)
        self.log_data.append(data)

    # Save the epoch_data object as a .npy file
    @staticmethod
    def save_epoch_data(epoch_data, filename):
        np.save(filename, epoch_data)

    @staticmethod
    # Load epoch_data object from .npy file
    def load_epoch_data(filename):
        data = np.load(filename, allow_pickle=True)
        return data.item()
    def to_csv(self):
        header = ['ID','Time','Epoch', 'Task Type', 'Loss', 'Acc', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity', 'TP','FP','TN','FN','MCC','AUC']
        file_exists = True if os.path.exists(f'log/{self.model_name}_log.csv') else False

        with open(f'log/{self.model_name}_log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # add a blank line before each appended data
            if file_exists:
                writer.writerow([])

            writer.writerow([f"[{self.msg}] 训练集大小：{self.train_data_pos_size+self.train_data_neg_size},正例数量：{self.train_data_pos_size}，反例数量：{self.train_data_neg_size},测试集大小：{self.test_data_pos_size+self.test_data_neg_size},正例数量：{self.test_data_pos_size}，反例数量：{self.test_data_neg_size}"])
            writer.writerow(header)
            # write log_data data
            for data in self.log_data:
                writer.writerow([data.id,data.time,data.epoch, data.task_type, data.loss, data.acc, data.precision, data.recall, data.f1,
                                 data.sensitivity, data.specificity,data.TP,data.FP,data.TN,data.FN,data.MCC,data.AUC])
                self.save_epoch_data(data,f'log/epoch_obj/{data.id}.npy')

class epoch_data():
    def __init__(self,time, epoch, task_type, loss, acc, precision, recall, f1, sensitivity, specificity,roc_poslist,reallab,TP,FP,TN,FN,MCC,AUC):
        # random 5-digit string
        random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))

        self.id=random_string
        self.time=time
        self.epoch = epoch
        self.task_type = task_type
        self.loss = loss
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.roc_poslist=roc_poslist
        self.reallab=reallab
        self.TP=TP
        self.FP=FP
        self.TN=TN
        self.FN=FN
        self.MCC=MCC
        self.AUC=AUC
