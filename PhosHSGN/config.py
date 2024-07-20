import torch


class Config:
    def __init__(self, batchsize, lr, weight_decay,dropout_ratio, epochs,train_dataset_path,test_dataset_path):
        self.batchsize = batchsize
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.cutproteinlen = 500
        self.train_dataset_path=train_dataset_path
        self.test_dataset_path=test_dataset_path
        self.device = torch.device('cuda')
        self.dropout_ratio=dropout_ratio
        self.train_data_pos_size=0
        self.train_data_neg_size=0
        self.test_data_pos_size = 0
        self.test_data_neg_size = 0
        self.emb_type='esm'#prot#seqvec
        self.model_name='default'
        self.task_msg='default'
