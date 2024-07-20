import csv
import os
import argparse
import numpy as np
import pandas as pd

from config import *
# Press the green button in the gutter to run the script.
from train import train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', help='train_dataset file path', required=True)
    parser.add_argument('--test_dataset_path', help='test_dataset  file path', required=True)
    parser.add_argument('--batchsize', help='batchsize', default='64')
    parser.add_argument('--lr', help='learn rate', default=0.0001)
    parser.add_argument('--weight_decay', help='weight_decay rate', default=0.0001)
    parser.add_argument('--dropout_ratio', help='dropout_ratio rate', default=0.3)
    parser.add_argument('--epochs', help='epoch', default=100)
    parser.add_argument('--model_name', help='model_name',required=True)#Deep Deep_GNN Deep_CNN
    parser.add_argument('--emb_type', help='emb_type', required=True) #esm seqvec prot
    parser.add_argument('--task_msg', help='task_msg', required=True)
    parser.add_argument('--train_data_pos_size', help='train_data_pos_size', required=True)
    parser.add_argument('--train_data_neg_size', help='train_data_neg_size', required=True)
    parser.add_argument('--test_data_pos_size', help='test_data_pos_size', required=True)
    parser.add_argument('--test_data_neg_size', help='test_data_neg_size', required=True)
    args = parser.parse_args()
    arg_config = Config(batchsize=args.batchsize, lr=args.lr, weight_decay=args.weight_decay, dropout_ratio=args.dropout_ratio, epochs=int(args.epochs),
                        train_dataset_path=args.train_dataset_path,
                        test_dataset_path=args.test_dataset_path)
    arg_config.emb_type=args.emb_type
    arg_config.task_msg = args.model_name+ "_"+arg_config.emb_type + "_" + args.task_msg
    arg_config.model_name = args.model_name
    arg_config.train_data_pos_size = args.train_data_pos_size
    arg_config.train_data_neg_size = args.train_data_neg_size
    arg_config.test_data_pos_size = args.test_data_pos_size
    arg_config.test_data_neg_size = args.test_data_neg_size
    # hyperparameter object
    train(arg_config)
