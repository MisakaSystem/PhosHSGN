import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dgl
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn, Tensor

import config
from dataset import LuDataset
from torch.utils.data import DataLoader

from log.logdata import Log, epoch_data
from ludeep import LuGTP,LuGTP_CNN,LuGTP_GNN
from utils.assess import calculate_indicators, calculate_auc
from datasetpre.dataprocess import get_graph, getseqmatrix, getpssms, getdssps, listclass_to_one_hot

from utils.views import  plot_loss

# performance recording indicators
Acc=0.
Precision=0.
Recall=0.
F1=0.
Sensitivity=0.
Specificity=0.
Auccal=0.
log=''
train_data_list=[]
test_data_list=[]
def collate(samples):
    protein, seq, label,position = map(list, zip(*samples))
    return protein, seq, label,position
def pad_dmap(dmaplist,arg_config):
    if arg_config.emb_type == 'esm':
        emblength = 1280
    if arg_config.emb_type == 'prot' or arg_config.emb_type == 'seqvec':
        emblength = 1024
    cutproteinlen = arg_config.cutproteinlen
    pad_dmap_tensors = torch.zeros((len(dmaplist), cutproteinlen, emblength)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors
def pad_dseq(dmaplist):
    pad_dmap_tensors = torch.zeros((len(dmaplist), 51, 50)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors
def evaluate_test_accuracy_gpu(epoch, net, data_iter, test_file,arg_config):
    device=arg_config.device
    pred_labs = []
    real_labs = []
    val_l = 0.
    net.eval()
    roc_poslist = []
    reallab = []
    predictions_t=[]
    proteinlist=[]
    poslist=[]
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (proteins, seqs, labels,positions) in enumerate(data_iter):
            print(f" Test[epoch:{epoch}] batch_idx:{batch_idx} process：{round(batch_idx/len(data_iter)*100,2)}%")
            graph, emd = get_graph(seqs, proteins,arg_config)
            pssms=getpssms(seqs,proteins,positions)
            seqmatrix, label,protein,pos=getseqmatrix(seqs,proteins,positions,labels)
            dssps = getdssps(seqs, proteins, positions)
            multiaccessdata=[]
            for (seq, pssm, dssp) in zip(seqmatrix, pssms, dssps):
                concatdata = np.concatenate([seq, pssm,dssp], axis=1)
                multiaccessdata.append(torch.tensor(concatdata))
            concatinputdata=pad_dseq(multiaccessdata)
            concatinputdata = concatinputdata.squeeze(1)
            concatinputdata = concatinputdata.permute(0, 2, 1)
            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd,arg_config), seqmatrix, pssms, dssps,concatinputdata)
            for pr in protein:
                proteinlist.append(pr)
            for ps in pos:
                poslist.append(ps)
            for yh in y_hat:
                yhh=Tensor.cpu(yh[1]).numpy()
                roc_poslist.append(Tensor.cpu(yh[1]).numpy())
                predictions_t.append(yhh)
            real_l = label.argmax(dim=1)
            for rl in real_l:
                reallab.append(Tensor.cpu(rl).numpy())
            pred = y_hat.argmax(dim=1)
            l = loss(y_hat, label.to(device))
            print(f' test_loss:{l}')
            list_len = len(pred)
            li = []
            for i in range(list_len):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            lab = torch.zeros(list_len, 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(lab)
            real_labs.append(label.to(device))
            val_l += l
    Loss=(val_l / len(data_iter)).detach().cpu().numpy()

    assess = calculate_indicators(epoch, pred_labs, real_labs)

    auccal = calculate_auc(roc_poslist, reallab)
    if auccal==None:
        auccal=0.
    global Acc
    global Precision
    global Recall
    global F1
    global Sensitivity
    global Specificity
    global Auccal
    if assess.get("accuracy") > Acc:
        results_S = np.column_stack((proteinlist, poslist, predictions_t[:]))
        result = pd.DataFrame(results_S)
        result.to_csv(test_file+"-prediction_phosphorylation.csv", index=False, header=None, sep='\t',
                      quoting=csv.QUOTE_NONNUMERIC)
        Acc = assess.get("accuracy")
        Precision = assess.get("precision")
        Recall = assess.get("recall")
        F1 = assess.get("f1")
        Sensitivity = assess.get("sensitivity")
        Specificity = assess.get("specificity")
        Auccal = auccal
    test_data_list.append(epoch_data(epoch, datetime.now(),"test", Loss, assess.get("accuracy"), assess.get("precision"),  assess.get("recall"), assess.get("f1"), assess.get("sensitivity"), assess.get("specificity"), roc_poslist,
                       reallab, assess.get("TP"),assess.get("FP"),assess.get("TN"),assess.get("FN"),assess.get("MCC"),Auccal))
    print(
        "Test  Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t MCC: {:.6f}\t AUC: {:.6f}\t".format(
            epoch, val_l / len(data_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
            assess.get("f1"), assess.get("sensitivity"), assess.get("specificity"),assess.get("MCC"),Auccal))
    print(
        "----------------------------------------------------------------   -----------------------------------------------------------------------------------")
    if assess.get("accuracy")>0.80:
        print('===> Saving models...')
        state = {
            'state': net.state_dict(),
            'epoch': epoch                   # save the epoch with state
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model'+f"-{arg_config.model_name}-{arg_config.task_msg}-"+str(assess.get("accuracy"))+'.t7')

def train(config):
    global log
    log=Log(config.model_name, config.task_msg, config.train_data_pos_size, config.train_data_neg_size,
        config.test_data_pos_size, config.test_data_neg_size)
    device = config.device
    batchsize=config.batchsize
    lr=config.lr
    weight_decay=config.weight_decay
    epochs=config.epochs
    train_dataset_path=config.train_dataset_path
    test_dataset_path=config.test_dataset_path
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    print("Loading the training dataset: " + train_dataset_path)
    train_data = LuDataset(file_path=train_dataset_path)
    train_iter = DataLoader(dataset=train_data, batch_size=int(batchsize), shuffle=True, drop_last=True, collate_fn=collate)
    print("Loading the test dataset: " + test_dataset_path)
    test_data = LuDataset(file_path=test_dataset_path)
    test_iter = DataLoader(dataset=test_data, batch_size=int(batchsize), shuffle=False, drop_last=True, collate_fn=collate)

    if 'GNN' in config.model_name:
        net = LuGTP_GNN(config).to(config.device)
    elif 'CNN' in config.model_name:
        net = LuGTP_CNN(config).to(config.device)
    else:
        # batch loading of datasets.
        net = LuGTP(config).to(config.device)
    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        train_l = 0.
        pred_labs = []
        real_labs = []
        for batch_idx, (proteins, seqs, labels,positions) in enumerate(train_iter):
            optimizer.zero_grad()
            # Pass data to the model
            # TODO Need to process it here
            print(f" Train[epoch:{epoch}] batch_idx:{batch_idx} process：{round(batch_idx/len(train_iter)*100,2)}%")
            graph, emd = get_graph(seqs, proteins,config)
            pssms=getpssms(seqs,proteins,positions)
            seqmatrix, label,protein,pos=getseqmatrix(seqs,proteins,positions,labels)
            dssps = getdssps(seqs, proteins, positions)

            multiaccessdata=[]
            for (seq, pssm, dssp) in zip(seqmatrix, pssms, dssps):
                concatdata = np.concatenate([seq,pssm,dssp], axis=1)
                multiaccessdata.append(torch.tensor(concatdata))
            concatinputdata=pad_dseq(multiaccessdata)
            concatinputdata = concatinputdata.squeeze(1)
            concatinputdata = concatinputdata.permute(0, 2, 1)

            y_hat = net(dgl.batch(graph).to(device), pad_dmap(emd,config), seqmatrix, pssms, dssps, concatinputdata)
            pred = y_hat.argmax(dim=1)
            li = []
            for i in range(len(pred)):
                li.append([pred[i]])
            one_hot_list = torch.LongTensor(li).to(device)
            pre_lab = torch.zeros(len(pred), 2).to(device).scatter_(1, one_hot_list, 1)
            pred_labs.append(pre_lab)
            l = loss(y_hat, label.to(device))
            print(f' train_loss:{l}')
            l.backward()
            train_l += l
            real_labs.append(label.to(device))
            optimizer.step()
        Loss=(train_l / len(train_iter)).detach().cpu().numpy()
        assess = calculate_indicators(epoch, pred_labs, real_labs)
        train_data_list.append(epoch_data(epoch, datetime.now() ,"train", Loss, assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
                           assess.get("f1"), assess.get("sensitivity"), assess.get("specificity"), pred_labs,
                           real_labs, assess.get("TP"),assess.get("FP"),assess.get("TN"),assess.get("FN"),assess.get("MCC"),0))

        print(
            "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\t Precision: {:.6f}\t Recall: {:.6f}\t F1: {:.6f}\t Sensitivity: {:.6f}\t Specificity: {:.6f}\t MCC: {:.6f}\t AUC: {:.6f}\t".format(
                epoch, train_l / len(train_iter), assess.get("accuracy"), assess.get("precision"), assess.get("recall"),
                assess.get("f1"), assess.get("sensitivity"), assess.get("specificity"),assess.get("MCC"),0))
        evaluate_test_accuracy_gpu(epoch, net, test_iter,test_dataset_path,config)
    for train_data in  train_data_list:
        log.log_data.append(train_data)
    for test_data in  test_data_list:
        log.log_data.append(test_data)
    log.to_csv()