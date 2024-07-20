import os
from datetime import datetime
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from torch import Tensor
styles=['fivethirtyeight',
 'dark_background',
 'bmh',
 'classic',
 'seaborn-dark',
 'grayscale',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-colorblind',
 'seaborn-white',
 'seaborn-dark-palette',
 'ggplot',
 'tableau-colorblind10',
 '_classic_test',
 'seaborn-darkgrid',
 'seaborn-notebook',
 'Solarize_Light2',
 'seaborn-paper',
 'seaborn-whitegrid',
 'seaborn-pastel',
 'seaborn-talk',
 'seaborn-bright',
 'seaborn',
 'seaborn-ticks',
 'seaborn-poster',
 'fast']

def calculate_auc(roc_poslist, reallab):
    y = reallab
    scores = roc_poslist
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4),dpi=100)
    # plt.style.use('seaborn-white')
    palette = plt.get_cmap('Set1')
    plt.plot(fpr, tpr, 'k--', alpha=0.9, label='AUC=%0.3f'%(roc_auc), lw=1)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f'images/{time}.jpg',dpi=300)
    plt.cla()
    return roc_auc
def calculate_indicators(epoch, pre_lablist, real_lablist):
    try:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for pre_lab, real_lab in zip(pre_lablist, real_lablist):
            for pl, rl in zip(pre_lab, real_lab):

                # s = torch.tensor([1., 0.]).to(torch.device('cuda'))
                # ps = torch.eq(pl, s)

                if (torch.eq(pl, torch.tensor([0., 1.]).to(torch.device('cuda'))).sum()) / 2 == 1:
                    if (torch.eq(pl, rl).sum()) / 2 == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                elif (torch.eq(pl, torch.tensor([1., 0.]).to(torch.device('cuda'))).sum()) / 2 == 1:
                    if (torch.eq(pl, rl).sum()) / 2 == 1:
                        TN = TN + 1
                    else:
                        FN = FN + 1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP+ 1e-6)
        recall = TP / (TP + FN+ 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall+ 1e-6)
        sensitivity = TP / (TP + FN+ 1e-6)
        specificity = TN / (TN + FP+ 1e-6)
        mcc=(TP*TN-TP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        par = {"epoch": epoch, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
               "sensitivity": sensitivity,
               "specificity": specificity,"TP":TP,"FP":FP,"TN":TN,"FN":FN,"MCC":mcc}
    except Exception as r:
        print('Skip error %s' % (r))
        par = {"epoch": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "sensitivity": 0, "specificity": 0,"TP":0,"FP":0,"TN":0,"FN":0,"MCC":0}
        return par
    return par