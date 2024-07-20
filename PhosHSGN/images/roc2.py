import numpy as np
import os
from datetime import datetime
import math
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from torch import Tensor
import torch
import pandas as pd
import csv
def multi_models_roc(names, colors,datalist, save=True, dpin=100):
    """
    Output the roc graphs of multiple models to one graph.

    Args:
        names: list, Names of multiple models.
        sampling_methods: list, Multiple model instantiation objects.
        save: Choose whether to save the result (default is png format).

    Returns:
        Returns the image object plt.
    """
    for da in datalist:
        plt.figure(figsize=(5, 4), dpi=dpin)
        for (name,colorname) in zip(names, colors):
            roc_poslist=[]
            reallist=[]
            roc_poslist_file=f'D:\paper\data/{name}/'+name+"_" + da + "poslist.csv"
            reallist_file=f'D:\paper\data/{name}/'+name+"_" + da + "reallab.csv"
            try:
                data = np.load(f'D:\paper\data/{name}/'+name+"_" + da + ".npy", allow_pickle=True)

                item = data.item()
                print(f"{name}_{da} : AUC_{item.AUC} sn_{item.sensitivity} acc_{item.acc} mcc_{item.MCC} pr_{item.precision} f1_{item.f1}")
            except Exception:
                print("debug")
            with open(roc_poslist_file, 'r') as rf:
                reader = csv.reader(rf)
                for row in reader:
                    data = row[0]
                    roc_poslist.append(float(data))
            with open(reallist_file, 'r') as rf:
                reader2 = csv.reader(rf)
                for row2 in reader2:
                    data2 = row2[0]
                    reallist.append(float(data2))
            fpr, tpr, thresholds = metrics.roc_curve(reallist, roc_poslist, pos_label=1)

            plt.plot(fpr, tpr, '-', alpha=0.9, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), lw=1,color=colorname)
            plt.axis('square')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('1-Specificity')
            plt.ylabel('Sensitivity')
            plt.title(f'AUC ROC in {da} dataset')
            plt.legend(loc="lower right")
        if save:
            plt.savefig(f'D:\paper\data\AUC_ROC_in_{da}_dataset.png')

    return None
colors = ['steelblue',
          'orange',
          'gold',
          'darkkhaki',
          'turquoise',
          'salmon',
          'crimson'
         ]


names = [ 'DeepPhos',
          'PhosIDN',
          'MusiteDeep',
          'MixHop',
          'Carate',
          'PhosHSGN'
         ]
data_list=['AGC','Atypical','CAMK','CDK','CK2','CMGC','MAPK','PKC','Src','TK']

#ROC curves
train_roc_graph = multi_models_roc(names, colors,data_list, save = True)

