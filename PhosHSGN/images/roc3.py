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
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import precision_recall_curve,average_precision_score
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
        plt.figure( dpi=dpin)
        # plt.axis('square')


        
        for (name,colorname) in zip(names, colors):
            roc_poslist=[]
            reallist=[]
            roc_poslist_file=f'D:\paper\data/{name}/'+name+"_" + da + "poslist.csv"
            reallist_file=f'D:\paper\data/{name}/'+name+"_" + da + "reallab.csv"
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
            PRC = average_precision_score(reallist, roc_poslist)
            precision, recall, _ = precision_recall_curve(reallist, roc_poslist)
            plt.plot(recall, precision, '-', alpha=0.9, label='{} (PRC={:.3f})'.format(name, PRC), lw=1,color=colorname)
            plt.xlim([0, 1])
            plt.ylim([0.5, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'AUC PRC in {da} dataset')
            plt.legend(loc="lower center")
            # Set the x-axis scale range to 0 to 1.
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            # Set the y-axis scale range to 0.5 to 1.
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
            plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio())

        if save:
            plt.savefig(f'D:\paper\data\AUC_PRC_in_{da}_dataset.png')

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

