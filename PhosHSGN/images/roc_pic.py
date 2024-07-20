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
from datasetpre.dataprocess import listclass_to_one_hot

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

data_list=['6rnF2r1XHo','9im7rqUBDS','wIyB60WWe9','IhsGJx675T','ndm0G1LhqG']

def calculate_auc(roc_poslist, reallab):
    # rb=[]
    # for re in reallab:
    #     rb.append(float(re))
    # y = np.array(rb)

    scores = roc_poslist
    fpr, tpr, thresholds = metrics.roc_curve(reallab, scores, pos_label=1)

    roc_auc = auc(fpr, tpr)
    print(f"auc:{roc_auc}")
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
    plt.show()
    plt.cla()
    return roc_auc
def calculate_auc2(roc_poslist, reallab):

    y=reallab
    scores =roc_poslist
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    roc_auc = auc(fpr, tpr)

    # plt.style.use('seaborn-white')
    palette = plt.get_cmap('Set1')
    plt.plot(fpr, tpr, 'k--', alpha=0.9, label='ROC curve (area = %0.2f)'%(roc_auc), lw=1)

    return roc_auc
def load_epoch_data(filename):
    try:
        data = np.load('temp/'+filename+".npy", allow_pickle=True)

        item= data.item()
        roc_poslist=item.roc_poslist
        # roc_poslist=[]

        # for yh in y_pred:
        #     yhh = yh[1]
        #     roc_poslist.append(yhh)

        # rc=[]
        # for ine in roc_poslist:
        #     r=ine[1]
        #     rc.append(r)
        reallab=item.reallab
        # reallab = np.argmax(reallab, axis=1)
        # y=np.argmax(reallab, axis=1)
        # rc = np.array(rc)
        calculate_auc(roc_poslist,reallab)
        column1 = ['0']
        column = ['0']
        poslist = pd.DataFrame(columns=column1, data=roc_poslist)
        poslist.to_csv('temp/'+filename+"poslist.csv")
        reallab = pd.DataFrame(columns=column, data=reallab)
        reallab.to_csv('temp/'+filename+"reallab.csv")
        print("done")
    except ModuleNotFoundError:
        print(filename)
    except TypeError:
        data = np.load('temp/' + filename + ".npy", allow_pickle=True)

        item = data.item()
        roc_poslist = item.roc_poslist
        mylist=[]
        for li in roc_poslist:
            for i in li:
                mylist.append(i)
        reallab = item.reallab
        reallist=[]
        for re in reallab:
            re=int(re)
            reallist.append(re)
        reallist=listclass_to_one_hot(reallist)
        mylist=mylist
        column = ['0', '1']
        poslist = pd.DataFrame(columns=column, data=mylist)
        poslist.to_csv('temp/' + filename + "poslist.csv")
        reallab = pd.DataFrame(columns=column, data=reallist)
        reallab.to_csv('temp/' + filename + "reallab.csv")
        calculate_auc(mylist,reallist)
        print("done")
def draw(datalist):
    plt.figure(figsize=(5, 4), dpi=100)
    for da in datalist:
        roc_poslist=[]
        reallist=[]
        roc_poslist_file='temp/' + da + "poslist.csv"
        reallist_file='temp/' + da + "reallab.csv"
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
        calculate_auc2(roc_poslist,reallist)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plt.show()
    plt.cla()
if __name__ == '__main__':
        load_epoch_data('Mixhop_CAMK')
        # draw(data_list)