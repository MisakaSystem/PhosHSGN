import os
import pickle

import numpy as np
import time
import sys
import math
max_value = 0.
min_value = 0.
def extract_lines(pssmFile):
    fin = open(pssmFile)
    pssmLines = []
    if fin == None:
        return
    for i in range(3):
        fin.readline()  # exclude the first three lines
    while True:
        psspLine = fin.readline()
        if psspLine.strip() == '' or psspLine.strip() == None:
            break
        pssmLines.append(psspLine)
    fin.close()
    return pssmLines

def load_fasta_and_compute(protein, position, pssm_path, start, end, l_padding=0, r_padding=0):
    pssm = LoadPSSMandPrintFeature(pssm_path, protein, end - start,position)
    # pssm=LoadPSSM()
    # fout.write(line_Pid)
    # fout.write(line_Pseq)n

    # fout.write(",".join(map(str,Feature)) + "\n")

    # pssm2 = pssm[:,int(start):int(end)]

    # pssm2 = pssm
    # if l_padding>0:
    #     newRows=np.zeros((20,int(l_padding)))
    #     pssm2=np.c_[newRows,pssm2]
    # if r_padding>0:
    #     newRows=np.zeros((20,int(r_padding)))
    #     pssm2=np.c_[pssm2,newRows]

    return pssm
def LoadPSSMandPrintFeature(pssm_fn, Pid, line_Pseq,position,window_size=51):
    global min_value, max_value
    fin = open(pssm_fn, "r")
    try:
        pssmLines = extract_lines(pssm_fn)
    except UnicodeDecodeError:
        pssm_np_2D = np.zeros(shape=(20, 51))
        return pssm_np_2D
    seqlen = len(pssmLines)
    if(position>seqlen):
        print(f"error {Pid}")
    upperBound = position + int(window_size / 2)
    lowerBound = position - int(window_size / 2)
    upexpend = 0
    lowexpend = 0
    ex = '0 0     0   0  0   0  0   0   0   0  0   0  0   0  0   0   0   0   0  0   0   0'
    if upperBound>seqlen and lowerBound<=0:
        try:
            upexpend = upperBound - seqlen
            lowexpend = 0 - lowerBound + 1
            pssmLines = pssmLines[0:seqlen]
            upexpendlist=[]
            lowexpendlist=[]
            for i in range(upexpend):
                upexpendlist.append(ex)
            pssmLines = pssmLines + upexpendlist
            for i in range(lowexpend):
                lowexpendlist.append(ex)
            pssmLines = lowexpendlist + pssmLines
        except:
            print("debug")
    if upperBound > seqlen:
        upexpend = upperBound - seqlen
        pssmLines = pssmLines[lowerBound - 1:seqlen]
        expendlist=[]
        for i in range(upexpend):
            expendlist.append(ex)
        pssmLines = pssmLines + expendlist
    if lowerBound <= 0:
        lowexpend = 0 - lowerBound + 1
        pssmLines = pssmLines[0:upperBound]
        expendlist = []
        for i in range(lowexpend):
            expendlist.append(ex)
        pssmLines = expendlist + pssmLines
    if lowerBound > 0 and upperBound <= seqlen:
        pssmLines = pssmLines[lowerBound - 1:upperBound]
    seq_len = len(pssmLines)

    pssm_np_2D = np.zeros(shape=(20, 51))
    less = 51 - len(pssmLines)
    skipLen = int(less / 2)

    for i in range(len(pssmLines)):
        values_20 = pssmLines[i].split()[2:22]
        for j in range(len(values_20)):
            max_value = max(max_value, float(values_20[j]))
            min_value = min(min_value, float(values_20[j]))

    max_value += 1
    min_value -= 1

    for i in range(51):
        if i < skipLen or i > 51 - skipLen - 1:
            continue
        else:
            try:
                values_20 = pssmLines[i - skipLen].split()[2:22]
            except IndexError:
                print(f"error out of index {pssm_fn}")
                pssm_np_2D = np.zeros(shape=(20, 51))
                return pssm_np_2D
        for aa_index in range(20):
            pssm_np_2D[aa_index][i] = (float(values_20[aa_index]) - min_value) / (
                        max_value - min_value)
    fin.close()
    return pssm_np_2D