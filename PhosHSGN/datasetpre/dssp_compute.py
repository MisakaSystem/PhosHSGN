import numpy as np


def extract_lines(pssmFile):
    fin = open(pssmFile)
    pssmLines = []
    if fin == None:
        return
    while True:
        psspLine = fin.readline()
        if psspLine.strip() == '' or psspLine.strip() == None:
            break
        pssmLines.append(psspLine)
    fin.close()
    return pssmLines

letterDict = {}
letterDict["H"] = 0
letterDict["B"] = 1
letterDict["E"] = 2
letterDict["G"] = 3
letterDict["I"] = 4
letterDict["T"] = 5
letterDict["S"] = 6
letterDict[" "] = 7
letterDict["N"] = 8

def loaddsspfile(dssp_file_url, protein,position, seqlen,window_size=51):
    sec=''
    fin = open(dssp_file_url, "r")
    residueCount = 0
    start = 0
    residueIndex = -1
    for i in range(100):
        content = fin.readline()
        if i == 6:
            residueCount = content.split()[0]
        if '#' in content:
            residueIndex = content.index("STRUCTURE")
            break
    for i in range(int(residueCount)):
        residue_data = fin.readline()
        sec=sec+residue_data[residueIndex]
    worstwindow=0
    upperBound = position + int(window_size/2)
    lowerBound = position - int(window_size/2)
    upexpend=0
    lowexpend=0
    if upperBound>seqlen and lowerBound<=0:
        try:
            upexpend = upperBound - seqlen
            lowexpend = 0 - lowerBound + 1
            sec = sec[0:seqlen]
            upexpendex=''
            lowexpendex=''
            for i in range(upexpend):
                upexpendex = upexpendex + "N"
            sec = sec + upexpendex
            for i in range(lowexpend):
                lowexpendex = lowexpendex + "N"
            sec = lowexpendex + sec
        except:
            print("debug")
    if upperBound>seqlen:
        upexpend=upperBound-seqlen
        sec=sec[lowerBound-1:seqlen]
        ex=''
        for i in range(upexpend):
            ex=ex+"N"
        sec=sec+ex
    if lowerBound<=0:
        lowexpend=0-lowerBound+1
        sec = sec[0:upperBound]
        ex = ''
        for i in range(lowexpend):
            ex = ex + "N"
        sec = ex+sec
    if lowerBound>0 and upperBound<=seqlen:
        sec=sec[lowerBound-1:upperBound]
    de=len(sec)
    ONE_HOT_SIZE = 9
    # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
    Matr = np.zeros((1, window_size, ONE_HOT_SIZE))[0]
    AANo = 0
    try:
        for AA in sec:
            index = letterDict[AA]
            Matr[AANo][index] = 1
            AANo = AANo + 1
    except IndexError as e:
        print("debug")
    return Matr
