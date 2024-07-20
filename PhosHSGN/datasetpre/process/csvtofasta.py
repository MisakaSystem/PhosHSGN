import csv
import pandas as pd

def fromcsvtofasta(path,out_path):
    with open(path, 'r', encoding="utf-8") as rf:
        reader = csv.reader(rf)
        for row in reader:
            print(row)
            cmdata = []
            upid = row[1]
            seq=row[3]
            label=int(row[0])
            data=[]
            data.append(upid)
            data.append(seq)
            file = open(out_path,"a")
            file.writelines([">"+upid+"\n",seq+"\n"])
            file.close()
        else:
            print(upid)
def fromcsvtofasta_win(path,out_pospath,out_negpath,window_size=25):
    with open(path, 'r', encoding="utf-8") as rf:
        reader = csv.reader(rf)
        half_len = int((window_size - 1) / 2)
        next(reader)
        for row in reader:
            print(row)
            cmdata = []
            upid = row[1]
            sec=row[3]
            label=int(row[0])
            position=int(row[2])
            centerno = position
            print("center:", centerno)
            lowerBound = centerno - int(half_len)
            upperBound = centerno + int(half_len)
            upexpend = 0
            lowexpend = 0
            seqlen = len(sec)
            if upperBound > seqlen and lowerBound <= 0:
                try:
                    upexpend = upperBound - seqlen
                    lowexpend = 0 - lowerBound + 1
                    sec = sec[0:seqlen]
                    upexpendex = ''
                    lowexpendex = ''
                    for i in range(upexpend):
                        upexpendex = upexpendex + "-"
                    sec = sec + upexpendex
                    for i in range(lowexpend):
                        lowexpendex = lowexpendex + "-"
                    sec = lowexpendex + sec
                except:
                    print("debug")
            if upperBound > seqlen:
                upexpend = upperBound - seqlen
                sec = sec[lowerBound - 1:seqlen]
                ex = ''
                for i in range(upexpend):
                    ex = ex + "-"
                sec = sec + ex
            if lowerBound <= 0:
                lowexpend = 0 - lowerBound + 1
                sec = sec[0:upperBound]
                ex = ''
                for i in range(lowexpend):
                    ex = ex + "-"
                sec = ex + sec
            if lowerBound > 0 and upperBound <= seqlen:
                sec = sec[lowerBound - 1:upperBound]

            sentence = sec
            if label==1:
                file = open(out_pospath,"a")
                file.writelines([">"+upid+"\n",sentence+"\n"])
                file.close()
            if label==0:
                file = open(out_negpath,"a")
                file.writelines([">"+upid+"\n",sentence+"\n"])
                file.close()
        else:
            print(upid)
def fromcsvto_maxlen_csv(path,out_path,maxlen):
    with open(path, 'r', encoding="utf-8") as rf:
        reader = csv.reader(rf)
        for row in reader:
            print(row)
            cmdata = []
            upid = row[1]
            seq=row[3]
            label=int(row[0])
            length=len(seq)
            if length<maxlen:
                with open(out_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    print(f"保存蛋白：{row}")
        else:
            print(upid)
def checkcsv(path):
    print("checkcsv")
    df = pd.read_csv(path)
    for index, row in df.iterrows():
            cmdata = []
            upid = row[1]
            seq=row[3]
            pos=int(row[2])
            label=int(row[0])
            length=len(seq)
            if pos>length:
                print(upid)
                print(f"pos {pos} length {length}")
                df.drop(index, inplace=True)
                print(f"删除蛋白:[{upid}] {seq}")
    df.to_csv(path, index=False)
    print("done")

if __name__ == '__main__':
    csvpath="D:\lujiale\PtmDeep\deep\data\dataset\cross_dataset\Atypical/1500/Atypical.csv"
    checkcsv(csvpath)
