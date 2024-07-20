import os
from sklearn.utils import shuffle
import pandas as pd
import csv
def sum_p_count2(csv_path):
    y_count = 0
    s_count = 0
    t_count = 0
    error_count=0
    error_row=[]
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
    # with open(csv_path, 'r') as rf:
    #     reader = csv.reader(rf)
    #     next(reader)
    #     for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            pos=position-1
            print(f"pos: {sseq[pos]}")
            if (sseq[pos] == 'Y'):
                y_count = y_count + 1
            if (sseq[pos] == 'S'):
                s_count = s_count + 1
            if (sseq[pos] == 'T'):
                t_count = t_count + 1
            elif ((sseq[pos] != 'Y')and (sseq[pos] != 'S')and (sseq[pos] != 'T')):
                print(f"error pos: {sseq[pos]}")
                error_count=error_count+1
                error_row.append(row)
                df.drop(index, inplace=True)
    # save the processed DataFrame as a csv file.
    df.to_csv(csv_path, index=False)
    print(f"y:{y_count}")
    print(f"s:{s_count}")
    print(f"t:{t_count}")
    print(f"e:{error_count}")
    for e in error_row:
        print(e)
    return  s_count,t_count,y_count

def sum_p_count(csv_path):
    y_count = 0
    s_count = 0
    t_count = 0
    error_count=0
    error_row=[]
    with open(csv_path, 'r') as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            pos=position-1
            print(f"pos: {sseq[pos]}")
            if (sseq[pos] == 'Y'):
                y_count = y_count + 1
            if (sseq[pos] == 'S'):
                s_count = s_count + 1
            if (sseq[pos] == 'T'):
                t_count = t_count + 1
            elif ((sseq[pos] != 'Y')and (sseq[pos] != 'S')and (sseq[pos] != 'T')):
                print(f"error pos: {sseq[pos]}")
                error_count=error_count+1
                error_row.append(row)
    print(f"y:{y_count}")
    print(f"s:{s_count}")
    print(f"t:{t_count}")
    print(f"e:{error_count}")
    for e in error_row:
        print(e)
    return  s_count,t_count,y_count
def new_dataset_n(s_count,t_count,y_count,dataset_n,output_csv):
    row_s=[]
    row_y=[]
    row_t=[]
    with open(dataset_n, 'r') as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            pos=position-1
            print(f"pos: {sseq[pos]}")
            if (sseq[pos] == 'Y'):
                row_y.append(row)
            if (sseq[pos] == 'S'):
                row_s.append(row)
            if (sseq[pos] == 'T'):
                row_t.append(row)
    row_ss=row_s[:s_count]
    row_tt = row_t[:t_count]
    row_yy = row_y[:y_count]
    merged_list =row_ss+row_tt+row_yy
    if os.path.exists(output_csv):
        print(f"file {output_csv} already exists and will be overwritten.")
    # open csv
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'protein', 'pos', 'sequence'])
        for m in merged_list:
            writer.writerow(m)

    print("done")
def shuffledata(path):
    data = pd.read_csv(path)
    data = shuffle(data)
    data.to_csv(path,index=False)
def five_foldtest(dataset_p,dataset_n,outpath):
    path=outpath
    shuffledata(dataset_p)
    shuffledata(dataset_n)
    s_count,t_count,y_count=sum_p_count(dataset_p)
    new_dataset=path+"neg.csv"
    new_dataset_n(s_count,t_count,y_count,dataset_n,new_dataset)
    shuffledata(new_dataset)
    shuffledata(new_dataset)
    shuffledata(new_dataset)
    for i in range(5):
        train_csv_path=path + "train_set"+str(i)+".csv"
        test_csv_path=path + "test_set"+str(i)+".csv"
        if (os.path.exists(train_csv_path) == False):
            open(train_csv_path, 'w')
            de = pd.read_csv(train_csv_path, header=None, names=['label', 'protein', 'pos', 'sequence', 'gene'])
            de.to_csv(train_csv_path, index=False)
        if (os.path.exists(test_csv_path) == False):
            open(test_csv_path, 'w')
            de = pd.read_csv(test_csv_path, header=None, names=['label', 'protein', 'pos', 'sequence', 'gene'])
            de.to_csv(test_csv_path, index=False)
    dataset_pd = pd.read_csv(dataset_p)
    leng=len(dataset_pd)
    dataset_pd.head()
    dataset_nd = pd.read_csv(new_dataset)
    dataset_nd.head()
    dataset_nd=dataset_nd[:leng]
    pleng=int(len(dataset_pd)/5)
    nleng=int(len(dataset_nd)/5)
    testpdata1=dataset_pd[0*pleng:pleng]
    testpdata1.to_csv(path+"test_set1.csv", mode='a', header=False,index=False)
    trainpdata1=dataset_pd[pleng:]
    trainpdata1.to_csv(path + "train_set1.csv", mode='a', header=False,index=False)
    testpdata2=dataset_pd[1*pleng:pleng*2]
    testpdata2.to_csv(path + "test_set2.csv", mode='a', header=False,index=False)
    trainpdata2_1 = dataset_pd[0:1*pleng]
    trainpdata2_1.to_csv(path + "train_set2.csv", mode='a', header=False,index=False)
    trainpdata2_2=dataset_pd[pleng * 2:]
    trainpdata2_2.to_csv(path + "train_set2.csv", mode='a', header=False,index=False)
    testpdata3 = dataset_pd[2 * pleng:pleng * 3]
    testpdata3.to_csv(path + "test_set3.csv", mode='a', header=False,index=False)
    trainpdata3_1 = dataset_pd[0:2 * pleng]
    trainpdata3_1.to_csv(path + "train_set3.csv", mode='a', header=False,index=False)
    trainpdata3_2=dataset_pd[pleng * 3:]
    trainpdata3_2.to_csv(path + "train_set3.csv", mode='a', header=False,index=False)
    testpdata4 = dataset_pd[3 * pleng:pleng * 4]
    testpdata4.to_csv(path + "test_set4.csv", mode='a', header=False,index=False)
    trainpdata4_1 = dataset_pd[0:3 * pleng]
    trainpdata4_1.to_csv(path + "train_set4.csv", mode='a', header=False,index=False)
    trainpdata4_2=dataset_pd[pleng * 4:]
    trainpdata4_2.to_csv(path + "train_set4.csv", mode='a', header=False,index=False)
    testpdata5 = dataset_pd[4 * pleng:]
    testpdata5.to_csv(path + "test_set5.csv", mode='a', header=False,index=False)
    trainpdata5_1 = dataset_pd[0:4 * pleng]
    trainpdata5_1.to_csv(path + "train_set5.csv", mode='a', header=False,index=False)

    testndata1=dataset_nd[0*nleng:nleng]
    testndata1.to_csv(path+"test_set1.csv", mode='a', header=False,index=False)
    testndata1=dataset_nd[nleng:]
    testndata1.to_csv(path + "train_set1.csv", mode='a', header=False,index=False)
    testndata2=dataset_nd[1*nleng:nleng*2]
    testndata2.to_csv(path + "test_set2.csv", mode='a', header=False,index=False)
    testndata2_1 = dataset_nd[0:1*nleng]
    testndata2_1.to_csv(path + "train_set2.csv", mode='a', header=False,index=False)
    testndata2_2=dataset_nd[nleng * 2:]
    testndata2_2.to_csv(path + "train_set2.csv", mode='a', header=False,index=False)
    testndata3 = dataset_nd[2 * nleng:nleng * 3]
    testndata3.to_csv(path + "test_set3.csv", mode='a', header=False,index=False)
    testndata3_1 = dataset_nd[0:2 * nleng]
    testndata3_1.to_csv(path + "train_set3.csv", mode='a', header=False,index=False)
    testndata3_2=dataset_nd[nleng * 3:]
    testndata3_2.to_csv(path + "train_set3.csv", mode='a', header=False,index=False)
    testndata4 = dataset_nd[3 * nleng:nleng * 4]
    testndata4.to_csv(path + "test_set4.csv", mode='a', header=False,index=False)
    testndata4_1 = dataset_nd[0:3 * nleng]
    testndata4_1.to_csv(path + "train_set4.csv", mode='a', header=False,index=False)
    testndata4_2=dataset_nd[nleng * 4:]
    testndata4_2.to_csv(path + "train_set4.csv", mode='a', header=False,index=False)
    testndata5 = dataset_nd[4 * nleng:]
    testndata5.to_csv(path + "test_set5.csv", mode='a', header=False,index=False)
    testndata5_1 = dataset_nd[0:4 * nleng]
    testndata5_1.to_csv(path + "train_set5.csv", mode='a', header=False,index=False)
if __name__ == '__main__':
    # shuffledata("D:\lujiale\PtmDeep\deep\data\dataset/posset/Phosphorylation/Phosphorylation-T-existpdb.csv")
    root="D:\lujiale\PtmDeep\deep\data\dataset\cross_dataset_new/"
    max_len=2000
    tag="CAMK"
    tag2 = tag
    csv_pathp=root+f"{tag}\{max_len}\{tag2}.csv"
    csv_pathn = f"D:\lujiale\PtmDeep\deep\data\dataset/negset/negdataset-exitpdb_new_best.csv"
    outpath=root+f"{tag}\{max_len}/"
    # five_foldtest(csv_pathp,csv_pathn,outpath)
    sum_p_count(f"D:\lujiale\PtmDeep\deep\data\dataset\cross_dataset_new/{tag}/2000/test_set1.csv")