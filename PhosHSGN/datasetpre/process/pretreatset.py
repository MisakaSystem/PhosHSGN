# This program is used to classify PTMVar.xlsx into various csv
import csv
import hashlib
import os
import sys
import pickle
import time
import pandas as pd
from lxml import etree

import requests
import urllib3

from datasetpre.process.negfastacsv import read_fasta, read_fasta_seqs

urllib3.disable_warnings()


header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36",
}
origen_dataset="D:\lujiale\deep\data\dataset\orgdataset\PTM.csv"
root="../"
categorymappath = root + 'data/dataset/posset/categorymap.pkl'
def creat_categoryfile():
    if(os.path.exists(root+'data/dataset/posset/'+"Ubiquitylation")==False):
        os.makedirs(root+'data/dataset/posset/Ubiquitylation')
        open(root+'data/dataset/posset/Ubiquitylation/Ubiquitylation-K.csv', 'w')
    if(os.path.exists(root+'data/dataset/posset/'+"Phosphorylation")==False):
        os.makedirs(root+'data/dataset/posset/Phosphorylation')
        open(root+'data/dataset/posset/Phosphorylation/Phosphorylation-S.csv', 'w')
        open(root + 'data/dataset/posset/Phosphorylation/Phosphorylation-T.csv', 'w')
        open(root + 'data/dataset/posset/Phosphorylation/Phosphorylation-Y.csv', 'w')
    if(os.path.exists(root+'data/dataset/posset/'+"Sumoylation")==False):
        os.makedirs(root+'data/dataset/posset/Sumoylation')
        open(root+'data/dataset/posset/Sumoylation/Sumoylation-K.csv', 'w')
    if(os.path.exists(root+'data/dataset/posset/'+"Acetylation")==False):
        os.makedirs(root+'data/dataset/posset/Acetylation')
        open(root+'data/dataset/posset/Acetylation/Acetylation-K.csv', 'w')
    if(os.path.exists(root+'data/dataset/posset/'+"Succinylation")==False):
        os.makedirs(root+'data/dataset/posset/Succinylation')
        open(root+'data/dataset/posset/Succinylation/Succinylation-K.csv', 'w')
    if (os.path.exists(root + 'data/dataset/posset/' + "Neddylation") == False):
        os.makedirs(root + 'data/dataset/posset/Neddylation')
        open(root + 'data/dataset/posset/Neddylation/Neddylation-K.csv', 'w')
    if (os.path.exists(root + 'data/dataset/posset/' + "Methylation") == False):
        os.makedirs(root + 'data/dataset/posset/Methylation')
        open(root + 'data/dataset/posset/Methylation/Methylation-R.csv', 'w')
    if (os.path.exists(categorymappath) == False):
        with open(categorymappath, "wb") as tf:
            pickle.dump({'':''}, tf)
            tf.close()
def writedatatocsv(data):
    with open(categorymappath, "rb") as tf:
        my_map = pickle.load(tf)
        tf.close()
    key=data["upid"]+":"+data["postion"]+":"+data['mod_type']
    value=data["category"]
    if(key in my_map):
        return
    # save csv
    gene=data['gene']
    upid=data['upid']
    postion=data['postion']
    category=data['category']
    mod_type=data['mod_type']
    seq=getseqbyid(upid,root+"data/dataset/orgdataset/uniprotkb.fasta")
    if seq==None:
        try:
            seq=getfastatocsv_with_retry("https://www.uniprot.org/uniprot/",upid,postion)[3]
        except IndexError as e:
            return
    path=root+'data/dataset/posset/'+mod_type+"/"+mod_type+'-'+category+".csv"
    # dataframe = pd.DataFrame({'label': 1, 'protein': upid, 'pos': postion, 'sequence':seq,'gene':gene})
    with open(path, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['1', upid, postion, seq, gene])
        print("save protein: "+"["+upid+']:'+postion+":"+seq+":"+gene)
    # save DataFrame as csv, index means whether to display row names，default=True
    my_map[key]=value
    with open(categorymappath, "wb") as tf:
        pickle.dump(my_map, tf)
        tf.close()
def readdatafromcsv(datasetpath):
    with open(datasetpath, 'r',encoding="utf-8") as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            gene = row[0]
            upid= row[1]
            postion=row[14]
            category=row[15]
            mod_type=row[17]
            proteindata={'gene':gene,"upid":upid,"postion":postion,"category":category,"mod_type":mod_type}
            writedatatocsv(proteindata)
proxies={
'http': 'http://127.0.0.1:7890',
'https': 'https://127.0.0.1:7890'
}
# fetch datas from uniprot
def getfastatocsv_with_retry(url, protein, position, max_retries=1):
    for retry in range(max_retries):
        try:
            retParam = []
            retParam.append(protein)
            retParam.append(position)

            md5 = hashlib.md5()
            md5.update((protein + position).encode(encoding='utf-8'))
            current_directory = os.getcwd()
            cache_folder = current_directory+'/cache'  # cache path
            path = os.path.join(cache_folder, md5.hexdigest())  # cache path

            if not os.path.isfile(path):
                resp = requests.get(url + protein + ".xml", verify=False,proxies=proxies)
                # make sure the cache folder exists
                os.makedirs(os.path.dirname(path), exist_ok=True)

                with open(path, "wb") as dw:
                    dw.write(resp.content)
                resp.close()

            stat_info = os.stat(path)
            if stat_info.st_size < 64:
                print('Hit cache: [protein=%s %s]' % (protein, position))
                return retParam

            parser = etree.XMLParser(encoding="utf-8")
            tree = etree.parse(path, parser)
            root = tree.getroot()[0]

            gene = root.findall('{http://uniprot.org/uniprot}gene')
            if len(gene) > 0:
                retParam.append(gene[0][0].text)
            else:
                retParam.append('')
            seq = root.findall('{http://uniprot.org/uniprot}sequence')
            if len(seq) > 0:
                retParam.append(seq[0].text)
            else:
                return None

            entry = root.findall('{http://uniprot.org/uniprot}accession')
            name = entry[0].text
            retParam.append(name)
            print(f"Fetch protein: {retParam}")
            return retParam
        except Exception as r:
            print('Skip error: [protein=%s] %s' % (protein, r))
            time.sleep(1)
            print(f"Retry count: {retry}")
            if retry < max_retries - 1:
                print(f'Retrying ({retry + 1}/{max_retries})...')
            else:
                print('Max retries reached. Skipping.')
                return retParam
def delete_protein_from_pdb(csv_path, pdb_fold_path):
    # read csv to pandas DataFrame
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        upid = row[1]
        seq = row[3]
        label = int(row[0])
        print(f"check protein [{upid}]")
        pdb_tag = os.path.exists(pdb_fold_path + upid + ".pdb")
        if pdb_tag == False:
            print(f"del protein: [{upid}]")
            try:
                df.drop(index, inplace=True)
            except TypeError:
                continue
    # save
    df.to_csv(csv_path, index=False)
def delete_ptm_from_fold(fold_path):
    zhaohuishu=0
    deletecount=0
    protein_faste = "./../../data/dataset/orgdataset/uniprotkb.fasta"
    seqdict=read_fasta_seqs(protein_faste)
    for root, dirs, files in os.walk(fold_path):
        for file in files:
            csvfile=os.path.join(root,file)
            df = pd.read_csv(csvfile)
            for index, row in df.iterrows():
                upid = row[1]
                seq = row[3]
                if isinstance(seq, float):
                    if isinstance(upid, float)==False:
                        if upid in seqdict:
                            seq = seqdict[upid]
                            df.at[index, 'sequence'] = seq
                            zhaohuishu=zhaohuishu+1
                        else:
                            df.drop(index, inplace=True)
                            print(f"del protein: [{upid}] {seq}")
                            deletecount=deletecount+1
                    else:
                        df.drop(index, inplace=True)
                        print(f"del protein: [{upid}] {seq}")
                        deletecount = deletecount + 1
                    continue
                check=seq[:8]
                length=len(seq)
                if check=='No entry':
                    if isinstance(upid, float)==False:
                        if upid in seqdict:
                            seq = seqdict[upid]
                            df.at[index, 'sequence'] = seq
                            zhaohuishu=zhaohuishu+1
                        else:
                            df.drop(index, inplace=True)
                            print(f"del protein: [{upid}] {seq}")
                            deletecount = deletecount + 1

                    else:
                        df.drop(index, inplace=True)
                        print(f"del protein: [{upid}] {seq}")
                        deletecount = deletecount + 1
                    continue
                print(f"check  protein [{upid}] len: {length}")
            df.to_csv(csvfile, index=False)
        print(f"done, found: {zhaohuishu}, deleted: {deletecount}")
if __name__ == '__main__':
    # creat_categoryfile()
    # data=readdatafromcsv(origen_dataset)
    delete_ptm_from_fold("D:\lujiale\PtmDeep\deep\data\dataset\ptm_pos")