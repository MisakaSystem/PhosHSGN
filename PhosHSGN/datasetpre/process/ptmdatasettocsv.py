import json
import os
import csv
import sys
serve_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(serve_path)
sys.path.append(serve_path)
from process.negfastacsv import read_fasta_only_protein_position
from pretreatset import getfastatocsv_with_retry, delete_protein_from_pdb
import pandas as pd

# assume you have defined a function called to_csv to process .fasta files

def ptm_parese():
    directory_path = 'D:\lujiale\PtmDeep\deep\data\dataset\ptm'
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for fasta_file in os.listdir(dir_path):
                filename = fasta_file.split('.')[0] + '.csv'
                csv_file_path = os.path.join(dir_path, filename)
                print(f"create csv: {csv_file_path}")
                if fasta_file.endswith(".fasta"):
                    fasta_file_path = os.path.join(dir_path, fasta_file)
                    seqs = read_fasta_only_protein_position(fasta_file_path)
                    with open(csv_file_path, 'w', newline='') as csvfile:

                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['label', "protein", 'pos', 'sequence'])
                        for seq in seqs:
                            upid = seq
                            postion = seqs[upid]
                            result = getfastatocsv_with_retry("https://www.uniprot.org/uniprotkb/", upid, postion)
                            try:
                                if result == None:
                                    continue
                                name = result[4]
                                pos = postion
                                sseq = result[3]
                                length = len(sseq)
                                if int(pos) >= length or int(pos) <= 0 or length > 4000:
                                    continue
                                if 'neg' in fasta_file:
                                    label = 0
                                if 'pos' in fasta_file:
                                    label = 1
                                print(f"write protein: {label, name, postion, sseq}")
                                csv_writer.writerow([label, name, postion, sseq])
                            except BaseException:
                                continue

def ptm_dataset_par():
    headers = ["label","protein",'pos','sequence']
    dataset_path="data/dataset/orgdataset/dataset.csv"
    output_path='data/dataset/ptm/'
    family_group_values = []
    with open(dataset_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            family_group_values.append(row['Family'])
            family_group_values.append(row['Group'])

    for value in list(set(family_group_values)):
        sanitized_value = value.replace("?", "")
        sanitized_value = sanitized_value.replace("/", "_")
        file_path = os.path.join(output_path, f"{sanitized_value}.csv")
        if os.path.exists(file_path):
            print(f"File {file_path} already exists and will be overwritten")
        with open(file_path, 'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    with open('data/dataset/orgdataset/uniprotkb.json', 'r') as f:
        loaded_dict = json.load(f)
    with open(dataset_path, 'r', encoding="utf-8") as rf:
        reader = csv.reader(rf)
        for row in reader:
            uid=row[2]
            postion=row[3]
            family=row[8]
            group=row[9]
            if uid in loaded_dict:
                seq=loaded_dict[uid]
            else:
                try:
                    result = getfastatocsv_with_retry("https://www.uniprot.org/uniprotkb/", uid, postion)
                    uid=result[4]
                    seq=result[3]
                except BaseException:
                    continue
            if seq=='':
                continue
            if len(seq)>=1500:
                print(f"protein {uid} > 1500, len: {len(seq)}")
                continue
            family_sanitized_value = family.replace("?", "")
            family_sanitized_value = family_sanitized_value.replace("/", "_")
            group_sanitized_value = group.replace("?", "")
            group_sanitized_value = group_sanitized_value.replace("/", "_")
            family_file_path = os.path.join(output_path, f"{family_sanitized_value}.csv")
            group_file_path = os.path.join(output_path, f"{group_sanitized_value}.csv")

            with open(family_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['1', uid, postion, seq])
            with open(group_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['1', uid, postion, seq])
            print(f"{uid}:done {['1', uid, postion, seq]}")
    for value in list(set(family_group_values)):
        sanitized_value = value.replace("?", "")
        sanitized_value = sanitized_value.replace("/", "_")
        file_path = os.path.join(output_path, f"{sanitized_value}.csv")
        frame = pd.read_csv(file_path, engine='python')
        data = frame.drop_duplicates(subset=['protein','pos'], keep='first', inplace=False)
        data = data.drop_duplicates(subset=['pos','sequence'], keep='first', inplace=False)
        data.to_csv(file_path, encoding='utf8',index=False)
        print(f'{file_path}: clean up duplicate rows')
    with open(output_path+"batch.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for value in list(set(family_group_values)):
            sanitized_value = value.replace("?", "")
            sanitized_value = sanitized_value.replace("/", "_")
            file_path = os.path.join(output_path, f"{sanitized_value}.csv")
            with open(file_path, 'r', encoding="utf-8") as rf:
                reader = csv.reader(rf)
                next(reader)
                for row in reader:
                    label=row[0]
                    name=row[1]
                    postion=row[2]
                    sseq=row[3]
                    print(f"write protein: {label, name, postion, sseq}")
                    writer.writerow([label, name, postion, sseq])
    for value in list(set(family_group_values)):
        sanitized_value = value.replace("?", "")
        sanitized_value = sanitized_value.replace("/", "_")
        file_path = os.path.join(output_path, f"{sanitized_value}.csv")
        delete_protein_from_pdb(file_path,'data/produce/pdb_v4/')
if __name__ == '__main__':
    ptm_dataset_par()





