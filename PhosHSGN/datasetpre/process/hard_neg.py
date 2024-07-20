import csv
import os
import random
def product_neg(old_neg,new_neg):
    if os.path.exists(new_neg):
        print(f"file {new_neg} already exists and will be overwritten.")
    with open(new_neg, 'w', newline='') as file:
        writer = csv.writer(file)
        with open(old_neg, 'r') as rf:
            reader = csv.reader(rf)
            # head=next(reader)
            # writer.writerow(head)
            for row in reader:
                sseq = row[3]
                protein = row[1]
                position = int(row[2])
                label = int(row[0])
                positions = []
                print(row)
                try:
                    for index, char in enumerate(sseq):
                        if char in ['Y']:
                            positions.append(index+1)
                    position=random.choice(positions)
                    writer.writerow([row[0],row[1],position,row[3]])
                except IndexError:
                    continue
if __name__ == '__main__':
    product_neg("D:\lujiale\PtmDeep\deep\data\dataset/negset/negdataset-exitpdb_new.csv","D:\lujiale\PtmDeep\deep\data\dataset/negset/negdataset-exitpdb_new_best_Y.csv")
