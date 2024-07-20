# This program is used to randomly generate negative sets for 40% of non-homologous proteins
import csv
import os
import pickle
import random
import json
negproteinmap = '../data/dataset/negset/negproteinmap.pkl'
negdataset = '../data/dataset/negset/negdataset.csv'
fastadict = '../data/dataset/fastadict.pkl'
protein_faste = "../data/dataset/orgdataset/uniprotkb40_2.fasta"
def read_fasta_only_protein_position(fasta_path, split_char="!", id_field=0):
    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                seq = line.replace('>', '')
                seq = seq.replace('\n', '')
                seqdata=seq.split('_')
                postion=seqdata[-1]
                protein_name=seqdata[:-1]
                protein= '_'.join(protein_name)
                seqs[protein]=postion
    return seqs

def read_fasta_seqs(fasta_path="./../../data/dataset/orgdataset/uniprotkb.fasta", split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                protein_id=uniprot_id.split("|")[1]
                seqs[protein_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[protein_id] += seq
    # save
    with open('./../../data/dataset/orgdataset/uniprotkb.json', 'w') as f:
        json.dump(seqs, f)
    return seqs
def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''
    if (os.path.exists(negdataset)==False):
        with open(negdataset, "wb") as tf:
            tf.close()
    if (os.path.exists(negproteinmap) == False):
        with open(negproteinmap, "wb") as tf:
            pickle.dump({'':''}, tf)
            tf.close()
    try:
        with open(negproteinmap, "rb") as tf:
            my_map = pickle.load(tf)
            tf.close()
    except EOFError:
        return {}

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                protein_id=uniprot_id.split("|")[1]
                seqs[protein_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[protein_id] += seq
    for data in iter(seqs):
        seq = seqs[data]
        length = len(seq)
        postion = random.randint(1, length-1)
        key = data + ":" + str(postion)
        if (key in my_map):
            continue
        else:
            with open(negdataset, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['0', data, postion, seq])
            my_map[key] = seq
            print(key)
            with open(negproteinmap, "wb") as tf:
                pickle.dump(my_map, tf)
                tf.close()

    return seqs
def getseqbyid(protein_id,fasta_path,split_char="!", id_field=0):

    if (os.path.exists(fastadict) == False):
        with open(fastadict, "wb") as tf:
            pickle.dump({'': ''}, tf)
            tf.close()
        seqs = dict()
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                    # replace tokens that are mis-interpreted when loading h5
                    uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                    protein_id = uniprot_id.split("|")[1]
                    seqs[protein_id] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                    seq = ''.join(line.split()).upper().replace("-", "")
                    # repl. all non-standard AAs and map them to unknown/X
                    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                    seqs[protein_id] += seq
        with open(fastadict, "wb") as tf:
            pickle.dump(seqs, tf)
            tf.close()
    else:
        with open(fastadict, "rb") as tf:
            my_map = pickle.load(tf)
            tf.close()
        if(protein_id in my_map):
            return my_map[protein_id]
        else:
            return None
if __name__ == '__main__':
    read_fasta_seqs()