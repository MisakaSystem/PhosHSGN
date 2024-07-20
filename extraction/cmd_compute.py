import argparse
import csv
import subprocess
import tempfile
import os
import numpy as np
from cmd_cmap import contacts_from_pdb
from biotite.structure.io.pdb import PDBFile, get_structure
from allennlp.commands.elmo import ElmoEmbedder
from Esm import Esm2_model
import time
import torch
import requests
import torch
import os
import tempfile
import random
import string
import numpy
from transformers import T5Tokenizer, T5EncoderModel
from pathlib import Path
import re

def main():
    count=0
    model_dir = Path('model/seqvec')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options,weights, cuda_device=3) # cuda_device=-1 for CPU
    device = torch.device('cuda:2')
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('model/prot/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Load the model
    prot_model = T5EncoderModel.from_pretrained("model/prot/prot_t5_xl_half_uniref50-enc").to(device)
    esm2_model=Esm2_model()
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='CSV file path',required=True)
    args = parser.parse_args()
    for root, dirs, files in os.walk(args.csv):
        for csv_name in files:
            csv_path = os.path.join(root, csv_name)
            print(f'CSV file path: {csv_path}')
            current_directory = os.getcwd()
            with open(csv_path, 'r') as rf:
                reader = csv.reader(rf)
                next(reader)
                for row in reader:
                    sseq = row[3]
                    protein = row[1]
                    position = int(row[2])
                    label=int(row[0])
                    pdb_path=f"./out/pdb/{protein}.pdb"
                    dssp_path=f"./out/dssp/{protein}.dssp"
                    pssm_path=f"./out/pssm/{protein}.pssm"
                    cmap_path=f"./out/cmap/{protein}.npy"
                    emb_path=f"./out/emb/{protein}.npy"
                    protranemb_path=f"./out/protranemb/{protein}.npy"
                    seqvecemb_path=f"./out/seqvecemb/{protein}.npy"
                    pdb_absolute_path = os.path.join(current_directory, pdb_path)
                    dssp_absolute_path = os.path.join(current_directory, dssp_path)
                    pssm_absolute_path = os.path.join(current_directory, pssm_path)
                    cmap_absolute_path = os.path.join(current_directory,cmap_path)
                    emb_absolute_path = os.path.join(current_directory,emb_path)
                    protranemb_absolute_path = os.path.join(current_directory,protranemb_path)
                    seqvecemb_absolute_path = os.path.join(current_directory,seqvecemb_path)
                    # Check if pdb exists
                    if not os.path.exists(pdb_absolute_path):
                        print(f"pdb not exist, path: {pdb_absolute_path}")
                        exit()
                    if not os.path.exists(cmap_absolute_path):
                        print(f'{cmap_absolute_path} not exist')
                        pdbfile=PDBFile.read(pdb_absolute_path)
                        try:
                            structure=get_structure(pdbfile)[0]
                        except ValueError:
                            return {"err_code": -1, "err_desc": "failed to fetch cmap, return cmap path", "result": pdb_path}
                        contacts=contacts_from_pdb(structure)
                        print(type(contacts))
                        npmascmap=contacts
                        np.save(cmap_absolute_path,npmascmap)
                        print(f"cmap saved {protein}.npy path: {cmap_absolute_path}")
                    if not os.path.exists(dssp_absolute_path):
                        print(f'{dssp_absolute_path} not exist')
                        cmd=f'mkdssp -i {pdb_absolute_path} -o {dssp_absolute_path}'
                        p=subprocess.Popen(cmd,shell=True)
                        return_code=p.wait()
                        print(f"dssp saved {protein}.dssp path: {dssp_absolute_path}")
                    
                    if not os.path.exists(pssm_absolute_path):
                        print(f'{pssm_absolute_path} not exist')
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_file:
                            temp_file.write(">sp\n".encode('utf-8'))
                            temp_file.write(sseq.encode('utf-8'))
                            temp_file.close()
                            file_name_all = os.path.basename(temp_file.name)
                            path=temp_file.name
                            file_directory = os.path.dirname(path)
                            file_name = os.path.splitext(os.path.basename(path))[0]
                            print(f"tmp file path: {path}")
                            file_path_without_extension = os.path.join(file_directory, file_name)
                        cmd=f'./model/ncbi-blast-2.14.1+/bin/psiblast -query {path}' + ' -db ./model/ncbi-blast-2.14.1+/bin/swissprot -evalue 0.001 -num_iterations 3' + f' -out_ascii_pssm {pssm_absolute_path}'
                        p=subprocess.Popen(cmd,shell=True)
                        return_code=p.wait()
                        print(f"pssm saved {protein}.pssm path: {pssm_absolute_path}")
                    if not os.path.exists(protranemb_absolute_path):
                        input_ids=None
                        print(f'{protranemb_absolute_path} not exist')
                        seqleng=len(sseq)
                        sequence_examples = [sseq]
                        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
                        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
                        # tokenize sequences and pad up to the longest sequence in the batch
                        ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

                        input_ids = torch.tensor(ids['input_ids']).to(device)
                        attention_mask = torch.tensor(ids['attention_mask']).to(device)
                        # generate embeddings
                        with torch.no_grad():
                            embedding_repr = prot_model(input_ids=input_ids, attention_mask=attention_mask)
                        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
                        test=embedding_repr.last_hidden_state[0,:]
                        emb_0 = embedding_repr.last_hidden_state[0,:seqleng] # shape (7 x 1024)
                        npemb=emb_0.cpu()
                        numpy.save(protranemb_absolute_path,npemb)
                        print(f"protemb saved {protein}.npy path: {protranemb_absolute_path}")
                    if not os.path.exists(seqvecemb_absolute_path):
                        print(f'{seqvecemb_absolute_path} not exist')
                        seqleng=len(sseq)
                        embedding = embedder.embed_sentence(list(sseq)) # List-of-Lists with shape [3,L,1024]
                        npemb=torch.tensor(embedding).sum(dim=0)
                        numpy.save(seqvecemb_absolute_path,npemb.cpu())
                        print(f"seqvecemb saved {protein}.npy path: {seqvecemb_absolute_path}")
                    if not os.path.exists(emb_absolute_path):
                        print(f'{emb_absolute_path} not exist')
                        seq1=''
                        seq2=''
                        seqleng=len(sseq)
                        print(f"seq len: {len(sseq)}")
                        if seqleng>1000 and seqleng<=2000:
                            print(f"seq len: {len(sseq)} > 1000, slices")
                            half=int(seqleng/2)
                            seq1=sseq[0:half]
                            seq2=sseq[half:seqleng]
                            print(f"clip1Len: {len(seq1)} clip2Len: {len(seq2)}")
                        elif seqleng>0 and seqleng<=1000:
                            seq1=sseq
                        else:
                            print("error")
                            return {"err_code": -1, "err_desc": "failed to fetch emb, return emb path", "result": None}
                        model=esm2_model.model
                        alphabet =esm2_model.alphabet 
                        batch_converter = alphabet.get_batch_converter()
                        data=[(protein,seq1)]
                        batch_labels, batch_strs, batch_tokens = batch_converter(data)
                        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                            # Extract per-residue representations (on CPU)
                        with torch.no_grad():
                            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                        token_representations = results["representations"][33]
                        tokens_len=batch_lens[0]
                        sequence_representations =token_representations[0][1 : tokens_len - 1]
                        npemb=sequence_representations.data.cpu().numpy()
                        if len(seq2)>0:
                            data2=[(protein,seq2)]
                            batch_labels2, batch_strs2, batch_tokens2 = batch_converter(data2)
                            batch_lens2 = (batch_tokens2 != alphabet.padding_idx).sum(1)
                                # Extract per-residue representations (on CPU)
                            with torch.no_grad():
                                results2 = model(batch_tokens2, repr_layers=[33], return_contacts=True)
                            token_representations2 = results2["representations"][33]
                            tokens_len2=batch_lens2[0]
                            sequence_representations2 =token_representations2[0][1 : tokens_len2 - 1]
                            npemb2=sequence_representations2.data.cpu().numpy()
                            print("Clip 1 Feature Shape: ",npemb.shape)
                            print("Clip 2 Feature Shape: ",npemb2.shape)
                            merged_array = np.concatenate((npemb, npemb2))
                            print("Merged FS: ",merged_array.shape)
                            npemb=merged_array

                        np.save(emb_absolute_path,npemb)
                        print(f"emb saved {protein}.npy path: {emb_absolute_path}")
                    count=count+1
                    print(f"[{protein}] Checke csv path: {csv_path} Done: {count}")
if __name__ == '__main__':
    main()
