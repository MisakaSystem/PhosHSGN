import torch
import os
import tempfile
import random
import string
from Esm import Esm2_model
import numpy 
async def get_seqemb(seq,esm2_model,name):
    try:
        emb_tag = os.path.exists(f"./out/emb/{name}.npy")
        if emb_tag == False:
            seq1=''
            seq2=''
            seqleng=len(seq)
            if seqleng>1000 and seqleng<=2000:
                print(f"seq len: {len(seq)} > 1000 slices")
                half=int(seqleng/2)
                seq1=seq[0:half]
                seq2=seq[half:seqleng]
                print(f"clip1Len: {len(seq1)} clip2Len: {len(seq2)}")
            elif seqleng>0 and seqleng<=1000:
                seq1=seq
            else:
                return {"err_code": -1, "err_desc": f"failed to get emb, len > 2000 len: {seqleng}", "result": None}
            model=esm2_model.model
            alphabet =esm2_model.alphabet 
            batch_converter = alphabet.get_batch_converter()
            data=[(name,seq1)]
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
                data2=[(name,seq2)]
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
                merged_array = numpy.concatenate((npemb, npemb2))
                print("Merged FS: ",merged_array.shape)
                npemb=merged_array
            current_directory = os.getcwd()

            absolute_path = os.path.join(current_directory, f"./out/emb/{name}.npy")

            print("emb path:", absolute_path)
            numpy.save(f"./out/emb/{name}.npy",npemb)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got emb, return emb path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to get emb, return emb path", "result": None}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/emb/{name}.npy")
            print("emb path:", absolute_path)
            return {"err_code": 0, "err_desc": "got emb, return emb path", "result": absolute_path}

    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
if __name__ == '__main__':
    ESM_m=Esm2_model()
    sequence = "MKTVEE"
    get_seqemb(sequence,ESM_m)