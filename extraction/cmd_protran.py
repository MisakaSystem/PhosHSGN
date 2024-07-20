import torch
import os
import tempfile
import random
import string
import numpy
from transformers import T5Tokenizer, T5EncoderModel
from pathlib import Path
import re

device = torch.device('cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('model/prot/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("model/prot/prot_t5_xl_half_uniref50-enc").to(device)
async def get_protranemb(seq,name):
        # prepare your protein sequences as a list
    sequence_examples = [seq]
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    try:
        emb_tag = os.path.exists(f"./out/protranemb/{name}.npy")
        if emb_tag == False:
            seqleng=len(seq)
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            # generate embeddings
            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
            test=embedding_repr.last_hidden_state[0,:]
            emb_0 = embedding_repr.last_hidden_state[0,:seqleng] # shape (7 x 1024)
            npemb=emb_0
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/protranemb/{name}.npy")
            print("protranemb path:", absolute_path)
            numpy.save(f"./out/protranemb/{name}.npy",npemb)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got protranemb, return protranemb path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to get protranemb, return protranemb path", "result": None}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/protranemb/{name}.npy")
            print("protranemb path:", absolute_path)
            return {"err_code": 0, "err_desc": "got protranemb, return protranemb path", "result": absolute_path}

    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
