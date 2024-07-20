import torch
import os
import tempfile
import random
import string
import numpy
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
model_dir = Path('model/seqvec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1) # cuda_device=-1 for CPU
async def get_seqvecemb(seq,name):
    try:
        emb_tag = os.path.exists(f"./out/seqvecemb/{name}.npy")
        if emb_tag == False:
            seqleng=len(seq)
            embedding = embedder.embed_sentence(list(seq)) # List-of-Lists with shape [3,L,1024]
            npemb=protein_embd = torch.tensor(embedding).sum(dim=0)
            current_directory = os.getcwd()

            absolute_path = os.path.join(current_directory, f"./out/seqvecemb/{name}.npy")

            print("seqvecemb path:", absolute_path)
            numpy.save(f"./out/seqvecemb/{name}.npy",npemb)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got seqvecemb, return seqvecemb path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to get seqvecemb, return seqvecemb path", "result": None}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/seqvecemb/{name}.npy")
            print("seqvecemb path:", absolute_path)
            return {"err_code": 0, "err_desc": "got seqvecemb, return seqvecemb path", "result": absolute_path}

    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
