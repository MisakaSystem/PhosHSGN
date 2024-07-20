import torch
import os
import tempfile
import random
import string
import subprocess
from biotite.structure.io.pdb import PDBFile, get_structure
from Esm import EsmFold_model
async def get_pdb(seq,esm_model,name):
    try:
        pdb_tag = os.path.exists(f"./out/pdb/{name}.pdb")
        if pdb_tag == False:
            model=esm_model.model
            with torch.no_grad():
                output = model.infer_pdb(seq)
            with open(f"./out/pdb/{name}.pdb", "w") as f:
                f.write(output)
            
            import biotite.structure.io as bsio
            struct = bsio.load_structure(f"./out/pdb/{name}.pdb", extra_fields=["b_factor"])
            print(struct.b_factor.mean())  # this will be the pLDDT
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/pdb/{name}.pdb")

            print("PDB path:", absolute_path)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got pdb, return pdb path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to get pdb, return pdb path", "result": absolute_path}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/pdb/{name}.pdb")
            print("pdb path:", absolute_path)
             # Check pdb file fotmat
            pdbfile=PDBFile.read(absolute_path)
            try:
                structure=get_structure(pdbfile)[0]
            except ValueError:
                return {"err_code": -1, "err_desc": "pdb file incorrect", "result": absolute_path}

            return {"err_code": 0, "err_desc": "got pdb, return pdb path", "result": absolute_path}

    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
if __name__ == '__main__':
    ESM_m=EsmFold_model()
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    get_pdb(sequence,ESM_m)