import os
import random
import tempfile
import string
import subprocess
async def get_dssp(pdb_path,name):
    try:
        # Check if dssp exists
        dssp_tag = os.path.exists(f"./out/dssp/{name}.dssp")
        if dssp_tag == False:
            cmd=f'mkdssp -i {pdb_path} -o ./out/dssp/{name}.dssp'
            print(cmd)
            p=subprocess.Popen(cmd,shell=True)
            return_code=p.wait()
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/dssp/{name}.dssp")
            print("dssp path:", absolute_path)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got dssp, return dssp path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "fetch to get dssp, return dssp path", "result": None}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/dssp/{name}.dssp")
            print("dssp path:", absolute_path)
            return {"err_code": 0, "err_desc": "got dssp, return dssp path", "result": absolute_path}
    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
if __name__ == '__main__':
    get_dssp('/data1/lujiale/esm/out/pdb/MGFLL.pdb')