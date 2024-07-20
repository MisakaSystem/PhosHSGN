import os
import subprocess
import tempfile

async def get_pssm(seq,name):
    try:
        pssm_tag = os.path.exists(f"./out/pssm/{name}.pssm")
        if pssm_tag == False:
            # Create a temporary file and write image data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_file:
                temp_file.write(">sp\n".encode('utf-8'))
                temp_file.write(seq.encode('utf-8'))
                temp_file.close()
                file_name_all = os.path.basename(temp_file.name)
                path=temp_file.name
                file_directory = os.path.dirname(path)
                file_name = os.path.splitext(os.path.basename(path))[0]
                file_path_without_extension = os.path.join(file_directory, file_name)

            cmd=f'./model/ncbi-blast-2.14.1+/bin/psiblast -query {path}' + ' -db ./model/ncbi-blast-2.14.1+/bin/swissprot -evalue 0.001 -num_iterations 3' + f' -out_ascii_pssm out/pssm/{name}.pssm'
            print(cmd)
            p=subprocess.Popen(cmd,shell=True)
            return_code=p.wait()
            current_directory = os.getcwd()

            absolute_path = os.path.join(current_directory, f"out/pssm/{name}.pssm")

            print("PSSM path:", absolute_path)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got pssm, return pssm path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to got pssm, return pssm path", "result": None}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/pssm/{name}.pssm")
            print("pssm path:", absolute_path)
            return {"err_code": 0, "err_desc": "got pssm, return pssm path", "result": absolute_path}

    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
if __name__ == '__main__':
    get_pssm('SHMRPEPRLITILFSDIVGFTRMSNALQSQGVAELLNEYLGEMTRAVFENQGTVDKFVGDAI')
