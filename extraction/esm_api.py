from cmd_protran import get_protranemb
from cmd_seqvec import get_seqvecemb
import uvicorn
import asyncio
import json
import os
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from typing import Dict, Tuple
from typing import List
from cmd_pdb import get_pdb
from cmd_pssm import get_pssm
from cmd_seqemb import get_seqemb
from cmd_dssp import get_dssp
from cmd_cmap import get_cmap,get_msacmap
from Esm import EsmFold_model,Esm2_model,Esm_MSA_cmap_model
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ESM_m=''
ESM2_m=''

@app.post("/api/esm")
async def esm_process(request: Request):
    max_retries = 3
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    seq = json_post_list.get('seq')
    seq_type = json_post_list.get('seq_type')
    name = json_post_list.get('name')
    if seq_type=="pdb":
        for retry_count in range(max_retries):
            res=await get_pdb(seq,ESM_m,name)
            if res['err_code']==0:
                pdb_path=res['result']
                return FileResponse(pdb_path, headers={"Content-Disposition": "attachment"})
            elif res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"pdb error:{res}, retry count: {retry_count}")
                os.remove(res['result'])
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return res
    if seq_type=="dssp":
        for retry_count in range(max_retries):
            res=await get_pdb(seq,ESM_m,name)
            if res['err_code']==0:
                pdb_path=res['result']
            elif res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"pdb error:{res}, retry count: {retry_count}")
                os.remove(res['result'])
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            dssp_res=await get_dssp(pdb_path,name)
            if dssp_res['err_code']==0:
                dssp_path=dssp_res['result']
                return FileResponse(dssp_path, headers={"Content-Disposition": "attachment"})
            elif dssp_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"dssp error:{dssp_res}, retry count: {retry_count}")
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return dssp_res
    if seq_type=="emb":
        for retry_count in range(max_retries):
            seq_res=await get_seqemb(seq,ESM2_m,name)
            if seq_res['err_code']==0:
                seqemb_path=seq_res['result']
                return FileResponse(seqemb_path, headers={"Content-Disposition": "attachment"})
            elif seq_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"emb error:{seq_res}, retry count: {retry_count}")
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return seq_res
    if seq_type=="seqvecemb":
        for retry_count in range(max_retries):
            seq_res=await get_seqvecemb(seq,name)
            if seq_res['err_code']==0:
                seqemb_path=seq_res['result']
                return FileResponse(seqemb_path, headers={"Content-Disposition": "attachment"})
            elif seq_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"emb error:{seq_res}, retry count: {retry_count}")
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return seq_res
    if seq_type=="protemb":
        for retry_count in range(max_retries):
            seq_res=await get_protranemb(seq,name)
            if seq_res['err_code']==0:
                seqemb_path=seq_res['result']
                return FileResponse(seqemb_path, headers={"Content-Disposition": "attachment"})
            elif seq_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"emb error:{seq_res}, retry count: {retry_count}")
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return seq_res
    if seq_type=="pssm":
        for retry_count in range(max_retries):
            pssm_res=await get_pssm(seq,name)
            if pssm_res['err_code']==0:
                seqpssm_path=pssm_res['result']
                return FileResponse(seqpssm_path, headers={"Content-Disposition": "attachment"})
            elif pssm_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"pssm error:{pssm_res}, retry count: {retry_count}")
                # If err_code is -1, and it's not the last retry, sleep for a while and then retry
                await asyncio.sleep(1)  # Adjust the delay as needed
                continue
            else:
                return pssm_res
    if seq_type=="cmap":
        for retry_count in range(max_retries):
            res=await get_pdb(seq,ESM_m,name)
            if res['err_code']==0:
                pdb_path=res['result']
            elif res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"cmap_pdb error:{res}, retry count: {retry_count}")
                os.remove(res['result'])
                continue
            cmap_res=await get_cmap(pdb_path,name)
            if cmap_res['err_code']==0:
                seqcmap_path=cmap_res['result']
                return FileResponse(seqcmap_path, headers={"Content-Disposition": "attachment"})
            elif cmap_res['err_code'] == -1 and retry_count < max_retries - 1:
                print(f"cmap error:{cmap_res}, retry count: {retry_count}")
                os.remove(cmap_res['result'])
                continue
            else:
                return cmap_res
    if seq_type=="cmap_predict":
        res=await get_msacmap(seq,Esm_MSA_m,name)
        if res['err_code']==0:
            msacmappath=res['result']
            return FileResponse(msacmappath, headers={"Content-Disposition": "attachment"})
        else:
            return res
if __name__ == "__main__": 
    ESM_m=EsmFold_model(chunk_size=1,cpu_only=True,cpu_offload=True)
    ESM2_m=Esm2_model()
    Esm_MSA_m=Esm_MSA_cmap_model()

    uvicorn.run(app, host="0.0.0.0", port=8002)