import os

import numpy
import requests
import random
import string
def get_emb(seq,name):
    max_retries = 3
    emb_tag = os.path.exists(f"data/emb/{name}.npy")
    if emb_tag == False:
        url = 'http://192.168.102.18:8002/api/esm'

        data = {'seq': seq,'name':name, 'seq_type': 'emb'}
        for retry_count in range(max_retries):
            print(f"got protein: {name} emb count: {retry_count}")
            response = requests.post(url, json=data)

            if response.status_code == 200:
                with open(f'data/emb/{name}.npy', 'wb') as f:
                    f.write(response.content)
                data=numpy.load(f"data/emb/{name}.npy")
                return data
            return None
    else:
        data = numpy.load(f"data/emb/{name}.npy")
        return data
def get_seqvecemb(seq,name):
    max_retries = 3
    emb_tag = os.path.exists(f"data/seqvecemb/{name}.npy")
    if emb_tag == False:
        url = 'http://192.168.102.18:8002/api/esm'

        data = {'seq': seq,'name':name, 'seq_type': 'seqvecemb'}
        for retry_count in range(max_retries):
            print(f"got protein: {name} seqvecemb count: {retry_count}")
            response = requests.post(url, json=data)

            if response.status_code == 200:
                with open(f'data/seqvecemb/{name}.npy', 'wb') as f:
                    f.write(response.content)
                data=numpy.load(f"data/seqvecemb/{name}.npy")
                return data
            return None
    else:
        data = numpy.load(f"data/seqvecemb/{name}.npy")
        return data
def get_protemb(seq,name):
    max_retries = 3
    emb_tag = os.path.exists(f"data/protemb/{name}.npy")
    if emb_tag == False:
        url = 'http://192.168.102.18:8002/api/esm'

        data = {'seq': seq,'name':name, 'seq_type': 'protemb'}
        for retry_count in range(max_retries):
            print(f"got protein: {name} protemb count: {retry_count}")
            response = requests.post(url, json=data)
            if response.status_code == 200:
                with open(f'data/protemb/{name}.npy', 'wb') as f:
                    f.write(response.content)
                data=numpy.load(f"data/protemb/{name}.npy")
                return data
            return None
    else:
        data = numpy.load(f"data/protemb/{name}.npy")
        return data
if __name__ == '__main__':
    get_emb('ASDASD')