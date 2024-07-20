import os

import requests
import random
import string


def get_dssp(seq,name):
    max_retries = 3
    dssp_tag = os.path.exists(f"data/dssp/{name}.dssp")
    if dssp_tag == False:
        url = 'http://192.168.102.18:8002/api/esm'
        data = {'seq': seq,'name':name, 'seq_type': 'dssp'}
        for retry_count in range(max_retries):
            print(f"got protein: {name} dssp count: {retry_count}")
            response = requests.post(url, json=data)
            if response.status_code == 200:
                random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
                with open(f'data/dssp/{name}.dssp', 'wb') as f:
                    f.write(response.content)
                current_directory = os.getcwd()
                absolute_path = os.path.join(current_directory, f'data/dssp/{name}.dssp')
                return absolute_path
            return None
    else:
        current_directory = os.getcwd()
        absolute_path = os.path.join(current_directory, f'data/dssp/{name}.dssp')
        return absolute_path