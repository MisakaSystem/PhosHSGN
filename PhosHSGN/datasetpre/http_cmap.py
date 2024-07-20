import os

import numpy
import requests
import random
import string
def get_cmap(seq,name):
    max_retries = 3
    cmap_tag = os.path.exists(f"data/cmap/{name}.npy")
    if cmap_tag == False:
        url = 'http://192.168.102.18:8002/api/esm'

        data = {'seq': seq,'name':name, 'seq_type': 'cmap'}
        for retry_count in range(max_retries):
            print(f"got protein: {name} cmap count: {retry_count}")
            response = requests.post(url, json=data)

            if response.status_code == 200:
                try:
                    random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
                    with open(f'data/cmap/{name}.npy', 'wb') as f:
                        f.write(response.content)
                    data=numpy.load(f"data/cmap/{name}.npy")
                    return data
                except ValueError as e:
                    return None
        return None
    else:
        data = numpy.load(f"data/cmap/{name}.npy")
        return data
if __name__ == '__main__':
    get_cmap('ASDASD')