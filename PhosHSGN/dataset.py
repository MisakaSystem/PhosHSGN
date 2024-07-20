import os.path

import numpy as np
from torch.utils.data import Dataset
import torch
from  datasetpre.dataprocess import seq_load
class LuDataset(Dataset):
    def __init__(self,file_path=None):
        super(LuDataset, self).__init__()
        if file_path != None:
            # get protein sequence
            self.protein,self.seq,self.label,self.position=seq_load(file_path)

    def __getitem__(self, item):
        return self.protein[item],self.seq[item],self.label[item],self.position[item]

    def __len__(self):
        return len(self.label)
