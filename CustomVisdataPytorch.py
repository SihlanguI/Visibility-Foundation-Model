# %%
import katdal
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import vis_access
import vis_transform 

# %%

path= "https://archive-gw-1.kat.ac.za/1696230173/1696230173_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNjk2NTE0MDEyLCJwcmVmaXgiOlsiMTY5NjIzMDE3MyJdLCJleHAiOjE2OTcxMTg4MTIsInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.6CI-3QZ8qj47vSz-W3rbJ3Ga3E2U3mMyU2XNbOwsJXpUGKc_RHQ497iWUyuqNmJRp0-_CTyaN9aHbtxA7rar-A"

def read_rdb(path):
    data = katdal.open(path)
    return data

data = vis_access.read_rdb(path)
vis_chunk,vis_flag = vis_access.vis_per_scan(data)

v_tensor=torch.from_numpy(vis_chunk)

from torch.utils.data import Dataset, DataLoader

class VisData(Dataset):
    def __init__(self, v_tensor):
        self.v_tensors=v_tensor
    def __len__(self):
        return tf.size(self.v_tensor)
    def __getitem__(self, index):
        return self.v_tensor[index]

#print(v_tensor)  

def CustomData(data):
    customdata=VisData(v_tensor)
    dataloader=DataLoader(customdata,batch_size=10, shuffle=False)
    print(dataloader)
    return dataloader

dataloader=CustomData(v_tensor)

# %%



