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

path= ""
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



