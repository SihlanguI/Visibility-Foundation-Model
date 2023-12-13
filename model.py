import numpy as np
import math
import torch
import torch.nn as nn
import katdal
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import vis_access
import corrprods

#Define Hyperparameters Required

batch_size = 4
block_size = 8 
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cpu')
eval_iters = 200
dropout = 0.1


#DataLoader 
"""
Connecting directly to the Archive.
"""

path="https://archive-gw-1.kat.ac.za/1701021676/1701021676_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzAyNDczOTY2LCJwcmVmaXgiOlsiMTcwMTAyMTY3NiJdLCJleHAiOjE3MDMwNzg3NjYsInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.CLCieZa-9IRgV4HrD8O37I9RwxvsavQu_KljILjP2uTUEiB9ePwVUia4Td4RZ_7c7xz-HvikgfCkxpo7LvyCbQ"

def read_rdb(path):
    data = katdal.open(path)
    data.select(dumps = slice(0,10), scans ='track', pol='HH', corrprods='cross')
    data_HH = np.zeros((4096, 2016))
    bl_idx_HH = corrprods.get_bl_idx( data, 64)
    data_HH[:,:] = np.nan
    data_HH[:, bl_idx_HH] = data.vis[2,:,:]
    bl_av_HH = np.mean((np.abs(data_HH[:, 0:2016])), axis =0)
    bl_av_HH_np =  np.reshape(bl_av_HH, -1)
    data_test =  torch.tensor(np.abs(bl_av_HH_np), dtype=torch.long)
    return data_test

data_test = read_rdb(path)

n = int(0.8*len(data_test)) 
train_points = data_test[:n]
test_points = data_test[n:]

def get_batch(split):

    data_test =  train_points if split == 'train' else test_points  
                                                                           
    ix = torch.randint(len(data_test)-block_size, (batch_size, )) 
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch('train')

def get_tensorBTC(xb):
    xb_BTC = tf.expand_dims(xb, axis=2)
    xb_BTC_new = tf.tile(xb_BTC , [1, 1, 8]) 
    numpy_array = xb_BTC_new.numpy()
    torch_tensor = torch.tensor(numpy_array, dtype=torch.int64)
    return torch_tensor



