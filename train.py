#Training Loop
import os
print(os.getcwd())
import numpy as np
import math
import torch
import torch.nn as nn
import katdal
import numpy as np
import torch
import sys
sys.path.append('/home/tkassie/.local/share/jupyter/runtime/')
import sys
sys.path.append('/home/tkassie/.local/share/jupyter/runtime/')
from corrprods import get_bl_idx, get_corrprods #saved isaac functions, for retrieving antenna-pol products for the baseline lengths orderings of the visibility data
from model import TransformerDecoder
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm



#Define Hyperparameters Required

batch_size = 4
block_size = 8
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 100
learning_rate = 0.001
eval_iters = 200
dropout = 0.1
max_iters = 3000
n_layer = 6 # number of layers for the deep NN
p = 0.1
d_model = 8
d_ff = 4*d_model #From Paper
n_head = 4
head_size =8
C=8


#DataLoader 

"""
Connecting directly to the Archive.
"""

path="https://archive-gw-1.kat.ac.za/1701021676/1701021676_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzA0NzE2MjAwLCJwcmVmaXgiOlsiMTcwMTAyMTY3NiJdLCJleHAiOjE3MDUzMjEwMDAsInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.v7zswHdPmiQpDE0DAiUe1-MpnytpEfCgyEZlG8J39oHYN1xKgD2x4UxlWN454fHHqmXS0VawQ4nX6qKqlFEyDA"
def read_rdb(path):
    data = katdal.open(path)
    data.select(dumps = slice(0,10), scans ='track', pol='HH', corrprods='cross')
    data_HH = np.zeros((4096, 2016))
    bl_idx_HH = get_bl_idx( data, 64)
    data_HH[:,:] = np.nan

    data_HH[:, bl_idx_HH] = np.abs(data.vis[2,:,:])
    bl_av_HH = np.mean((np.abs(data_HH[:, 0:2016])), axis =0)
    bl_av_HH_np =  np.reshape(bl_av_HH, -1)
    data_test =  torch.tensor(bl_av_HH_np)
    data_size = len(data_test)
    return data_test, data_size

def get_batch(split):

    data_test =  train_points if split == 'train' else test_points

    ix = torch.randint(len(data_test)-block_size, (batch_size, ))
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
def get_xtensorBTC(xb):
    xb_BTC = np.tile(xb.cpu()[:,:,np.newaxis], (1,1,8))
    xtorch_tensor = torch.tensor(xb_BTC, dtype=torch.int64).to('cuda')
    B,T,C = xtorch_tensor.shape
    #print("Tensor Shape:",xtorch_tensor.shape)
    return xtorch_tensor, B,T,C

def get_ytensorBTC(yb):
    yb_BTC = np.tile( yb.cpu()[:,:, np.newaxis], (1,1,8))
    ytorch_tensor = torch.tensor(yb_BTC, dtype=torch.int64).to('cuda')
    B,T,C = ytorch_tensor.shape
    return ytorch_tensor, B,T,C


#Loss Estimate for Training
@torch.no_grad()
def estimate_loss():
    #out = {'train':0.0, 'val':0.0
    out ={}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            xtorch_tensor, B, T,C = get_xtensorBTC(X)
            ytorch_tensor, B, T,C = get_ytensorBTC(Y)

            prob, loss = model(xtorch_tensor, ytorch_tensor)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



data_test, data_size = read_rdb(path)
#print(data_test.dtype)
n = int(0.8*len(data_test))
train_points = data_test[:n]
test_points = data_test[n:]
xb, yb = get_batch('train')

# Training Loop
model = TransformerDecoder(block_size, n_layer).to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.train()  # the model is in training mode


for iter in range(max_iters):
    X, Y = get_batch('train')
    xtorch_tensor, B, T, C = get_xtensorBTC(X)
    ytorch_tensor, B, T, C = get_ytensorBTC(Y)
    prob, loss = model(xtorch_tensor, ytorch_tensor)
    loss.backward()
    #clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

   # optimizer.zero_grad() # Back propation of the model weights

    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}")
