import numpy as np
import torch
import torch.nn as nn
import katdal
import tensorflow as tf

from vis_access import *
from model import *

# Hyperparameters
batch_size = 4
block_size = 8
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
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


def get_batch(split):
    data_test =  train_points if split == 'train' else test_points         
    ix = torch.randint(len(data_test)-block_size, (batch_size, )) 
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    
def get_tensorBTC(xb):
    xb_BTC = tf.expand_dims(xb, axis=2)
    xb_BTC_new = tf.tile(xb_BTC , [1, 1, 8]) 
    numpy_array = xb_BTC_new.numpy()
    torch_tensor = torch.tensor(numpy_array, dtype=torch.int64)
    B,T,C = torch_tensor.shape
    return torch_tensor, B,T,C

    bl_idx_HH = get_bl_idx( data, 64)
    data_HH[:,:] = np.nan

    data_HH[:, bl_idx_HH] = np.abs(data.vis[2,:,:])
    bl_av_HH = np.mean((np.abs(data_HH[:, 0:2016])), axis =0)
    bl_av_HH_np =  np.reshape(bl_av_HH, -1)
    data_test =  torch.tensor(bl_av_HH_np)
    data_size = len(data_test)
    return data_test, data_size
  
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


path="https://archive-gw-1.kat.ac.za/1696230173/1696230173_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNjk2NTE0MDEyLCJwcmVmaXgiOlsiMTY5NjIzMDE3MyJdLCJleHAiOjE2OTcxMTg4MTIsInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.6CI-3QZ8qj47vSz-W3rbJ3Ga3E2U3mMyU2XNbOwsJXpUGKc_RHQ497iWUyuqNmJRp0-_CTyaN9aHbtxA7rar-A"

data_test, data_size = read_rdb(path)
n = int(0.8*len(data_test)) 
train_points = data_test[:n]
test_points = data_test[n:]

xb, yb = get_batch('train')

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
