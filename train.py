
#Training Loop
import numpy as np
import torch.nn as nn
import katdal
import torch
from corrprods import get_bl_idx, get_corrprods
from model import TransformerDecoder



#Define Hyperparameters Required


batch_size = 4
block_size = 8
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
learning_rate = 1e-3
eval_iters = 300
dropout = 0.1
n_layer = 6 # number of layers for the deep NN
C=8


#DataLoader 

"""
Connecting directly to the Archive.
"""

path="https://archive-gw-1.kat.ac.za/1701021676/1701021676_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzA1NDAwMTc3LCJwcmVmaXgiOlsiMTcwMTAyMTY3NiJdLCJleHAiOjE3MDYwMDQ5NzcsInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.5qFpWTvtMupQj1w_up5EmAvp3VHE7QUOD2CKkhmsUULerzB5vqVAvDAHgbIwPp6HwZU9qmvuAiJEBmfXuHsqWw"

def read_rdb(path):
    data = katdal.open(path)
    data.select(dumps = slice(0,50), scans ='track', pol='HH', corrprods='cross')
    data_HH = np.zeros((4096, 2016))
    bl_idx_HH = get_bl_idx( data, 64)
    data_HH[:,:] = np.zeros_like(data_HH)
    data_HH[:, bl_idx_HH] = np.abs(data.vis[2,:,:])
    bl_av_HH = np.mean((np.abs(data_HH[:, 0:2016])), axis =0)
    bl_av_HH_np =  np.reshape(bl_av_HH, -1)
    data_test =  torch.tensor(bl_av_HH_np, dtype=torch.int64)
    data_size = len(data_test)
    
    return data_test, data_size

data_test, data_size = read_rdb(path)

characters = sorted(list(set(data_test))) #sorts the data corpus
#print(characters)
#print(len(characters))

character_masked = [char if char>=0 else 0 for char in characters]
#print(character_masked)


data = torch.tensor(character_masked, dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
#print(data)
#print(data.dtype)

def get_batch(split):

    data =  train_points if split == 'train' else test_points  
                                                                           
    ix = torch.randint(len(data_test)-block_size, (batch_size, )) 
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_tensorBTC(xb):
    xb_BTC = np.tile(xb.cpu()[:,:,np.newaxis], (1,1,8))
    torch_tensor = torch.tensor(xb_BTC, dtype=torch.int64).to('cuda')
    #print("Tensor Shape:",xtorch_tensor.shape)
    return torch_tensor

#Loss Estimate for Training
@torch.no_grad()
def estimate_loss():
    #out = {'train':0.0, 'val':0.0
    out ={}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            xtorch_tensor = get_tensorBTC(X)
            ytorch_tensor = get_tensorBTC(Y)

            prob, loss = model(xtorch_tensor, ytorch_tensor.float())
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
    xtorch_tensor = get_tensorBTC(X)
    ytorch_tensor = get_tensorBTC(Y)
    prob, loss = model(xtorch_tensor, ytorch_tensor.float())

    loss.backward()
    #clip_grad_norm_(model.parameters(), max_norm=1.0)
    #for param in model.parameters():
        #print(param.grad)


    optimizer.step()

    optimizer.zero_grad() # Back propation of the model weights

    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}")

