# Model Imports
import math
import torch
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.nn import functional as F 

#Training loop Imports
import katdal
import vis_access
import corrprods

#Define Hyperparameters Required

block_size = 4  # B Essentially the context length, in this case predicts 5th token, will change the paramter later
batch_size = 8  # T
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cpu')
eval_iters = 200
n_layer = 4 # number of layers for the deep NN
p = 0.1
d_model = 8 #C
n_head = 4
head_size = 8 #Dimension of d_model, still unsure about this, however if doesnt match d_model, results in issues with the linear layers multiplcation.


torch.manual_seed(1337) #Torch seed for randomising inputs, somehow works to prevent exploding loss 

class LayerNorm(nn.Module):
    """
    LayerNorm Class:

    * Calculates the mean and var independantly for the batch of inputs.
    * Calculates new values based on the mean and var (Standardized)
    * Epsilon for numerical stability

    """
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multipled to the input #Makes the parameter learnable
        self.bias = nn.Parameter(torch.ones(1))  # Addition parameter to the input

        

    def forward(self,Input):

        """
        The Layer Norm Forward uses the mean standardising formular to normalise the input batches. 
        Option can also be to use F.LayerNorm from pytorch.
        """

        mean = Input.mean(dim = -1, keepdim =True)
        std = Input.std(dim = -1, keepdim = True)
        x_nu = self.alpha *(Input-mean)/(std+self.eps)+self.bias

        return x_nu


class MaskedAttention(nn.Module):
    def __init__(self, d_model, head_size):
        super(MaskedAttention, self).__init__()

        """
        Causal/Masked Attention Class:

        Map Q.K,V of input size data_size to head_size, this is done to control 
        the dimensionality of the Q,K,V input projects.

        """
        self.Q = nn.Linear(d_model, head_size, bias=False)
        self.K = nn.Linear(d_model, head_size, bias=False)
        self.V = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(batch_size, batch_size)))  #Shape (T, T)
        self.trill = torch.tril(torch.ones((T,T))) #The above doesnt seem to work well replace with this implemnetaion of the triangular matrix torch.ones

    def forward(self, torch_tensor):
        """
        Apply the linear layers self.K &self.Q to compute the respective key and query tensors.

        """
        
        key = self.K(torch_tensor.float())    #Shape (B,T,C)
        query = self.Q(torch_tensor.float())  #Shape (B,T,C)
        value = self.V(torch_tensor.float())
        trill = self.trill

        """
        Compute the Attention Scores:
        * Dot product of the query tensor with the transpose of the key tensor.
        * Mask applied to the scaled weights.
        * Normalise the scaled weights by applying softmax.
        * Matrix Multiply the populated weight (average*scaled*normalised) by the value tensor 

        Returns: Scaled Attention 
        
        """

        wei = query @ key.transpose(-2,-1)*C**-0.5         #Shapes (B, T, C) @ (B, C, T) ----> (B, T, T)  #Normalised using scaled attention
        wei = wei.masked_fill(trill == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        Output = wei @ value                              #Shape (B, T, headsize)

        return Output



class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout:float=0.1,max_len: int=5000 ):

        """
        Parameters:
        -----------

        d_model: Dimension of the model
        dropout: Facilitates randomly zeroing some inputs of the elements,  this is done to help with the model co-adapting/relying on each other
        max_len: Default max length for a transformer.
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
     
        pe = torch.zeros(max_len, d_model)    
        k = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
   
        pe[:, 0::2] = torch.sin(k * div_term)    
  
        pe[:, 1::2] = torch.cos(k * div_term)  
    
        pe = pe.unsqueeze(0) 

        self.register_buffer('pe', pe) #The position will be saved in a register (not be updated with the model)

    def forward(self, x):  
        x = x+ self.pe[:, :x.size(1)].requires_grad_(False) # Adds the X inputtesnor to the positions, the requires_grad makes sure that the positions are not learnt (ie the positions will not be updated with weights/bias)
        Output = self.dropout(x)


        return Output
    


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model:  int, dropout: float=0.1): #removes d_ff i dont think its useful, also complicated my dimensions, rather remove until i figure out why its needed, or if its needed
        super().__init__()
        """
        FeedForward Class:
        * Helps the model learn complex information.
        * Initialise weights and bias for 2 linear layers. 
        

        """

        self.L1 = nn.Linear(d_model, d_model)      
        self.L2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)  

    def forward(self, input_tensor):
        """
        Parameters:
        ----------
        x: Output of the attention layer.
        (Batch, Block, d_model) ----> (Batch, Block, d_ff) ---> (Batch, Block, d_model)

        Returns:
        -------
        The forward method returns the conversion of the batched inputs.
        """
        self.FFnet = self.dropout(torch.relu(self.L2(self.L1(input_tensor))))

        return self.FFnet
    



class Block(nn.Module):
    def __init__(self, d_model, n_head):

        torch_tensor = torch.zeros(block_size, batch_size, 8)
        data_size = len(torch_tensor)
        d_model = torch_tensor.size(0) # shape of B
        super().__init__()
        """
        The purpose of the block layer is to apply the layer normalization, masked attention, and feedforward network. 

        Returns:
        -------
        This layer outputs the addition of the original input tensor and output of the attention layer.

        Need to add a residual component to either this block or the self Attention block.

        """
        self.ln1 = LayerNorm(d_model)
        head_size = d_model//n_head
        self.MaskedAttention = MaskedAttention(d_model, head_size)
        self.ffd = PositionwiseFFN(d_model,p)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x ):
        attention_output = self.MaskedAttention(self.ln1(x))
        x_attention = x + attention_output
        x_feedforward = self.ffd(self.ln2(x_attention))
        x = x_attention + x_feedforward

        return x



class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_layer):
        super().__init__()

        torch_tensor = torch.zeros(block_size, batch_size, 8)
        d_model = torch_tensor.size(0) # shape of B

        """
        No embeddings.
        * Initialise position, block, prediction instances. 
        * Position instance of the PositionalEncoder class will add positional information to the input tensor.
        * Block - Sequence of transformer blocks, the number of blocks are determined by n_layer. 
        * Prediction instance is responsible for predicting the next value.Contains a linear layer to produce final model prediction.
        """
        self.position = PositionalEncoder(d_model)
        self.block = nn.Sequential(*[Block(C, n_head) for _ in range(n_layer)])
        self.prediction = nn.Linear(d_model,1 ) #set to one, since we predicting the next value


    def forward(self, torch_tensor, targets=None):

        """
        The input tensor has been passed through the hidden states of the transfomer blocK
        * x is the tensor that was passed through the block, the prediction instance will project the learnt representations to the oupu dimension.

        """

        PositionToken = self.position(torch_tensor)
        x = torch_tensor+PositionToken
        x=self.blocks(x)
        output = self.prediction(x[:,-1,:])  #simlar to andrej logits = logits[:,-1,:]
        targets =targets[:,-1,-1:]           #Trying to change the targets to match the output 4,1
         
        loss = torch.nn.functional.mse_loss(output, targets)


        return output


#DataLoader + Training Loop

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
    data_size = len(data_test)
    return data_test, data_size

data_test = read_rdb(path)
data_size = read_rdb(path)

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
 


def get_xtensorBTC(xb):
    """
    Function Returns Input as tensor with shape B,T,C. Necessary for masked attention level.
    """
    xb_BTC = tf.expand_dims(xb, axis=2)
    xb_BTC_new = tf.tile(xb_BTC , [1, 1, 8]) 
    numpy_array = xb_BTC_new.numpy()
    xtorch_tensor = torch.tensor(numpy_array, dtype=torch.int64)
    B,T,C = xtorch_tensor.shape
    return xtorch_tensor, B,T,C  

def get_ytensorBTC(yb):
    """
    Function Returns Target as tensor with shape B,T,C. Necessary for loss, has targets and inupt tensors should have same shape.
    """
    yb_BTC = tf.expand_dims(yb, axis=2)
    yb_BTC_new = tf.tile(yb_BTC , [1, 1, 8]) 
    numpy_array = yb_BTC_new.numpy()
    ytorch_tensor = torch.tensor(numpy_array, dtype=torch.int64)
    B,T,C = ytorch_tensor.shape
    return ytorch_tensor, B,T,C

"""
Below I am using the above functions to get the B,T,C shape Input and Target tensors for the model parameter.
"""
xtorch_tensor, B, T,C = get_xtensorBTC(xb)
ytorch_tensor, B, T,C = get_ytensorBTC(yb)
xtorch_tensor = xtorch_tensor.float()
ytorch_tensor = ytorch_tensor.float()


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            prob, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return prob
    


model = TransformerDecoder()
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #takes the gradients and updates the parameters
 


for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()

   
    # Evaluate the loss
    prob, loss = model(xb, yb) 
    #print("Loss:", loss)

    loss.requires_grad
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        

    











        
        
    

    











