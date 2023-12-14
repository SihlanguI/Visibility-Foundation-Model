
import math
import torch
import torch.nn as nn
import katdal
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
#import tensorflow as tf
from torch.nn import functional as F 

#Define Hyperparameters Required


block_size = 4  # B Essentially the context length, in this case predicts 5th token, will change the paramter later
batch_size = 8  # T
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cpu')
eval_iters = 200
dropout = 0.1


class LayerNorm(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.linear = nn.Linear(data_size)
        self.parameters = list(model.parameters())
        self.weight = self.parameters[0]
        self.bias = self.parameters[1]

    def forward(self,Input):
        return  F.layer_norm(Input, self.weight,self.weight.shape, self.bias, 1e-5)  


class MaskedAttention(nn.Module):
    def __init__(self, d_model, head_size):
        super(MaskedAttention, self).__init__()

        """
        Map Q.K,V of input size data_size to head_size, this is done to control 
        the dimensionality of the Q,K,V input projects.

        """
        self.Q = nn.Linear(d_model, head_size, bias=False)
        self.K = nn.Linear(d_model, head_size, bias=False)
        self.V = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(batch_size, batch_size)))  #Shape (T, T)

    def forward(self, torch_tensor):
        """
        Apply the linear layers self.K &self.Q to compute the respective key and query tensors.

        """
        B,T,C = torch_tensor
        key = self.K(torch_tensor.float())    #Shape (B,T,C)
        query = self.Q(torch_tensor.float())  #Shape (B,T,C)
        value = self.V(torch_tensor.float())

        """
        Compute the Attention Scores:
        * Dot product of the query tensor with the transpose of the key tensor.
        * Mask applied to the scaled weights.
        * Normalise the scaled weights by applying softmax.
        * Matrix Multiply the populated weight (average*scaled*normalised) by the value tensor 

        Returns: Scaled Attention 
        
        """

        wei = query @ key.transpose(-2,-1)*C**-0.5         #Shapes (B, T, C) @ (B, C, T) ----> (B, T, T)  #Normalised using scaled attention
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        OutputAttention = wei @ value

        return OutputAttention



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

        self.register_buffer('pe', pe)

    def forward(self, OutputAttention):
        OutputAttention = OutputAttention+ self.pe[:, : OutputAttention.size(1)].requires_grad_(False) 


        return self.dropout(OutputAttention)
    


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1):
        super().__init__()
        """
        Initialise weights. Take the linear transform of the weights.

        """

        self.w1 = nn.Linear(d_model, d_ffn)      
        self.w2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, OutputAttention):
        """
        Parameters:
        ----------
        x: Output of the attention layer. 

        Returns:
        -------
        """
        self.net = self.w2(self.dropout(self.w1(OutputAttention).relu()))

        return self.net
    
torch_tensor = torch.zeros(block_size, batch_size, 8)
class Block(nn.Module):
    def __init__(self,):

        data_size = len(torch_tensor)
        d_model = torch_tensor.size(0) #B
        super().__init__()
        """
        The purpose of the block layer is to apply the layer normalization, masked attention, and feedforward network. 

        Returns:
        -------
        This layer outputs the addition of the original inputs and outputs of the attention layer.

        """
        self.ln1 = LayerNorm(data_size)
        head_size = data_size//d_model
        self.MaskedAttention = MaskedAttention(d_model, head_size)
        self.ffd = PositionwiseFFN(data_size)
        self.ln2 = LayerNorm(data_size)

    def forward(self, x ):

        x = x + self.MaskedAttention(self.ln1)
        x = x + self.ffd(self.ln2)
        
        return x



class TransformerDecoder(nn.Module):
    def __init__(self,):
        super().__init__()

        
        
    

    











