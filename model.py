
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
n_layer = 4 # number of layers for the deep NN
dropout = 0.1
#d_ff = 4*d_model #From Paper

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
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1):
        super().__init__()
        """
        FeedForward Class:
        * Helps the model learn complex information.
        * Initialise weights and bias for 2 linear layers. 
        

        """

        self.L1 = nn.Linear(d_model, d_ffn)      
        self.L2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, Output):
        """
        Parameters:
        ----------
        x: Output of the attention layer.
        (Batch, Block, d_model) ----> (Batch, Block, d_ff) ---> (Batch, Block, d_model)

        Returns:
        -------
        The forward method returns the conversion of the batched inputs.
        """
        self.FFnet = self.L1(self.dropout(torch.relu(self.L2(Output))))

        return self.FFnet
    


torch_tensor = torch.zeros(block_size, batch_size, 8)
class Block(nn.Module):
    def __init__(self, d_model):

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
        head_size = data_size//d_model
        self.MaskedAttention = MaskedAttention(d_model, head_size)
        self.ffd = PositionwiseFFN(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x ):

        x = x + self.MaskedAttention(self.ln1)
        x = x + self.ffd(self.ln2)
        
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, n_layer):
        super().__init__()

        d_model = torch_tensor.size(0) # shape of B

        """
        No embeddings.
        * Initialise position, block, prediction instances. 
        * Position instance of the PositionalEncoder class will add positional information to the input tensor.
        * Block - Sequence of transformer blocks, the number of blocks are determined by n_layer. 
        * Prediction instance is responsible for predicting the next value.Contains a linear layer to produce final model prediction.
        """
        self.position = PositionalEncoder(d_model)
        self.block = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.prediction = nn.Linear(d_model,1 ) #set to one, since we predicting the next value


    def forward(self, torch_tensor, targets=None):

        """
        The input tensor has been passed through the hidden states of the transfomer blocK
        * x is the tensor that was passed through the block, the prediction instance will project the learnt representations to the oupu dimension.

        """

        PositionToken = self.position(torch_tensor)
        x = torch_tensor+PositionToken
        x=self.blocks(x)
        output = self.prediction(x)







        
        
    

    











