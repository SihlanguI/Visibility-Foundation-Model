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
from torch.nn import functional as F 

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

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(data_size)
        self.parameters = list(model.parameters())
        self.weight = self.parameters[0]
        self.bias = self.parameters[1]

    def forward(self,Input):
        return  F.layer_norm(Input, self.weight,self.weight.shape, self.bias, 1e-5)  


class MaskedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        




