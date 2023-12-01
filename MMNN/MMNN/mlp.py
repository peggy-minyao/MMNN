import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils_c import *

class MLP(torch.nn.Module):
    def __init__(self,n_output = 1, dropout =0.3):
        super(MLP,self).__init__()
        self.linear = nn.Linear(3,6)
        self.output = nn.Linear(6,n_output)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self,data):
        out = self.linear(data)
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out,out