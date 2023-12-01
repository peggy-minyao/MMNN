import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils_c import *

# GCN-CNN based model
class GRU(torch.nn.Module):
    def __init__(self, n_output=1, num_features_seq= 35, embed_dim=300,  bidirectional =True, hidden_size = 300 ,output_dim=128, dropout=0,n_layers =2, batch_size = 512):

        super(GRU, self).__init__()

        self.n_output = n_output
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.n_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(num_features_seq+1, embed_dim)
        self.gru =  torch.nn.GRU(hidden_size,hidden_size,n_layers, bidirectional = bidirectional )
        self.fc1 = torch.nn.Linear(hidden_size*self.n_layers, 516)
        self.fc2 = torch.nn.Linear(516,128)
        self.fc3 = torch.nn.Linear(128, self.n_output)
    def _init_hidden(self, batch_size):
        device = torch.device('cpu')
        hidden = torch.zeros(self.n_directions*self.n_layers, batch_size, self.hidden_size)
        return hidden.to(device)

    def forward(self, data):
        x, edge_index, batch, edge_attr,edge_weight,descriptor, seq= data.x, data.edge_index, data.batch,data.edge_attr,data.edge_etype,data.x_mlp, data.x_seq
        seq = seq.t()
        hidden = self._init_hidden(self.batch_size)
        input_seq = self.embedding(seq)
        #print(input_seq.shape)
        output, hidden = self.gru(input_seq, hidden)
        hidden_cat = torch.cat([hidden[-1],hidden[-2]], dim = 1)
        #hidden_cat = torch.cat([hidden_cat,hidden[-3]], dim = 1)
        xl = self.fc1(hidden_cat)
        xl = self.fc2(xl)
        xl = self.fc3(xl)
        xl  = self.dropout(xl)
        out = torch.sigmoid(xl)
        return out,out
