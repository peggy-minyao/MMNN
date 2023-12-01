import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils_c import *
# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=37,edge_dim = 20, output_dim=128,heads = 12,features = 1500,dropout = 0.7):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, edge_dim = edge_dim, heads=heads)
        self.conv2 = GCNConv(num_features_xd*heads, num_features_xd*heads)
        self.fc_g1 = torch.nn.Linear(num_features_xd*heads*2,features)
        self.fc_g2 = torch.nn.Linear(features, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(output_dim, features//2)
        self.out = nn.Linear(features//2, self.n_output)

    def forward(self, data):
        x, edge_index, batch, edge_attr,edge_weight = data.x, data.edge_index, data.batch,data.edge_attr,data.edge_etype
      #  print(x.shape)
        x,w = self.conv1(x, edge_index,edge_attr = edge_attr ,return_attention_weights= True)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight = edge_weight)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))

        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        out = torch.sigmoid(self.out(x))
        return out,w
