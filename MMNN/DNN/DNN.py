import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils_c import *

# GCN-CNN based model
class DNN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_m=1690, output_dim=128, dropout=0.3,feature = 1024):

        super(DNN, self).__init__()

        self.n_output = n_output
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        #DNN
        self.batch_norm1 = nn.BatchNorm1d(num_features_m)
        self.linear1=nn.Linear(num_features_m,feature)
        self.batch_norm2 = nn.BatchNorm1d(feature)
        self.linear2=nn.Linear(feature,feature//2)
        self.batch_norm3 = nn.BatchNorm1d(feature//2)
        self.linear3=nn.Linear(feature+feature//2,feature//2)
        self.batch_norm4 = nn.BatchNorm1d(feature//2)
       # self.linear4=nn.Linear(3500,output_dim)
        self.linear5=nn.Linear(feature//2,feature//4)
        self.batch_norm5 = nn.BatchNorm1d(feature//4)
        self.linear6=nn.Linear(feature*2 +feature//4,feature)
        self.linear7=nn.Linear(feature,feature//2)
        self.out = nn.Linear(feature//2, self.n_output)

    def forward(self, data):
        x, edge_index, batch, edge_attr,edge_weight,descriptor = data.x, data.edge_index, data.batch,data.edge_attr,data.edge_etype,data.x_mlp
        #DNN
        descriptor1 = self.batch_norm1(descriptor)
        xf1 = self.tanh(descriptor1)
        xf1 = self.linear1(xf1)
        xf1 = self.batch_norm2(xf1)
        xf1 = self.tanh(xf1)
        xf2 = self.linear2(xf1) 
      #  xf2 = self.dropout(xf2)
        xf2 = self.batch_norm3(xf2)
        xf2 = self.tanh(xf2)
        xf2 = torch.cat([xf2,xf1],dim = 1)
        xf3 = self.linear3(xf2) 
        xf3 = self.batch_norm4(xf3)
        xf3 = self.tanh(xf3)
        #xf3 = torch.cat([xf2,xf3],dim = 1)
        xf4 = self.linear5(xf3)
       #xf3 = self.batch_norm5(xf3)
        xf4 = self.dropout(xf4)
        xf3 = torch.cat([xf2,xf3],dim = 1)
        xf4 = torch.cat([xf4,xf3],dim = 1)
        xf4 = self.linear6(xf4)
        xf4 = self.linear7(xf4)
        out = torch.sigmoid(self.out(xf4))
        return out,out
