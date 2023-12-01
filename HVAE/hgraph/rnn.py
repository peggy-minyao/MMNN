import torch
import torch.nn as nn
from hgraph.nnutils import *

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def get_init_state(self, fmess, init_state=None):
        h = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        return h if init_state is None else torch.cat( (h, init_state), dim=0)

    def get_hidden_state(self, h):
        return h

    def GRU(self, x, h_nei):
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x,sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)
        
        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x,sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess, bgraph):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0 #first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h

    def sparse_forward(self, h, fmess, submess, bgraph):
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_i = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() )
         #记忆门，先由一个线性网络处理，再由sigmoid网络层处理，input_size是Tn时刻输入数据大小，hideen_size是Tn-1时刻的隐藏数据的大小
        self.W_o = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() ) #输出门
        self.W_f = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() ) #遗忘门
        self.W = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Tanh() ) #这一层是候选记忆单元

    def get_init_state(self, fmess, init_state=None): #设置初始状态值，即全0，主要是设置指定大小的0
        h = torch.zeros(len(fmess), self.hidden_size, device=fmess.device) #是一个tensor
        c = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        if init_state is not None:
            h = torch.cat( (h, init_state), dim=0)
            c = torch.cat( (c, torch.zeros_like(init_state)), dim=0)
        return h,c

    def get_hidden_state(self, h): #取初始化状态h的第一个数 
        return h[0]

    def LSTM(self, x, h_nei, c_nei): #LSTM的计算方式
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i( torch.cat([x, h_sum_nei], dim=-1) )
        o = self.W_o( torch.cat([x, h_sum_nei], dim=-1) )
        f = self.W_f( torch.cat([x_expand, h_nei], dim=-1) )
        u = self.W( torch.cat([x, h_sum_nei], dim=-1) )
        c = i * u + (f * c_nei).sum(dim=1) #记忆门乘输入的信息，加上遗忘门乘上一次时间的输出信息
        h = o * torch.tanh(c) #输出门乘tanh网络层
        return h, c

    def forward(self, fmess, bgraph):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device) #?这是什么？
        mask[0, 0] = 0 #first message is padding

        for i in range(self.depth): #进行多次的信息的更迭
            h_nei = index_select_ND(h, 0, bgraph) 
            c_nei = index_select_ND(c, 0, bgraph)
            h,c = self.LSTM(fmess, h_nei, c_nei)
            h = h * mask
            c = c * mask
        return h,c

    def sparse_forward(self, h, fmess, submess, bgraph):
        h,c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h,c



