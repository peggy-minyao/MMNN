import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.nnutils import *
from hgraph.mol_graph import MolGraph
from hgraph.rnn import GRU, LSTM

class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout): 
    #rnn_type是指选择LSTM还说GRU;
    #input_size是指编码器输入的大小，在这里graph层，attachment层和motif层是不一样的；
    #node_fdim每个节点内特征的维度？，同样三个层是不一样的
    #hidden_size即隐藏层大小，三个层都是一样的
    #depth即三个层有几个LSTM神经元，attachment层和motif层是一样的
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential( 
                nn.Linear(node_fdim + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        if rnn_type == 'GRU':
            self.rnn = GRU(input_size, hidden_size, depth) 
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth) 
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph): #这四个值是embeding后的
        h = self.rnn(fmess, bgraph) #bgraph应该是bond相关的信息
        h = self.rnn.get_hidden_state(h) 
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
        mask[0, 0] = 0 #first node is padding
        return node_hiddens * mask, h #return only the hidden state (different from IncMPNEncoder in LSTM case)

class HierMPNEncoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout): 
        super( , self).__init__()
        self.vocab = vocab #这里的vocab是args.vocab，需要读取vocab.txt文件内的内容
        self.hidden_size = hidden_size #default=250
        self.dropout = dropout #default=0
        self.atom_size = atom_size = avocab.size() 
        #avocab = common_atom_vocab= COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
        #avocab是一个类，其中.size()是函数之一，等于38，总共有12种原子类型，具体信息在vocab.py里
        #不知道为什么是38，明明只有12种原子类型
        #embed_size =250
        self.bond_size = bond_size = len(MolGraph.BOND_LIST) + MolGraph.MAX_POS # 24，其中前面有4类键的类型，后面的MAX_POS=20，why？

        self.E_c = nn.Sequential(
                nn.Embedding(vocab.size()[0], embed_size),
                nn.Dropout(dropout)
        )
        self.E_i = nn.Sequential(
                nn.Embedding(vocab.size()[1], embed_size),
                nn.Dropout(dropout)
        )
        self.W_c = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )
        self.W_i = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        self.E_a = torch.eye(atom_size).cuda() # 38*38大小的矩阵，对角线为1，torch.eye(n，m=None，out=None) 生成n行m列的对角线全1，其余部分全0的二维数组
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cuda() #4*4大小的矩阵，对角线为1，len(MolGraph.BOND_LIST) = 3 ，单键、双键、三键、芳香键
        self.E_apos = torch.eye( MolGraph.MAX_POS ).cuda() #MAX_POS =20
        self.E_pos = torch.eye( MolGraph.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh() #root activation is tanh
        )
        self.tree_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    def embed_inter(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_i(fnode[:, 1])

        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput, hnode], dim=-1) )

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return hnode, hmess, agraph, bgraph

    def embed_tree(self, tree_tensors, hinter):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_c(fnode[:, 0])
        hnode = self.W_c( torch.cat([finput, hinter], dim=-1) )

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return hnode, hmess, agraph, bgraph
    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = self.E_a.index_select(index=fnode, dim=0) #one-hot模式对每个原子的类型编码，embeding过程。按照fonde的index，取E_a中的数据，dim=0代表按行取。
        #假设该分子的第一个原子是5，那么就取出E_a中的第五行，第五行第五个数是1，其他全是0的。
        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0) #取出每个键的start-id的原子类型的one-hot编码
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0) #键的类型的one-hot编码
        fpos = self.E_apos.index_select(index=fmess[:, 3], dim=0) #键的权重的one-hot编码，E_apos是一个20*20的矩阵
        hmess = torch.cat([fmess1, fmess2, fpos], dim=-1) #按照一定的维度拼接
        return hnode, hmess, agraph, bgraph

    def embed_root(self, hmess, tree_tensors, roots):
        roots = tree_tensors[2].new_tensor(roots) 
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors): #graph_tensors = fnode, fmess, agraph, bgraph, scope ，tree_tensors同graph_tensors
        tensors = self.embed_graph(graph_tensors) 
        hatom,_ = self.graph_encoder(*tensors)

        tensors = self.embed_inter(tree_tensors, hatom) #在这里hatom是上面graph_encoder的输出，？？这里为什么用tree_tensors作为输入？
        hinter,_ = self.inter_encoder(*tensors)

        tensors = self.embed_tree(tree_tensors, hinter) #hinter是上面inter_encoder的输出
        hnode,hmess = self.tree_encoder(*tensors) #*在调用函数时代表分配函数，即将tensors中的值依次输入到该函数中去
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])

        return hroot, hnode, hinter, hatom

class IncMPNEncoder(MPNEncoder):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(IncMPNEncoder, self).__init__(rnn_type, input_size, node_fdim, hidden_size, depth, dropout)

    def forward(self, tensors, h, num_nodes, subset):
        fnode, fmess, agraph, bgraph = tensors
        subnode, submess = subset

        if len(submess) > 0: 
            h = self.rnn.sparse_forward(h, fmess, submess, bgraph)

        nei_message = index_select_ND(self.rnn.get_hidden_state(h), 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
        node_hiddens = index_scatter(node_hiddens, node_buf, subnode)
        return node_hiddens, h

class IncHierMPNEncoder(HierMPNEncoder):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(IncHierMPNEncoder, self).__init__(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.tree_encoder = IncMPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = IncMPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = IncMPNEncoder(rnn_type, self.atom_size + self.bond_size, self.atom_size, hidden_size, depthG, dropout)
        del self.W_root

    def get_sub_tensor(self, tensors, subset):
        subnode, submess = subset
        fnode, fmess, agraph, bgraph = tensors[:4]
        fnode, fmess = fnode.index_select(0, subnode), fmess.index_select(0, submess)
        agraph, bgraph = agraph.index_select(0, subnode), bgraph.index_select(0, submess)

        if len(tensors) == 6:
            cgraph = tensors[4].index_select(0, subnode)
            return fnode, fmess, agraph, bgraph, cgraph, tensors[-1]
        else:
            return fnode, fmess, agraph, bgraph, tensors[-1]

    def embed_sub_tree(self, tree_tensors, hinput, subtree, is_inter_layer):
        subnode, submess = subtree
        num_nodes = tree_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(tree_tensors, subtree)

        if is_inter_layer:
            finput = self.E_i(fnode[:, 1])
            hinput = index_select_ND(hinput, 0, cgraph).sum(dim=1)
            hnode = self.W_i( torch.cat([finput, hinput], dim=-1) )
        else:
            finput = self.E_c(fnode[:, 0])
            hinput = hinput.index_select(0, subnode)
            hnode = self.W_c( torch.cat([finput, hinput], dim=-1) )

        if len(submess) == 0:
            hmess = fmess
        else:
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(hnode, node_buf, subnode)
            hmess = node_buf.index_select(index=fmess[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
            hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return hnode, hmess, agraph, bgraph 

    def forward(self, tree_tensors, inter_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph):
        num_tree_nodes = tree_tensors[0].size(0)
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1] #graph tensor is already embedded
            hgraph.node, hgraph.mess = self.graph_encoder(sub_graph_tensors, hgraph.mess, num_graph_nodes, subgraph)

        if len(subtree[0]) + len(subtree[1]) > 0:
            sub_inter_tensors = self.embed_sub_tree(inter_tensors, hgraph.node, subtree, is_inter_layer=True)
            hinter.node, hinter.mess = self.inter_encoder(sub_inter_tensors, hinter.mess, num_tree_nodes, subtree)

            sub_tree_tensors = self.embed_sub_tree(tree_tensors, hinter.node, subtree, is_inter_layer=False)
            htree.node, htree.mess = self.tree_encoder(sub_tree_tensors, htree.mess, num_tree_nodes, subtree)

        return htree, hinter, hgraph

