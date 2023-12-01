import rdkit
import rdkit.Chem as Chem
import copy
import torch

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list] #copy
        self.vmap = {x:i for i,x in enumerate(self.vocab)} #对vocab内的内容进行编码成为字典
        
    def __getitem__(self, smiles):
        return self.vmap[smiles] 

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

class PairVocab(object): #object是继承了更高级的类的一些属性，在python3中其实可以不用写。

    def __init__(self, smiles_pairs, cuda=True): #在train_generator.py里是直接输入vocab.txt
        cls = list(zip(*smiles_pairs))[0] #zip(*)为zip()的逆解压过程。 smiles_pairs有两列，取smiles_pairs的第一列
        self.hvocab = sorted( list(set(cls)) ) #对cls中的片段求交集并排序
        self.hmap = {x:i for i,x in enumerate(self.hvocab)} #基于上面排序顺序，为片段编号，得到‘片段’：编号形式的list

        self.vocab = [tuple(x) for x in smiles_pairs] #copy ； 将x转换为元组
        self.inter_size = [count_inters(x[1]) for x in self.vocab]# 统计第二列中的被标记的原子个数，被标记的原子应该是和其他片段有连接的原子。
        self.vmap = {x:i for i,x in enumerate(self.vocab)} #对所有的片段对排序，得到('B', 'B'): 0, ('B1C=CCCO1', 'C1=C[BH:1]OCC1'): 1,('B1CCC=NN1', 'B1CCC=N[NH:1]1'): 2,

        self.mask = torch.zeros(len(self.hvocab), len(self.vocab))#生成一个tensor为0的矩阵，行等于片段种类的数量，列等于片段的数量
        for h,s in smiles_pairs:
            hid = self.hmap[h] #将一列片段转换为片段种类的对应的index
            idx = self.vmap[(h,s)] # 这里是转换为所有片段的对应的index
            self.mask[hid, idx] = 1000.0 #在上面生成的行是片段种类，列是片段数量的矩阵类标出每个片段属于哪一类，并打上1000

        if cuda: self.mask = self.mask.cuda()
        self.mask = self.mask - 1000.0 #？为啥要都减0
            
    def __getitem__(self, x):
        assert type(x) is tuple
        return self.hmap[x[0]], self.vmap[x] #

    def get_smiles(self, idx):
        return self.hvocab[idx]

    def get_ismiles(self, idx):
        return self.vocab[idx][1] 

    def size(self):
        return len(self.hvocab), len(self.vocab)

    def get_mask(self, cls_idx):
        return self.mask.index_select(index=cls_idx, dim=0)

    def get_inter_size(self, icls_idx):
        return self.inter_size[icls_idx]

COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
common_atom_vocab = Vocab(COMMON_ATOMS) #返回一个类

def count_inters(s): #统计一个smiles内被编码的原子有多少个
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))


