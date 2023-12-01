import sys
import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set() #创建空集合
    for line in data:
        s = line.strip("\r\n ") #删掉分行符
        hmol = MolGraph(s) #hmol为一个类
        for node,attr in hmol.mol_tree.nodes(data=True): # hmol.mol_tree返回一个基于片段节点表征的分子，node是原子编码，attr应该是性质
            smiles = attr['smiles']
            vocab.add( attr['label'] ) #向集合中加入元素
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]]  #sys.stdin是一个标准输入的方法，把输入的信息读取进来，在这里读取的是all.txt
    data = list(set(data))#这一步求交集，即删掉重复数据

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)] #把数据一个batch一个batch的分好后，放在batches里。

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches) #将batches中的数据用process函数处理
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab] #读取vocab_list里的vocab里的(x,y)
    vocab = list(set(vocab)) #求交集

    for x,y in sorted(vocab):
        print(x, y)
