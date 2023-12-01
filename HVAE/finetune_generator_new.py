import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

import hgraph
from hgraph import HierVAE, common_atom_vocab, PairVocab
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()])) #一个函数,对m内的参数做一个处理
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

'''
class Chemprop(object):
    
    def __init__(self, checkpoint_dir):
        self.features_generator = ['rdkit_2d_normalized']
        self.checkpoints, self.scalers, self.features_scalers = [], [], []
        for root, _, files in os.walk(checkpoint_dir): #walk用于通过在目录中游走而输出目录中的文件名，root是当前正在遍历的文件夹本身的名字，files是一个list，代表文件夹中文件的名字
            for fname in files: 
                if fname.endswith('.pt'): #pt文件是checkpoint文件
                    fname = os.path.join(root, fname) #则fname返回该checkpoint文件的路径
                    scaler, features_scaler = load_scalers(fname) #chemprop内的函数，用于加载训练模型时的数据的标量和特征的标量？？好像是chemprop内定义的参数。
                    self.scalers.append(scaler)
                    self.features_scalers.append(features_scaler)
                    model = load_checkpoint(fname) #加载模型的checkpoint
                    self.checkpoints.append(model)

    def predict(self, smiles, batch_size=500):
        test_data = get_data_from_smiles(
            smiles=[[s] for s in smiles],
            skip_invalid_smiles=False,
            features_generator=self.features_generator
        ) # get_data_from_smiles也是chemprop内的函数，将smile转换为mol格式。
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol[0] is not None]#筛选非none的分子的index
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])# MoleculeDataset是一个类。
        test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=batch_size)

        sum_preds = np.zeros((len(test_data), 1))#行为test_data长度，宽为1的0。
        for model, scaler, features_scaler in zip(self.checkpoints, self.scalers, self.features_scalers): #对不同的checkpoints都进行了一个预测，然后取平均值
            test_data.reset_features_and_targets() #Resets the features (atom, bond, and molecule) and targets to their raw values.
            if features_scaler is not None:
                test_data.normalize_features(features_scaler) #标准化测试集的数据

            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            sum_preds += np.array(model_preds)

        # Ensemble predictions
        avg_preds = sum_preds / len(self.checkpoints)
        avg_preds = avg_preds.squeeze(-1).tolist()

        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]

        return np.array(full_preds, dtype=np.float32) #返回每个分子的预测值，分子为none的地方返回0
'''

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--generative_model', required=True)
    #parser.add_argument('--chemprop_model', required=True) #输入一个路径，里面存放了一个模型
    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--inner_epoch', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--min_similarity', type=float, default=0.1)
    parser.add_argument('--max_similarity', type=float, default=0.5)
    parser.add_argument('--nsample', type=int, default=10000)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(args.train) as f:
        train_smiles = [line.strip("\r\n ") for line in f] #args.train：一个路径，路径内包含确定有活性的分子

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] #args.vocab：一个路径，包含语法库
    args.vocab = PairVocab(vocab)

    score_func = Chemprop(args.chemprop_model)#args.chemprop_model：包含路径，路径内有模型的checkpoint
    good_smiles = train_smiles
    train_mol = [Chem.MolFromSmiles(s) for s in train_smiles]#转换为mol
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in train_mol]#转换为fps

    model = HierVAE(args).cuda() #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Loading from checkpoint ' + args.generative_model) 
    model_state, optimizer_state, _, beta = torch.load(args.generative_model)#导入待优化的模型的checkpoint
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    for epoch in range(args.epoch): #epoch = 10
        good_smiles = sorted(set(good_smiles))
        random.shuffle(good_smiles)
        dataset = hgraph.MoleculeDataset(good_smiles, args.vocab, args.atom_vocab, args.batch_size)

        print(f'Epoch {epoch} training...')
        for _ in range(args.inner_epoch): #args.inner_epoch = 10
            meters = np.zeros(6)
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x:x[0], shuffle=True, num_workers=16)
            for batch in tqdm(dataloader):
                model.zero_grad()
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta) #这些返回的都是啥？
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])

            meters /= len(dataset)
            print("Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))

        ckpt = (model.state_dict(), optimizer.state_dict(), epoch, beta)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{epoch}")) #保存优化好的模型

        print(f'Epoch {epoch} decoding...')
        decoded_smiles = []
        with torch.no_grad():
            for _ in tqdm(range(args.nsample // args.batch_size)): #args.nsample=1000
                outputs = model.sample(args.batch_size, greedy=True) #decoder解码分子
                decoded_smiles.extend(outputs)

    print(decoded_smiles)
'''
        print(f'Epoch {epoch} filtering...')
        scores = score_func.predict(decoded_smiles) #预测decoded_smiles的分数
        outputs = [(s,p) for s,p in zip(decoded_smiles, scores) if p >= args.threshold] #args.threshold =0.5，如果生成的分子的activity_score大于0.5则认为是有活性的分子
        print(f'Discovered {len(outputs)} active molecules')

        novel_entries = []
        good_entries = []
        for s, p in outputs:
            mol = Chem.MolFromSmiles(s)
            fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps, train_fps)) #计算用来学习的分子和生成的分子之间的相似性
            good_entries.append((s, p, sims.max())) #挑选出相似性最大的值，放入good_entries内
            if args.min_similarity <= sims.max() <= args.max_similarity:#如果该最大值在0.1和0.5之间
                novel_entries.append((s, p, sims.max())) #则认为该生成的分子是novel分子
                good_smiles.append(s)

        print(f'Discovered {len(novel_entries)} novel active molecules')
        with open(os.path.join(args.save_dir, f"new_molecules.{epoch}"), 'w') as f:
            for s, p, sim in novel_entries:
                print(s, p, sim, file=f)

        with open(os.path.join(args.save_dir, f"good_molecules.{epoch}"), 'w') as f:
            for s, p, sim in good_entries:
                print(s, p, sim, file=f)'''
