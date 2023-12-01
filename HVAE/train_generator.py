import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50) #每几个step打印一次
parser.add_argument('--save_iter', type=int, default=5000)#每几个step保存一次

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
args.vocab = PairVocab(vocab) #返回一个类

model = HierVAE(args).cuda() 
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)) #x.nelement()为统计x中的元素个数

for param in model.parameters(): #初始化参数，如果参数是一维的，则赋全0，否则赋xavier_normal
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model: #决定是否从某一个checkpoint开始训练
    print('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
    total_step = beta = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()])) #匿名函数，功能是啥？
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
for epoch in range(args.epoch):
    dataset = DataFolder(args.train, args.batch_size)

    for batch in tqdm(dataset):
        total_step += 1
        model.zero_grad()
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, loss.item(), wacc.cpu() * 100, iacc.cpu() * 100, tacc.cpu() * 100, sacc.cpu() * 100])

        if total_step % args.print_iter == 0: #print_iter=50
            meters /= args.print_iter #c /= a 等效于 c = c / a，前面加了多个step的meters，现在除回来，相当于求了一个平均值
            print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
            sys.stdout.flush() #得到程序跑完，再一次性显示所有信息，而不是边跑边显示信息。
            meters *= 0 #c *= a 等效于 c = c * a，将meters重新赋值为0.

        if total_step % args.save_iter == 0: #save_iter  = 5000
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta) #跑了5000步后，就保存一个模型。
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

        if total_step % args.anneal_iter == 0: #anneal_iter =25000,由于lr会随着训练而下降，所以每隔25000步看一下lr的大小。
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta) #beta在每一个epoch，从0开始以0.001依次增加，最大为1。
