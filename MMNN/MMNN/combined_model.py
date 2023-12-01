import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from models.gat_gcn_bond import GAT_GCN
from models.ginconv import GINConvNet
from models.DNN import DNN
from models.GRU import GRU
from models.CNN import CNN
from models.mlp import MLP
from models.densenet import DenseNet
import matplotlib.pyplot as plt
from padelpy import padeldescriptor
from utils_c import *
#from sampler import ImbalancedDatasetSampler

torch.manual_seed(0)

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train = torch.Tensor()
    total_label = torch.Tensor()
    train_losses = []
    for batch_idx, data in enumerate(train_loader):
        data0 = data[0].to(device)
        data1 = data[1].to(device)
        optimizer.zero_grad()
        output,w = model(data0)
        loss = loss_fn(output, data1.view(-1, 1).float()).to(device)
        loss = torch.mean(loss).float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data0),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))                                                                                                                                 	
    total_train = torch.cat((total_train, output.cpu()), 0)
    total_label = torch.cat((total_label, data1.view(-1, 1).cpu()), 0)
    G_train = total_label.detach().numpy().flatten()
    P_train = total_train.detach().numpy().flatten()
    ret = [auc(G_train,P_train),pre(G_train,P_train),recall(G_train,P_train),f1(G_train,P_train),acc(G_train,P_train),mcc(G_train,P_train),spe(G_train,P_train)]
    print('train_auc',ret[0])
    print('train_pre',ret[1])
    print('train_recall',ret[2])
    print('train_f1',ret[3])
    print('train_acc',ret[4])
    print('train_mcc',ret[5])
    print('train_spe',ret[6])
    print('train_loss',np.average(train_losses))
    return G_train, P_train, np.average(train_losses)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    losses = []
    with torch.no_grad():
        for data in loader:
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            output,w = model(data1)
            loss = loss_fn(output, data2.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data2.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses)

def predicting_model(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses)

class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
        
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
#device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

model_GCNN_name = 'model_GAT_GCN_cyp.model'
model_GRU_name = 'model_GRU_cyp.model'
model_DNN_name = 'model_DNN_cyp.model'

modeling = MLP
modeling_GCNN = GAT_GCN
modeling_DNN = DNN
modeling_GRU = GRU
model_GCNN = modeling_GCNN().to(device)
model_DNN = modeling_DNN().to(device)
model_GRU = modeling_GRU().to(device)
model = modeling().to(device)
model_GCNN.load_state_dict(torch.load(model_GCNN_name))
model_GRU.load_state_dict(torch.load(model_GRU_name))
model_DNN.load_state_dict(torch.load(model_DNN_name))

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test = 'data/processed/cyp_test.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'

train_data = TestbedDataset(root='data', dataset='cyp_train')
test_data = TestbedDataset(root='data', dataset='cyp_test')
valid_data = TestbedDataset(root='data', dataset='cyp_valid')

train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False,drop_last = True)
#train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last = True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last = True) 
    
G_1,P_1,loss_1 = predicting_model(model_GCNN, device, train_loader)
G_2,P_2,loss_2 = predicting_model(model_GRU, device, train_loader)
G_3,P_3,loss_3= predicting_model(model_DNN, device, train_loader)
a = list(zip(P_1,P_2,P_3))
train_predic =torch.Tensor([list(a[i]) for i in range(len(a))])
train_true =torch.Tensor(G_1)

G_11,P_11,loss_11 = predicting_model(model_GCNN, device, valid_loader)
G_22,P_22,loss_22 = predicting_model(model_GRU, device, valid_loader)
G_33,P_33,loss_33 = predicting_model(model_DNN, device, valid_loader)
b = list(zip(P_11,P_22,P_33))
valid_predic =torch.Tensor([list(b[i]) for i in range(len(b))])
valid_true =torch.Tensor(G_11)

G_111,P_111,loss_111 = predicting_model(model_GCNN, device, test_loader)
G_222,P_222,loss_222 = predicting_model(model_GRU, device, test_loader)
G_333,P_333,loss_333 = predicting_model(model_DNN, device, test_loader)
c = list(zip(P_111,P_222,P_333))
test_predic =torch.Tensor([list(c[i]) for i in range(len(c))])
test_true =torch.Tensor(G_111)

print('Learning rate: ', LR)

#train_torch = GetLoader(np.array(train_predic),np.array(train_true))
#valid_torch = GetLoader(np.array(valid_predic),np.array(valid_true))
#test_torch = GetLoader(np.array(test_predic),np.array(test_true))
#print(train_predic.shape)
train_torch = GetLoader(train_predic,train_true)
valid_torch = GetLoader(valid_predic,valid_true)
test_torch = GetLoader(test_predic,test_true)
train_c = DataLoader(train_torch, batch_size=512, shuffle=False, drop_last=False)
valid_c = DataLoader(valid_torch, batch_size=512, shuffle=False, drop_last=False)
test_c = DataLoader(test_torch, batch_size=512, shuffle=False, drop_last=False)

# training the model
best_loss = 100
best_test_auc = 1000
best_test_ci = 0
best_epoch = -1
patience = 15
early_stopping = EarlyStopping(patience=patience, verbose=True)
model_file_name = 'model_mlp_' + 'cyp.model'
result_file_name = 'result_mlp' + 'cyp.csv'
train_losses=[]
train_accs=[]
valid_losses=[]
valid_accs=[]
for epoch in range(NUM_EPOCHS):
    G_T,P_T,train_loss= train(model, device, train_c, optimizer, epoch+1)
    print('predicting for valid data')
    G,P,loss_valid= predicting(model, device, valid_c)
    loss_valid_value = loss_valid
    print('valid_loss',loss_valid)
    print('valid_auc',auc(G,P))
    print('valid_pre',pre(G,P))
    print('valid_recall',recall(G,P))
    print('valid_f1',f1(G,P))
    print('valid_acc',acc(G,P))
    print('valid_mcc',mcc(G,P))
    print('valid_spe',spe(G,P))
    train_losses.append(np.array(train_loss))
    valid_losses.append(loss_valid)
    train_accs.append(acc(G_T,P_T))
    valid_accs.append(acc(G,P))
    b = pd.DataFrame({'value':G,'prediction':P})
    names = 'model_'+'value_validation'+'.csv'
    b.to_csv(names,sep=',') 
    early_stopping(loss_valid, model, model_file_name)
    if early_stopping.early_stop:
        print("Early stopping")
        print('predicting for test data')
        model.load_state_dict(torch.load(model_file_name))
        G,P,loss = predicting(model, device, test_c)
        ret = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
        print('cyp_2c19 ',best_epoch,'auc',ret[0],'pre',ret[1],'recall',ret[2],'f1',ret[3],'acc',ret[4],'mcc',ret[5],'spe',ret[6])
        a = pd.DataFrame({'value':G,'prediction':P})
        name = 'model_'+'value_test'+'.csv'
        a.to_csv(name,sep=',')
        break
    else:
        print('no early stopping')
df = pd.DataFrame({'train_loss':train_losses,'valid_loss':valid_losses,'train_accs':train_accs,'valid_accs':valid_accs})
names = 'model_'+'loss_acc'+'.csv'
df.to_csv(names,sep=',')