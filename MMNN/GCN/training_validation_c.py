import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from models.gat_gcn_c import Model
from models.gat_gcn_bond import GAT_GCN
from models.ginconv import GINConvNet
from models.DNN import DNN
from models.GRU import GRU
from models.CNN import CNN
from models.densenet import DenseNet
from utils_c import *
import matplotlib.pyplot as plt
from padelpy import padeldescriptor
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
        data = data.to(device)
        optimizer.zero_grad()
        output,w = model(data)
        loss = loss_fn(output, data.y.view(-1,1).float()).to(device)
        loss = torch.mean(loss).float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))                                                                                                                                 	
    total_train = torch.cat((total_train, output.cpu()), 0)
    total_label = torch.cat((total_label, data.y.view(-1, 1).cpu()), 0)
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
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses),w

modeling = [Model,DNN,GRU,CNN,GINConvNet,DenseNet,GAT_GCN][int(sys.argv[1])]
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
    
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_cyp')
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test = 'data/processed/cyp_test.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='cyp_train')
    test_data = TestbedDataset(root='data', dataset='cyp_test')
    valid_data = TestbedDataset(root='data', dataset='cyp_valid')
    train_set= pd.read_csv('cyp_data/cyp_train.csv')

    
    lables_unique, counts = np.unique(train_set['score'],return_counts = True)
    class_weights = [sum(counts)/ c for c in counts]
    example_weights = [class_weights[e] for e in train_set['score']]
    sampler = WeightedRandomSampler(example_weights, len(train_set['score']))
    # GCN
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, sampler=sampler,drop_last = True)
    #train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last = True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last = True) 
    # training the model
    #device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = modeling().to(device)
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = 100
    best_test_auc = 1000
    best_test_ci = 0
    best_epoch = -1
    patience = 15
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_file_name = 'model_' + model_st + '_' + 'cyp.model'
    result_file_name = 'result_' + model_st + '_' + 'cyp.csv'
    train_losses=[]
    train_accs=[]
    valid_losses=[]
    valid_accs=[]
    for epoch in range(NUM_EPOCHS):
        G_T,P_T,train_loss = train(model, device, train_loader, optimizer, epoch+1)
        print('predicting for valid data')
        G,P,loss_valid,w= predicting(model, device, valid_loader)
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
            G,P,loss,w = predicting(model, device, test_loader)
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
