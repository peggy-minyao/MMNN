import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils_c import *
import nni


params = {
    'features': 512,
    'lr': 0.001,
    'momentum': 0,
    'dropout': 0.3,
    'optimizer':0.9
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# GCN-CNN based model
class DNN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_m=1690, output_dim=128, dropout=0.5):

        super(DNN, self).__init__()

        self.n_output = n_output
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(params['dropout'])
        #DNN
        self.batch_norm1 = nn.BatchNorm1d(num_features_m)
        self.linear1=nn.Linear(num_features_m,params['features'])
        self.batch_norm2 = nn.BatchNorm1d(params['features'])
        self.linear2=nn.Linear(params['features'],params['features']//2)
        self.batch_norm3 = nn.BatchNorm1d(params['features']//2)
        self.linear3=nn.Linear(params['features']+params['features']//2,params['features']//2)
        self.batch_norm4 = nn.BatchNorm1d(params['features']//2)
       # self.linear4=nn.Linear(3500,output_dim)
        self.linear5=nn.Linear(params['features']//2,params['features']//4)
        self.batch_norm5 = nn.BatchNorm1d(params['features']//4)
        self.linear6=nn.Linear(params['features']*2+params['features']//4,params['features'])
        self.linear7=nn.Linear(params['features'],params['features']//2)
        self.out = nn.Linear(params['features']//2, self.n_output)

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
        


#trainning


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
    num_batches = len(loader)
    model.eval()
    test_loss = 0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        losses = []
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            test_loss += torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            ACC = acc(total_labels.numpy().flatten(),total_preds.numpy().flatten())
    return ACC,total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses)

modeling =DNN
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
    
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_cyp')
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test = 'data/processed/cyp_test.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='cyp_train')
#    test_data = TestbedDataset(root='data', dataset='cyp_test')
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
#    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last = True) 
    # training the model
    #device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = modeling().to(device)
    print(model)
    loss_fn = nn.BCELoss()
    '''
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.7, weight_decay=5e-4)
    if params['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=params['lr'])
    if params['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params['lr'])
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])'''
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'],momentum=params['momentum'])
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
        acc_valid,G,P,loss_valid= predicting(model, device, valid_loader)
        nni.report_intermediate_result(acc_valid)
        loss_valid_value = loss_valid
        early_stopping(loss_valid, model,model_file_name)
        if early_stopping.early_stop:
            print("Early stopping")
            print('predicting for test data')
            model.load_state_dict(torch.load(model_file_name))
            acc_valid,G,P,loss_valid= predicting(model, device, valid_loader)
 #           accs,G,P,loss = predicting(model, device, test_loader)
            nni.report_final_result(acc_valid)
            break
        else:
            print('no early stopping')
    
