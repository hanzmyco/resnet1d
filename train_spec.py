"""
test on spec data

Shenda Hong, Oct 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

from util import read_data_generated
from resnet1d import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import math


if __name__ == "__main__":
    
    # make data



    task = 'salt'

    if task == 'an' or task == 'acid':
        n_samples = 316
        split_size = [265,51]
    
    elif task == 'od':
        n_samples = 313
        split_size = [262,51]
    
    else:
        n_samples = 281
        split_size = [230,51]
    




    #n_samples = 316
    #n_samples = 281
    n_length = 22
    n_channel = 1
    
    n_classes = 1
    data = []
    label = []

    data_file = 'data/'+task+'-quartz.txt'

    for line in open(data_file):
        if line.startswith('序号'):
            continue
        line = line.strip().split('\t')
        
        data.append(np.transpose(np.asarray([[int(ite)] for ite in line[6].split(',')])))
        label.append([float(line[8])])
        
    #print(labels)
    #print(features)
    data = np.asarray(data)

    label = np.asarray(label)

    




    n_epochs = 5000
    
    # make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## change the hyper-parameters for your own data
    # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
    # 34 layer (16*2+2): 16, 2, 4
    # 98 layer (48*2+2): 48, 6, 12
    model = ResNet1D(
        in_channels=n_channel, 
        base_filters=128, 
        kernel_size=16, 
        stride=2, 
        n_block=48, 
        groups=32,
        n_classes=n_classes, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        verbose=False)
    model.to(device)
    summary(model, (data.shape[1], data.shape[2]))
    #exit()


    # train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.MSELoss()
    
    all_loss = []
    test_loss = []

        
    #model.load_state_dict(torch.load('model_files/model_acid.pt'))
    dataset = MyDataset(data, label)
    train_set, test_set = torch.utils.data.random_split(dataset, split_size)
    dataloader = DataLoader(train_set, batch_size=32)
    dataloader_test = DataLoader(test_set, batch_size=51, drop_last=False)
        
    
    for i in range(0,n_epochs):
        out_file = 'logs/' + task +'/log.txt' 
        #out_file = 'logs/result_salt.txt'
        f = open(out_file, 'a')
    
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
    
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            single_loss = math.sqrt(loss.item())
            all_loss.append(single_loss)
            #print(single_loss)



        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)

            single_loss = math.sqrt(loss.item())
            print(single_loss)
            f.write(str(single_loss))
            f.write('\n')

        f.close()

            
        

    
        
    #plt.plot(all_loss)
    #model_file = 'model_files/model_salt.pt'
    model_file = 'model_files/model_' + task+'.pt' 
    torch.save(model.state_dict(), model_file)
    f.close()
    
    '''
    # test
    data_test, label_test = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
    print(data_test.shape, Counter(label_test))
    dataset_test = MyDataset(data_test, label_test)
    dataloader_test = DataLoader(dataset_test, batch_size=64, drop_last=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    ## classification report
    print(classification_report(all_pred, label_test))
    '''    
    
    
    