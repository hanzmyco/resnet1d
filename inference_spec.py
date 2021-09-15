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
    n_samples = 316
    n_length = 22
    n_channel = 1

    n_classes = 1
    data = []
    label = []

    for line in open('data/an-quartz.txt'):
        if line.startswith('序号'):
            continue
        line = line.strip().split('\t')

        data.append(np.transpose(np.asarray(
            [[int(ite)] for ite in line[6].split(',')])))
        label.append([float(line[8])])

    #print(labels)
    #print(features)
    data = np.asarray(data)

    label = np.asarray(label)

    n_epochs = 1500

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

    data_test = data
    label_test = label
    model.load_state_dict(torch.load('model_files/model.pt'))


    dataset_test = MyDataset(data_test, label_test)
    dataloader_test = DataLoader(dataset_test, batch_size=326, drop_last=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    model.eval()

    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        pred = pred.cpu().data.numpy()
        for ite in pred:
            all_pred_prob.append(float(ite))

        #all_pred_prob.append(pred)
    
    print(all_pred_prob)
    print(label)
    label_list = []
    for ite in label:
        label_list.append(float(ite))

    print(label_list)

    f = open('diff.txt', 'a')
    for index in range(0,len(label_list)):
        f.write(str(abs(label_list[index] - all_pred_prob[index])/label_list[index] * 100)+'%')
        f.write('\n')
    f.close()



    


    
