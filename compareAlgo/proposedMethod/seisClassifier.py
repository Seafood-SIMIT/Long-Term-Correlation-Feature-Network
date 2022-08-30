#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:49:33 2020

@author: sunlin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SEISClassifier(nn.Module):
    #声明带有模型参数的层，这里声明了四层
  
    def __init__(self):
        super(SEISClassifier, self).__init__()
        #假设数据格式不正常
        
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 16, kernel_size=5, stride=2,bias=True),#nx16x510
            nn.Conv1d(16, 16, kernel_size=1, stride=1,bias=True),#nx16x510
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx16x254
            nn.Conv1d(16, 32, kernel_size=5, stride=2,bias=True),#nx32x125
            nn.Conv1d(32, 32, kernel_size=1, stride=1,bias=True),#nx32x125
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx32x62
            nn.Conv1d(32, 48, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.Conv1d(48, 48, kernel_size=1, stride=1,bias=True),#nx48x29
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(48, 64, kernel_size=3, stride=2,bias=True),#nx64x6
            nn.Conv1d(64, 64, kernel_size=1, stride=1,bias=True),#nx64x6
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=1),#nx64x4
            nn.Conv1d(64, 3, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            )
        
        #self.fc1 = nn.Linear(64*64, hp.model.fc1_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),nn.ReLU(inplace=True),
            nn.Linear(8,3)
            )
       # self.fc2 = nn.Linear(hp.model.fc4_dim, hp.model.fc5_dim)
        
        
    #定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x = x.unsqueeze(1)      #[16,1,1]
        
        x = self.conv(x)
        
        x = x.view(x.size(0),-1)
        #print(x.shape)
        #x = self.fc1(x)     #x:[B,T,fc1_dim]
        x = F.softmax(x,dim=1)
        
        return x
        