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

class ACOClassifier(nn.Module):
    #声明带有模型参数的层，这里声明了四层
    
    def __init__(self):
        super(ACOClassifier, self).__init__()
        #self.hp = hp
        #假设数据格式不正常
        
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 96, kernel_size=5, stride=2,bias=True),#nx16x510
            nn.Conv1d(96, 96, kernel_size=1, stride=1,bias=True),#nx16x510
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx16x254
            #conv2
            nn.Conv1d(96, 256, kernel_size=5, stride=2,bias=True),#nx32x125
            nn.Conv1d(256, 256, kernel_size=1, stride=1,bias=True),#nx32x125
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx32x62

            #left conv
            nn.Conv1d(256, 384, kernel_size=3, stride=1,bias=True),#nx48x29
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(384, 384, kernel_size=3, stride=1,bias=True),#nx48x29
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(384, 256, kernel_size=3, stride=1,bias=True),#nx64x6
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx64x4
            )
        
        #self.fc1 = nn.Linear(64*64, hp.model.fc1_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(256*5, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,3)
            )
       # self.fc2 = nn.Linear(hp.model.fc4_dim, hp.model.fc5_dim)
        
        
    #定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x = x.unsqueeze(1)      #[16,1,1]
        
        x = self.conv(x)
        
        x = x.view(x.size(0),-1)
        #print(x.shape)
        x = self.fc1(x)     #x:[B,T,fc1_dim]
        #x = F.softmax(x)
        
        return x
        
    
if __name__=='__main__':
    fake_input = torch.rand((32,1024))
    hp=0
    model=ACOClassifier(hp)
    predict=model(fake_input)
    print(predict)