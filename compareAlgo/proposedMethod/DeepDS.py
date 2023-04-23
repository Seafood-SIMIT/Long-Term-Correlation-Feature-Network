#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 20:29:51 2021

@author: sunlin
"""

import torch
import torch.nn as nn

class DeepDS(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DeepDS,self).__init__()
        self.lstm=nn.Sequential(
            nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=2,batch_first=True),
        )
        self.fc=nn.Sequential(
            nn.Linear(hidden_size*2,3),
            nn.ReLU(),
            )
        
    def forward(self,x):
        x,_ = self.lstm(x)
        #print(x.shape)
        x = torch.cat((x[:,0],x[:,-1]),-1)
        x = self.fc(x)
        return x

if '__name__' == '__main__':
    x = torch.randn([1,500,6])
    deep_ds = DeepDS(6,16)
    print(deep_ds(x))