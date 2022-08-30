# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:46:05 2018

@author: Administrator
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class SeismicNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SeismicNet, self).__init__()


#全卷积
#model1 原始数据4倍降采样成256维
        self.features1=nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=3, stride=2,bias=True),#nx127*4  (256-3+1)/2=127  31
                nn.Conv1d(16, 16, kernel_size=3, stride=1,bias=True),#nx127*16 (127-1+1)/1=127  31
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3,stride=2),#nx16x63   15
                )
      
        self.features2=nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, stride=2,bias=True),#nx31*32  7
                nn.Conv1d(32, 32, kernel_size=3, stride=1,bias=True),#nx31*32   7
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                )
        self.max2=nn.Sequential(
                nn.MaxPool1d(kernel_size=4,stride=1),#nx32x15      3
                )
        self.features5=nn.Sequential(
                nn.Conv1d(32, 4, kernel_size=1, stride=1,bias=True),#nx4x1
                )
                
                

    def forward(self, x):
        x1 = self.features1(x)              #第1层  16*63
      #  print(x1.shape)
        out2 = self.features2(x1)             #第2层  32*31
      #  print(out2.shape)
        x2 = self.max2(out2)                  #第3层  32*15


 
        x5 = self.features5(x2)             #第8层  4*1
      #  print(x5.shape)
        out = x5.view(x5.size(0), -1)       
   
        out = F.log_softmax(out,dim=1)
        return out
        
        

