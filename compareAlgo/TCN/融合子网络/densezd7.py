# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:19:57 2020

@author: 小F
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)


        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

      

        self.net = nn.Sequential(self.conv1,self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2,self.chomp2, self.relu2,self.dropout2 )


    def forward(self, x):
        out = self.net(x)
        return out
    


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            if i==0:
                in_channels = num_inputs
            else:
                in_channels = in_channels+num_channels[i-1]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.network:
            out = layer(x)
           # print(x.shape)
           # print(out.shape)
            x = torch.cat((out,x),dim=1)
           # print('拼接后：',x.shape)
        return x
    
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        
       
        nn.Conv1d(in_channel, out_channel,kernel_size=3),
        
        nn.BatchNorm1d( out_channel),
        nn.ReLU()
 
    )
    return trans_layer

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.transition = transition(193,64)
        self.linear = nn.Linear(64, output_size)
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)   (256,64,20)
        y1 = self.transition(y1)
        o = self.linear(y1[:, :, -1])
        return y1[:, :, -1]
        #return F.log_softmax(o, dim=1)


