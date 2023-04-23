
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import math
class SEISAlexNetBiLSTM(nn.Module):
    #alex type network with Bi-LSTM
  
    def __init__(self,fc_outputs):
        super(SEISAlexNetBiLSTM, self).__init__()
        #假设数据格式不正常
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 16, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(16, 64, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14

            nn.Conv1d(64, 64, kernel_size=3, stride=2,bias=True),#nx64x6
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, stride=1,bias=True),#nx64x6
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1, stride=1,bias=True),#nx64x6
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=1),#nx64x4

            )
        #self.rnn = nn.LSTM(hp.lstm.input_size,hp.lstm.hidden_size,hp.lstm.num_layers,bias=True,bidirectional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(3584,64),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,fc_outputs),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(3),
            #nn.Sigmoid(),
        ) 
    #定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        #print(output.shape)
        #output = torch.cat((output[0],output[-1]),-1)
        #print(output.shape)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        return x


if __name__=="__main__":
            
    fc_outputs = 3
    model=AlexNetBiLSTM(fc_outputs)
    x = torch.rand(2,1024)
    output = model(x)
    print( 'out:', output)