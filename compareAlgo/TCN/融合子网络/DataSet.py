# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:48:25 2020

@author: 小F
"""


import random 
import numpy as np
import torch
import torch.utils.data as data
import os
def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files
class DataSet(data.Dataset):
    def __init__(self,datapath='/home/fanyuying/two_streams_data/未降噪训练集特征_相加',train=True,filter_r=3,sigma=1):
        self.datapath=datapath
        self.datas=list_all_files(datapath)  
        self.length=len(self.datas)
        self.train=train
        self.filter_r=filter_r
        self.sigma=sigma
        
    def __getitem__(self,index):
        loaddatapath=self.datas[index]

        if loaddatapath.lower().find('largewheel')!=-1:
            label=0
        if loaddatapath.lower().find('smallwheel')!=-1:
            label=1
        if loaddatapath.lower().find('track')!=-1:
            label=2
        if loaddatapath.lower().find('person')!=-1:
            label=3
        data=np.loadtxt(loaddatapath)

        return (torch.from_numpy(np.array(data)).float().view(2,-1), label)  ##cnn
#numpy中的ndarray转化成pytorch中的tensor
        #return torch.from_numpy(np.array(data)).float().view(32,32), label  #LSTM
    def __len__(self):
        return len(self.datas)
    
  