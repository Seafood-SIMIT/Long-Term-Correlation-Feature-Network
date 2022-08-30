# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:17:15 2018

@author: Administrator
调用数据集

"""

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
    def __init__(self,datapath='/home/fanyuying/two_streams_data/训练1震动分帧',train=True,filter_r=3,sigma=1):
        self.datapath=datapath
        self.datas=list_all_files(datapath)  
        self.length=len(self.datas)
        self.train=train
        self.filter_r=filter_r
        self.sigma=sigma
        
    def __getitem__(self,index):
        loaddatapath=self.datas[index]
        #loaddatapath = os.path.join(self.datapath,self.datas[index])
        #person:0 wheel:1 track:2 heightcar:3
        if loaddatapath.lower().find('largewheel')!=-1:
            label=0
        if loaddatapath.lower().find('smallwheel')!=-1:
            label=1
        if loaddatapath.lower().find('track')!=-1:
            label=2
        if loaddatapath.lower().find('person')!=-1:
            label=3

        data=np.loadtxt(loaddatapath)
        #start=np.random.randint(0,512)  #前闭后开
        #data=data[start:start+1024]
#        if self.train:
#            if (np.random.randint(0,10)>5):
#                data=data[::-1]
        data=data[::4]#四倍降采样成256维
        
        data=data-np.mean(data)
        data = data/np.max(data)
        return (torch.from_numpy(np.array(data)).float().view(1,-1), label)  ##cnn
        #return torch.from_numpy(np.array(data)).float().view(32,32), label  #LSTM
    def __len__(self):
        return len(self.datas)
    
  