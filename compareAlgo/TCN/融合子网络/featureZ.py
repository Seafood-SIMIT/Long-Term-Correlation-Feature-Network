# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:10:56 2019

@author: 小F
"""


import torch
import os
import dense_zd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
model1 =  dense_zd.TCN(input_size=1,output_size=4, num_channels= [12,12,24,32,48,64] , kernel_size=7, dropout=0.2)  
model1.load_state_dict(torch.load('/home/fanyuying/dense1/zhengdong_dense_shoushen/weights/z_model199.pth')['model_state_dict'])

model1.eval()

def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files
path = '/home/fanyuying/two_streams_data/训练集震动_分帧'
files1=list_all_files(path)  #得到path下所有的目录

for j in range(len(files1)):
    data=np.loadtxt(files1[j])
    name = files1[j].split('/')[-1]
    data=data[::4]#四倍降采样成256维
    data=data-np.mean(data)
    data = data/np.max(data)
    data = torch.from_numpy(np.array(data)).float().view(1,-1)
    data = data.unsqueeze(0)
    out = model1(data)
    out = out.view(1,-1)
    out = out.permute(1,0)
    print(out.shape)
    savepath=os.path.join(r'/home/fanyuying/two_streams_data/训练集特征','[S]'+name)
    np.savetxt(savepath,out.data.numpy())
            
