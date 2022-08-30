# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:49:16 2019

@author: 小F
"""


import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
import densezd7
os.environ['CUDA_VISIBLE_DEVICES']='1'


num_classes=4
framelength=1024
inc=256
#model=smallmodel.SeismicNet(num_classes=num_classes)
result = np.zeros((4, 4))
all_correct = 0
all_num = 0


def yuchuliZ(data):

     data=data[::4]#四倍降采样成256维  
     data=data-np.mean(data)
     data = data/np.max(data)
     return torch.from_numpy(np.array(data)).float().view(1,-1)






model=densezd7.TCN(input_size=1,output_size=4, num_channels= [12,12,24,32,48,64]  , kernel_size=7, dropout=0.1)
model.load_state_dict(torch.load('/home/fanyuying/dense1/zhengdong/weights_eemd/z_model199.pth')['model_state_dict'])
model.eval()

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

pathZ=r'/home/fanyuying/two_streams_data/EEMD算法分类/测试震动'


filez=list_all_files(pathZ)

for i in range(len(filez)):#len(files)
    if filez[i].endswith('.txt'):
        correct=0
        labels=[]
        if filez[i].lower().find('largewheel')!=-1:
            label=0
        if filez[i].lower().find('smallwheel')!=-1:
            label=1
        if filez[i].lower().find('track')!=-1:
            label=2
        if filez[i].lower().find('person')!=-1:
            label=3
   
       
        
        dataz=np.loadtxt(filez[i])
       
#        print(datas.shape)
        dataz=dataz[::8]
#        print(dataz.shape)
        #num=math.floor(len(datas)/framelength)
        num=(len(dataz)-framelength+inc)//inc
        print(num)

        for j in range(num):
            
            
            data_z=dataz[j*inc:j*inc+framelength]
            data_2=yuchuliZ(data_z)
            data2 = data_2.unsqueeze(0)
           
            
            out = model(data2)
           
           
            out=out.data.numpy()
            if np.argmax(out, axis = 1) == label:
                correct+=1
#            
            labels.append(np.argmax(out, axis = 1))
            #del data,pred,output
            #gc.collect()
        #print(labels)
            pred=np.argmax(out)
            result[label,pred] = result[label,pred] + 1

        plt.subplot(211)
        plt.plot(dataz)
        plt.subplot(212)
        plt.plot(labels)
        plt.ylim(-1, 5)
        plt.suptitle('correctrate={}'.format(correct/num))
        print(correct)
        print('correct:',1.0*correct/num)
        savepath=os.path.join(r'/home/fanyuying/dense1/zhengdong/测试结果图emd/', filez[i].split('.')[-2].split('/')[-1]+'.png')
        plt.savefig(savepath)
        plt.close('all')

        all_correct += correct
        all_num += num
print('****************************************')
print (all_num)
print (all_correct)
print(all_correct/all_num)
print (result)