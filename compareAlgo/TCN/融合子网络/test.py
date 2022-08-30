# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:53:46 2020

@author: 小F
"""

import numpy as np
import math
import os
import torch
from torch.autograd import Variable
import densesy7
import densezd7
import smallmodel
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_classes =4
framelength = 1024
inc = 256

result = np.zeros((4, 4))
all_correct = 0
all_num = 0


def yuchuliz(data):
    data = data[::4]  # 四倍降采样成256维
    data = data - np.mean(data)
    data = data / np.max(data)
    return torch.from_numpy(np.array(data)).float().view(1, -1)


def yuchulis(data):
    data = data[::4]  # 四倍降采样成256维
    data = data - np.mean(data)
    data = data / np.max(data)
    return torch.from_numpy(np.array(data)).float().view(4, -1)


modelz = densezd7.TCN(input_size=1,output_size=4, num_channels= [12,12,24,32,48,64] , kernel_size=7, dropout=0.2)  
modelz.load_state_dict(torch.load('/home/fanyuying/dense1/zhengdong/weights_xiugai1/z_model199.pth')['model_state_dict'])
modelz.eval()

models =  densesy7.TCN(input_size=4,output_size=4, num_channels= [12,12,24,32,48,64] , kernel_size=7, dropout=0.2)  
models.load_state_dict(torch.load('/home/fanyuying/dense1/shengyin/weights_xiugai1/z_model160.pth')['model_state_dict'])
models.eval()

model=smallmodel.SeismicNet()
model.load_state_dict(torch.load('/home/fanyuying/dense1/Fullconnect/weights_eemd/z_model99.pth')['model_state_dict'])
model.eval()


def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


path1 = r'/home/fanyuying/two_streams_data/EEMD算法分类/测试震动' #输入为测试集的分帧数据

file = list_all_files(path1)

for i in range(len(file)):
    prefix = file[i].split('[')[0]
    ednfix = file[i].split('[')[1]
    correct = 0
    labels = []
    if ednfix[0] == 'S':

        if file[i].lower().find('largewheel') != -1:
            label = 0
        if file[i].lower().find('smallwheel') != -1:
            label = 1
        if file[i].lower().find('track') != -1:
            label = 2
        if file[i].lower().find('person') != -1:
            label = 3


        dataz = np.loadtxt(file[i])
        dataz = dataz[::8]
        print(dataz.shape)
       
        Sy_name = prefix + '[A' + ednfix[1:]
        datas = np.loadtxt(Sy_name)
        print(datas.shape)
        datas = datas[::8]
        
        num = (len(dataz) - framelength + inc) // inc
        print(num)
        for j in range(num):

            data_z = dataz[j * inc:j * inc + framelength]
            data_2 = yuchuliz(data_z) #(1,256)
            # print(data_2.shape)
            data2 = data_2.unsqueeze(0)# (1,1,256)
            # print(data2.shape)
            # data2 = data2.permute(0,2,1) ###(1,256,1)
            # print(data2.shape)
            outz = modelz(data2)
            outz = outz.view(1,-1)##
            outz = outz.permute(1, 0)  # (137,1)


            data_s = datas[j * inc:j * inc + framelength]
            data_2 = yuchulis(data_s)
            data2 = data_2.unsqueeze(0)
            outs = models(data2)  # (1,137)
            outs = outs.view(1,-1)##
            outs = outs.permute(1, 0)

            out = torch.cat((outz, outs), 1)
            out = out.view(2,-1)##
            print(out.shape)
            ####out = out.permute(1, 0)
            #           print(out.shape)
            out = out.unsqueeze(0)
            #           print(out.shape)
            out = model(out)
            out = out.data.numpy()
            print(out)

 
  
            print(label)
            if np.argmax(out, axis=1) == label:
                correct += 1
            #
            pred = np.argmax(out, axis=1)
            print(pred)
            labels.append(pred)

            result[label, pred] = result[label, pred] + 1

        plt.subplot(311)
        plt.plot(dataz)
        plt.subplot(312)
        plt.plot(datas)
        plt.subplot(313)
        plt.plot(labels)
        plt.ylim(-1, 5)
        plt.suptitle('correctrate={}'.format(1.0 * correct / num))
        print('correct:', 1.0 * correct / num)
        path2 = '/home/fanyuying/dense1/Fullconnect/测试结果图'
        # print(file[i].split('.')[-2].split('\\')[-1])
        savepath = os.path.join(path2, file[i].split('.')[-2].split('/')[-1] + '.png')
        print(savepath)
        plt.savefig(savepath)
        plt.close('all')

        all_correct += correct
        all_num += num
print('****************************************')
print(all_num)
print(all_correct)
print(1.0 * all_correct / all_num)
print(result)


