# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:02:41 2018

@author: Administrator
"""

##对原始数据分帧保存
from __future__ import division
import numpy as np
import math
import os
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


def list_all_files(rootdir):   #找到所有的文件，并返回文件列表
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))  # [3,4].extend([1,2])  result:[3,4,1,2]
           if os.path.isfile(path):
              _files.append(path)           #[3,4].append([1,2])  result:[3,4,[1,2]]
    return _files   #[path1,path2,,,,pathn]
    print('the num of files is:{}'.format(files[i]))


old_path='/home/fanyuying/two_streams_data/训练1提取的特征'

files=list_all_files(old_path)
print('the num of files is:{}'.format(len(files)))
for j in tqdm(range(len(files))):
    prefix = files[j].split('[')[0]
    ednfix = files[j].split('[')[1]
    if ednfix[0] == 'A':
        if files[j].endswith('.txt'):
            data=np.loadtxt(files[j])   #[1,2,3,,,n]  [[1,2,3,4],[5,6,7,8],,,[]]
            data = np.reshape(data,[-1,1])
            print(files[j])
            print(prefix)
            print(ednfix)
            S_name = prefix + '[S' + ednfix[1:]
            print(S_name)

            data_s = np.loadtxt(S_name)
            data_s = np.reshape(data_s,[-1,1])
            print(data.shape)
            print(data_s.shape)

            #data = data_s+data
            data = np.concatenate([data_s,data],axis=1)
            print(data.shape)

            savepath =  os.path.join('/home/fanyuying/two_streams_data/训练1提取的特征融合','[1+1'+ ednfix[1:])
           
            print(savepath)
            np.savetxt(savepath, data)

    # if j==5:
    #     break