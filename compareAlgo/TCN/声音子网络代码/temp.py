# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 10:53:19 2019

@author: Simit
"""
import os

#==============================================================================
# There are all variables
#==============================================================================
weights_123 = './weights_xiugai1'
txt_path = './weights'

if not os.path.exists(weights_123):
    os.mkdir(weights_123)

files_list = os.listdir(weights_123)
if len(files_list)  !=0 :
    para_num = [int(i.split('l')[1].split('.')[0]) for i in files_list ]
    para_num.sort()
    lasted = str(para_num[-1])
    first = para_num[0]
    lasted_name = 'z_model{}.pth'.format(lasted)
    model_lasted_path = os.path.join(weights_123,lasted_name)
else:
    model_lasted_path = 0
    lasted = 0
    first=-1
    
if __name__ == '__main__':
    print(para_num)
    print(model_lasted_path)