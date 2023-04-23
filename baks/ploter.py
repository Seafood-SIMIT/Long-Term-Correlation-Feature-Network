#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:56:23 2021

@author: sunlin
"""

'''
    def __init__(self):
        self.filename=[]
        self.label=[]
        self.acc_origin_aco=[]
        self.acc_origin_seis=[]
        self.acc_classic_DS=[]
        self.pba_origin_aco=[]
        self.pba_origin_seis=[]
        self.frame_number=[]
''' 
import pickle
import torch
import numpy as np
from utils.fuseFeature import seriFeatureCal

from matplotlib import pyplot as plt
df=open('output/result.pkl','rb')
data = pickle.load(df)

for file_index in range(1):
    single_file =  data[file_index]
    print("target type: %d"%(single_file.label))
    pba_seirs_aco=np.array(single_file.pba_origin_aco).reshape(-1,3)
    pba_seirs_aco = torch.softmax(torch.tensor(pba_seirs_aco),dim=1)
    pba_seirs_seis=np.array(single_file.pba_origin_seis).reshape(-1,3)
    plt.subplot(2, 2, 1)
    plt.plot(pba_seirs_aco)
    plt.title("origin_aco_distribute")
    plt.subplot(2, 2, 2)
    plt.plot(pba_seirs_seis)
    plt.title("origin_seis_distribute")
    plt.subplot(2,2,3)
    seri_feature = seriFeatureCal(pba_seirs_aco,pba_seirs_seis)
    plt.plot(seri_feature)
    plt.title('feature')
    plt.show()