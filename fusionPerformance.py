#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:48:24 2022

@author: sunlin
@融合结果计算以及算法比较
"""
import torch
import numpy as np
def resultLSTM(predict_aco,predict_seis,label_per):
    fuse_feature = np.resize(np.hstack([np.vstack(predict_aco),np.vstack(predict_seis)]),(500,6))
    #fuse_feature =np.appendnp.vstack(predict_aco)
    fuse_feature = torch.tensor(fuse_feature,dtype=torch.float).unsqueeze(0)
    predict_lstm = model_lstm(fuse_feature).data.numpy()
    return predict_lstm

def resultFusionCompareWithOther():
    #--------------------------Classic DS Theory
    acc_classic_DS = classicDSFusion(np.array(pba_aco),np.array(pba_seis),label)
    print("Classic D-S Theory: acc: %.2f"%(acc_classic_DS))
    #-------------------------Hu2014
    acc_Hu_2014 = hu2014DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    print("Hu 2014  D-S Theory: acc: %.2f"%(acc_Hu_2014))
    #-------------------------Xiao2020
    acc_Xiao_2020 = xiao2020DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    print("Xiao 2020  D-S Theory: acc: %.2f"%(acc_Xiao_2020))
    #-------------------------TCN
    #-------------------------Alamir
    #-------------------------fuzzy
    