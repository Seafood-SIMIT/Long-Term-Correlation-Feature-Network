#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:48:24 2022

@author: sunlin
@融合结果计算以及算法比较
"""
import compareAlgo
from utils.accCalculator import theOutputfromArray
from utils.confusionMatrixGenerator import confusionMatrixGenerator,plot_confusion_matrix
import torch
import numpy as np
def ltcfnPerform(features,model_lstm):
    fuse_feature = np.resize(np.array(features),(500,6))
    #fuse_feature =np.appendnp.vstack(predict_aco)
    fuse_feature = torch.tensor(fuse_feature,dtype=torch.float).unsqueeze(0)
    predict_lstm = model_lstm(fuse_feature).data.numpy()
    return predict_lstm
def resultFusion(predict_lstm,labels):
    cm_lstm,acc_lstm = confusionMatrixGenerator(predict_lstm, labels)

    plot_confusion_matrix(cm_lstm,7,'confusion matrix of LTCFN')
    return acc_lstm
def resultFusionCompareWithOther(pba_aco,pba_seis,
                                 label):
    #--------------------------Classic DS Theory
    acc_classic_DS = compareAlgo.DS.classicDSFusion(np.array(pba_aco),np.array(pba_seis),np.array(label))
    #-------------------------Hu2014
    acc_Hu_2014 = compareAlgo.DS.hu2014DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    #-------------------------Xiao2020
    acc_Xiao_2020 = compareAlgo.DS.xiao2020DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    #-------------------------TCN
    #-------------------------Alamir
    #-------------------------fuzzy
    return [acc_classic_DS,acc_Hu_2014,acc_Xiao_2020]