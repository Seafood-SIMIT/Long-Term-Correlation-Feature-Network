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
from utils.classificationPerformance import resultPerform
from sklearn.metrics import classification_report

import torch
import numpy as np
def votePerform(predict):
    ini = [0,0,0]
    for i in predict:
        ini[np.argmax(i)] += 1
    return ini == np.max(ini)

def ltcfnPerform(features,model_lstm):
    fuse_feature = np.resize(np.array(features),((500,6)))
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

def fusionProject(labels,label_per,
                 predict_lstm, predict_tcn, predict_ds,predict_ds_xiao, predict_vote_aco,predict_vote_seis,
                target_names):
    #与其他融合算法做比较
    #主要包括分类器性能、对噪声的影响
    #LTCFN performance
    fig_num = 7
    print(" ltcfn fusion performance")
    #print(labels,predict_lstm)
    resultPerform(n=7,info='ltcfn fusion', label_per=labels, predict=predict_lstm, target_names=target_names)
    
    print("tcn fusion performance")
    #resultPerform(n=8,info='Improved TCN classifier', label_per=labels, predict=predict_tcn, target_names=target_names)

    print("classic DS fusion performance")
    resultPerform(n=9,info='Classic DS Fusion', label_per=label_per, predict=predict_ds, target_names=target_names)

    print("Improced DS Xiao2020 performance")
    resultPerform(n=10,info='Improved DS Xiao2020', label_per=label_per, predict=predict_ds_xiao, target_names=target_names)

    print("Vote aco performance")
    resultPerform(n=11,info='Vote aco', label_per=labels, predict=predict_vote_aco, target_names=target_names)

    print("Vote seis performance")
    resultPerform(n=12,info='Vote seis', label_per=labels, predict=predict_vote_seis, target_names=target_names)
    #print(" seis_mfcc classifier performance")
    #classification_report(label_per,predict_seis_mfcc,target_names = target_names)
    
    print("other performance")
    #resultPerform(n=10,info='Vote', label_per=labels, predict=predict_ds_xiao, target_names=target_names)


