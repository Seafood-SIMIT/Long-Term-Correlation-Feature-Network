#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:15:37 2022

@author: sunlin

@实现声音震动的单通道模型准确率计算和混淆矩阵生成
"""
from utils.confusionMatrixGenerator import confusionMatrixGenerator,plot_confusion_matrix
from utils.accCalculator import accCalculateFrame

import torch
import numpy as np

def classificationPerform(model_aco,model_seis,frame_data_aco, frame_data_seis):
    predict_aco=model_aco(torch.tensor(frame_data_aco, dtype=torch.float).reshape(1,-1)).data.numpy()
    predict_seis=model_seis(torch.tensor(frame_data_seis, dtype=torch.float).reshape(1,-1)).data.numpy()
    return predict_aco, predict_seis

def resultClassifier(predict_aco,predict_seis,predict_aco_mfcc,predict_seis_medium,predict_aco_wavelet,label_per,predict_seis_wavelet,frame_length):
    cm_aco,acc_aco = confusionMatrixGenerator(predict_aco, label_per)
    cm_seis,acc_seis  = confusionMatrixGenerator(predict_seis, label_per)
    cm_aco_mfcc,acc_aco_mfcc  = confusionMatrixGenerator(predict_aco_mfcc, label_per)
    cm_seis_medium,acc_seis_medium  = confusionMatrixGenerator(predict_seis_medium, label_per)
    cm_aco_wavelet,acc_aco_wavelet  = confusionMatrixGenerator(predict_aco_wavelet, label_per)
    cm_seis_wavelet,acc_seis_wavelet  = confusionMatrixGenerator(predict_seis_wavelet, label_per)
    #图片1、2，分类器的混淆矩阵
    plot_confusion_matrix(cm_aco,1,'confusion matrix of acoustic classifier')
    plot_confusion_matrix(cm_seis,2,'confusion matrix of seismic classifier')
    plot_confusion_matrix(cm_aco_mfcc,3,'confusion matrix of acoustic classifier with MFCC')
    plot_confusion_matrix(cm_seis_medium,4,'confusion matrix of seismic classifier with medium scale')
    plot_confusion_matrix(cm_aco_wavelet,5,'confusion matrix of acoustic classifier with wavelet')
    plot_confusion_matrix(cm_seis_wavelet,6,'confusion matrix of seismic classifier with wavelet')
    #输出准确率
    return [acc_aco,acc_seis,acc_aco_mfcc,acc_seis_medium,acc_aco_wavelet,acc_seis_wavelet]
def resultClassifierCompareWithOther(predict_aco,predict_seis,
                                     label_per,
                                     predict_aco_mfcc,predict_seis_medium,
                                     predict_aco_wavalet,predict_seis_wavalet):
    '''
    aco

    '''
    #--------------------------MFCC
    
    #--------------------------wavelet
    '''
    seis
    '''
    #--------------------------wavelet
    
    
    
    