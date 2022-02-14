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

def classificationPerform(model_aco,model_seis,frame_data_aco, frame_data_seis):
    predict_aco=model_aco(torch.tensor(frame_data_aco, dtype=torch.float).reshape(1,-1)).data.numpy()
    predict_seis=model_seis(torch.tensor(frame_data_seis, dtype=torch.float).reshape(1,-1)).data.numpy()
    return predict_aco, predict_seis

def resultClassifier(predict_aco,predict_seis,label_per,frame_length):
    cm_aco = confusionMatrixGenerator(predict_aco, label_per)
    cm_seis = confusionMatrixGenerator(predict_seis, label_per)
    #图片1、2，分类器的混淆矩阵
    plot_confusion_matrix(cm_aco,1,'confusion matrix of acoustic classifier')
    plot_confusion_matrix(cm_seis,2,'confusion matrix of seismic classifier')
    #输出准确率
    return accCalculateFrame(predict_aco,label_per,frame_length),accCalculateFrame(predict_seis,label_per,frame_length)

def resultClassifierCompareWithOther(predict_aco,predict_seis,label_per):
    '''
    aco

    '''
    #--------------------------MFCC
    #--------------------------wavelet
    '''
    seis
    '''
    #--------------------------wavelet
    
    
    
    