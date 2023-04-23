#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:15:37 2022

@author: sunlin

@实现声音震动的单通道模型准确率计算和混淆矩阵生成
"""
from cProfile import label
from utils.confusionMatrixGenerator import confusionMatrixGenerator,plot_confusion_matrix
from utils.accCalculator import accCalculateFrame

from sklearn.metrics import classification_report
import torch
import numpy as np

def classificationPerform(model_aco,model_seis,frame_data_aco, frame_data_seis):
    predict_aco=model_aco(torch.tensor(frame_data_aco, dtype=torch.float).reshape(1,-1)).data.numpy()
    predict_seis=model_seis(torch.tensor(frame_data_seis, dtype=torch.float).reshape(1,-1)).data.numpy()
    return predict_aco.reshape(-1), predict_seis.reshape(-1)

def resultClassifier(predict_aco,predict_seis,predict_aco_mfcc,predict_seis_medium,predict_aco_wavelet,
                    predict_seis_wavelet,label_per):
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
    return [acc_aco,acc_seis,acc_aco_mfcc,acc_seis_medium,acc_aco_wavelet,acc_seis_wavelet ]

def resultPerform(n,info,label_per,predict,target_names):
    #label_per = np.argmax(np.array(label_per),axis=1)
    #label_per = [np.argmax(i) for i in label_per]
    predict = [np.argmax(i) for i in predict]
    #predict = np.argmax(np.array(predict),axis=1)
    #y = (predict == predict.max(axis=1,keepdims=1)).astype(int)
    #print(predict,label_per)
    #print(label_per,predict)
    print(classification_report(label_per,predict,target_names = target_names))
    cm_aco,acc_aco = confusionMatrixGenerator(predict, label_per)
    plot_confusion_matrix(cm_aco,n,'confuse_matrix of '+info)

def classificationProject(label_per,
                            predict_aco,predict_aco_mfcc,predict_aco_wavelet,
                            predict_seis,predict_seis_medium,predict_seis_wavelet,
                            target_names):
    # frame classifier performance
    print(" aco AlexNet classifier performance")
    resultPerform(1,'acoustic classifier',label_per=label_per, predict=predict_aco, target_names=target_names)


    print(" seis AlexNet classifier performance")
    resultPerform(2,'seismic classifier',label_per=label_per,predict=predict_seis,target_names=target_names)

    print(" aco_mfcc classifier performance")
    resultPerform(3,'mfcc classifier',label_per=label_per,predict=predict_aco_mfcc,target_names=target_names)

    print(" aco_wavelet classifier performance")
    resultPerform(4,'ACO wavelet classifier',label_per=label_per,predict=predict_aco_wavelet,target_names=target_names)

    print(" seis_wavelet classifier performance")
    resultPerform(5,'SEIS wavelet classifier',label_per=label_per,predict=predict_seis_wavelet,target_names=target_names)

    print(" medium classifier performance")
    resultPerform(6,'SEIS medium scale classifier',label_per=label_per,predict=predict_seis_medium,target_names=target_names)

    # frame fusion performance

    #LTCFN performance

if __name__ == "__main__":

    y_true = [0,1,2,2,2]
    y_pred = [0,0,2,2,1]
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true,y_pred,target_names = target_names))
    
    
    