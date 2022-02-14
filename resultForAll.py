#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:44:49 2022

@author: sunlin
"""
from modelGenerator import modelGenerator, modelCompareGenerate
from utils.readData import normaLization,readDataFilelists,readDataInFile,preProcessAco
from classificationPerformance import resultClassifier, classificationPerform,resultClassifierCompareWithOther
from tqdm import tqdm


import os
import sys
sys.path.append('/Users/sunlin/Documents/workdir/vehicleClassification')
import numpy as np

import torch


model_aco, model_seis, model_lstm = modelGenerator()
model_aco_com, model_seis_com = modelCompareGenerate()
data_dir = '/Volumes/T7/DataBase/aco_seis_Dataset'
frame_length = 1024
aco_dir, seis_dir,aco_filelist, seis_filelist = readDataFilelists(data_dir)
acc = 0
count_file, count_frame=0,0
pred = []
labels = []
#for index in tqdm(range(len(aco_filelist))):
predict_aco, predict_seis,label_per = [],[],[]
for index in tqdm(range(4)):
    flag,label,origin_signal_aco,origin_signal_seis = readDataInFile(
                                                        aco_dir, 
                                                        seis_dir,
                                                        aco_filelist, 
                                                        seis_filelist,
                                                        index)
    if flag == False:
        continue
    
    for i in range(len(origin_signal_aco)//frame_length-1):
        count_frame+=1
        #读数据
        frame_data_aco = normaLization(origin_signal_aco[i*frame_length:frame_length*(i+1)])
        frame_data_seis = normaLization(origin_signal_seis[i*frame_length:frame_length*(i+1)])
        #预测
        frame_predict_aco, frame_predict_seis=classificationPerform(
                                            model_aco,
                                            model_seis, 
                                            frame_data_aco,
                                            frame_data_seis)
        predict_aco.append(frame_predict_aco)
        predict_seis.append(frame_predict_seis)
        label_per.append(label)
        frame_predict_compare_aco, frame_predict_compare_seis = classificationPerform(
                                                                model_aco_com,
                                                                model_seis_com, 
                                                                preProcessAco(frame_data_aco),
                                                                frame_data_seis)
    
    
    #print(fuse_feature.shape,'1')
    
    #acc+= 1 if np.argmax(predict_lstm[-1]) == label else 0
    #print('[%d]\tlabel:%d\tpredict:%d\n'%(index+1,label,int(np.argmax(predict_lstm))))
    #pred.append(np.argmax(predict_lstm))
    labels.append(label)
    count_file+=1
acc_aco,acc_seis=resultClassifier(
                np.argmax(np.array(predict_aco).reshape(len(pred),3),axis=1),
                np.argmax(np.array(predict_seis).reshape(len(pred),3),axis=1),
                label_per,
                frame_length)
resultClassifierCompareWithOther(predict_aco,predict_seis,label_per)
#resultLSTM()
#resultFusionCompareWithOther()
#print(cm)
#print(acc/count_file)