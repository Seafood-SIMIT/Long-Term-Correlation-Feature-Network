#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:44:49 2022

@author: sunlin
"""
from utils.accCalculator import theOutputfromArray
from modelGenerator import modelGenerator, modelCompareGenerate
from utils.readData import normaLization,readDataFilelists,readDataInFile,preProcessAco,waveletPreprocess
from classificationPerformance import resultClassifier, classificationPerform,resultClassifierCompareWithOther,classificationProject
from fusionPerformance import resultFusionCompareWithOther, ltcfnPerform,resultFusion
from tqdm import tqdm


import os
import sys
sys.path.append('/Users/sunlin/Documents/workdir/vehicleClassification')
import numpy as np

import torch

system_name = 'macos'
model_aco, model_seis, model_lstm = modelGenerator(system_name)
model_aco_mfcc, model_seis_medium,model_aco_wavelet, model_seis_wavelet = modelCompareGenerate()

data_dir = '/Volumes/T7/DataBase/aco_seis_Dataset/validset'

frame_length = 1024
aco_dir, seis_dir,aco_filelist, seis_filelist = readDataFilelists(data_dir)
acc = 0
count_file, count_frame=0,0
pred = []
labels = []
#for index in tqdm(range(len(aco_filelist))):
predict_aco, predict_seis,label_per = [],[],[]
predict_lstm = []
predict_aco_mfcc,predict_seis_medium,predict_aco_wavelet,predict_seis_wavelet=[],[],[],[]
for index in tqdm(range(len(aco_filelist))):
    flag,label,origin_signal_aco,origin_signal_seis = readDataInFile(
                                                        aco_dir, 
                                                        seis_dir,
                                                        aco_filelist, 
                                                        seis_filelist,
                                                        index)
    if flag == False:
        continue
    frame_feature_lstm=[]
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
        frame_predict_aco_mfcc, frame_predict_seis_medium = classificationPerform(
                                                                model_aco_mfcc,
                                                                model_seis_medium, 
                                                                preProcessAco(frame_data_aco),
                                                                frame_data_seis)
        predict_aco_mfcc.append(frame_predict_aco_mfcc)
        predict_seis_medium.append(frame_predict_seis_medium)
        frame_predict_aco_wavelet, frame_predict_seis_wavelet = classificationPerform(
                                                                model_aco_wavelet,
                                                                model_seis_wavelet, 
                                                                waveletPreprocess(frame_data_aco),
                                                                waveletPreprocess(frame_data_seis))
        predict_aco_wavelet.append(frame_predict_aco_wavelet)
        predict_seis_wavelet.append(frame_predict_seis_wavelet)

        #预测
        frame_feature_lstm.append(np.concatenate([frame_predict_aco,frame_predict_seis]))
    #print(fuse_feature.shape,'1')
    
    

    predict_lstm.append(ltcfnPerform(frame_feature_lstm,model_lstm))
    #acc+= 1 if np.argmax(predict_lstm[-1]) == label else 0
    #print('[%d]\tlabel:%d\tpredict:%d\n'%(index+1,label,int(np.argmax(predict_lstm))))
    #pred.append(np.argmax(predict_lstm))
    labels.append(label)
    count_file+=1
print("Perform Down, Acc Counting")
target_names = ['light wheel vehicle', 'heavy wheel vehicle ','tracked vehicle']
# proposed method
resultClassifier(predict_aco,predict_seis, predict_aco_mfcc, predict_seis_medium, predict_aco_wavelet,
                                    predict_seis_wavelet, label_per, frame_length, label_per)
classificationProject('proposed',labels,label_per,predict_aco,
                        predict_aco_mfcc,predict_aco_mfcc,predict_aco_wavelet,
                        predict_seis,predict_seis_medium,predict_seis_wavelet,
                        predict_lstm, target_names)


fusionProject(labels, predict_aco, predict_seis)