#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:44:49 2022

@author: sunlin
"""
from modelGenerator import modelGenerator, modelCompareGenerate
from utils.readData import normaLization,readDataFilelists,readDataInFile
from utils.addNoise import addNoise
from utils.classificationPerformance import classificationPerform
from utils.fusionPerformance import ltcfnPerform,fusionProject,votePerform
from tqdm import tqdm
import compareAlgo

import numpy as np


system_name = 'macos'
model_aco, model_seis, model_lstm = modelGenerator(system_name)
#model_aco_mfcc, model_seis_medium,model_aco_wavelet, model_seis_wavelet = modelCompareGenerate()

data_dir = '/Volumes/T7/DataBase/aco_seis_Dataset/validset'
#data_dir = 'data'

frame_length = 1024
aco_dir, seis_dir,aco_filelist, seis_filelist = readDataFilelists(data_dir)
acc = 0
count_file, count_frame=0,0
pred = []
labels = []
#for index in tqdm(range(len(aco_filelist))):
predict_lstm = []

def featureExtractor(signal_aco,signal_seis,a_snr,a_label):

    predict_aco, predict_seis,label_per = [],[],[]
    frame_feature_lstm = []
    for i in range(len(signal_aco)//frame_length-1):
        #读数据
        frame_data_aco = normaLization(signal_aco[i*frame_length:frame_length*(i+1)])
        frame_data_seis = normaLization(signal_seis[i*frame_length:frame_length*(i+1)])
        #预测
        if a_snr != -1:
            frame_data_aco = addNoise(frame_data_aco,a_snr)
            frame_data_seis = addNoise(frame_data_seis,a_snr)
        frame_predict_aco, frame_predict_seis=classificationPerform(
                                            model_aco,
                                            model_seis,
                                            frame_data_aco,
                                            frame_data_seis)
        predict_aco.append(frame_predict_aco)
        predict_seis.append(frame_predict_seis)
        label_per.append(a_label)
        #print(np.concatenate([frame_predict_aco,frame_predict_seis]))
        frame_feature_lstm.append(np.concatenate([frame_predict_aco,frame_predict_seis]))
    return predict_aco, predict_seis, label_per, frame_feature_lstm

signal_aco_total = []
signal_seis_total = []
labels = []
for index in tqdm(range(len(aco_filelist))):
    flag,label,origin_signal_aco,origin_signal_seis = readDataInFile(
                                                        aco_dir,
                                                        seis_dir,
                                                        aco_filelist,
                                                        seis_filelist,
                                                        index)
    if flag == False:
        continue
    signal_aco_total.append(origin_signal_aco)
    signal_seis_total.append(origin_signal_seis)
    labels.append(label)
    count_file+=1

snr_set = [70,50,20,10,1,0.5]
for a_snr in snr_set:
    print("=======================================SNR: ", a_snr)
    predict_aco_total,predict_seis_total = [],[]
    label_per_total = []
    predict_lstm,predict_vote_aco,predict_vote_seis = [],[],[]
    print('perform LTCFN')
    for a_data in tqdm(range(count_file)):
        predict_aco, predict_seis, label_per ,frame_feature_lstm= featureExtractor(signal_aco_total[a_data],signal_seis_total[a_data],a_snr,labels[a_data])

        #预测
        #print(fuse_feature.shape,'1')
        predict_aco_total = predict_aco_total+predict_aco
        predict_seis_total = predict_seis_total+predict_seis
        label_per_total = label_per_total+label_per
        #print(frame_feature_lstm)
        predict_lstm.append(ltcfnPerform(frame_feature_lstm,model_lstm))
        predict_vote_aco.append( votePerform(predict_aco))
        predict_vote_seis.append( votePerform(predict_seis))
        #acc+= 1 if np.argmax(predict_lstm[-1]) == label else 0
        #print('[%d]\tlabel:%d\tpredict:%d\n'%(index+1,label,int(np.argmax(predict_lstm))))
        #pred.append(np.argmax(predict_lstm))
        #------------------------TCN
    predict_tcn = []
    print('perform DS')
    predict_ds = compareAlgo.DS.classicDSFusion(predict_aco_total,predict_seis_total,label_per_total)
        #-------------------------Hu2014
        #acc_Hu_2014 = compareAlgo.DS.hu2014DSFusion(np.array(pba_aco),np.array(pba_seis),label)
        #-------------------------Xiao2020
    print("perform Improved DS")
    predict_ds_xiao = compareAlgo.DS.xiao2020DSFusion(predict_aco_total,predict_seis_total,label_per_total)
        #------------------------


    print("Perform Down, Acc Counting")
    target_names = ['light wheel vehicle', 'heavy wheel vehicle ','tracked vehicle']
        # proposed method

    fusionProject(labels,label_per_total, predict_lstm, predict_tcn, predict_ds,predict_ds_xiao, predict_vote_aco,predict_vote_seis,
                    target_names)