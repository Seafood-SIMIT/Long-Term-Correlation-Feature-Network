#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:32:17 2020

@author: seafood
"""
## temporal data 


import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import MediumScaleModel
import UrbanSoundModel
import ReadData
## parament 
num_classes = 3
frame_length = 1024;
inc = 1024;
path=r'/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet'
#semi_path=r'/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/test_semi'
savedir='./semi仿真结果/'

## read the semi model
model_semi=MediumScaleModel.SeismicNet(class_num=5)
model_semi.load_state_dict(torch.load('MediumScaleModel-20200703_210118_Epoch093[81.626%].pth',map_location='cpu'))
model_semi.eval()

## read the aco model
model_aco=UrbanSoundModel.UrbanSound8KModel()
model_aco.load_state_dict(torch.load('../outputs/model_aco.pth',map_location='cpu'))
model_aco.eval()

## read the data
true_label,datas_aco,datas_semi,num_length = ReadData.get_data_infile(path,frame_length,inc)

# reright the data
def resetTheBpa(out_BPA,acc):
    return out_BPA*acc+0.5*(1-out_BPA)*(1-acc)

def trustyCalculate(trusty_yangben,out_predlabel,acc):
    if out_predlabel == trusty_yangben[4] and out_predlabel == trusty_yangben[3]:
        acc = 0.99;
    #elif out_predlabel == trusty_yangben[4] or out_predlabel == trusty_yangben[3]:
        #acc = acc;
    else:
        count=0;
        for i in range(len(trusty_yangben)):
            if out_predlabel == trusty_yangben[i]:
                count+=1;
        if count<3:
            #acc = acc * (count/len(trusty_yangben));
            acc = 0.4;
        else:
            acc = 0.7;
        #acc = acc;
    return acc
count_wrong_aco_be4_fusion = 0;
count_wrong_semi_be4_fusion = 0;
count_wrong_fused_after_fusion = 0;
trusty_aco_yangben=[0,0,0,0,0];
trusty_semi_yangben=[0,0,0,0,0];
labels_pred = np.zeros((num_length,3))
#test
for i in range(num_length):
    data_aco = datas_aco[i*inc:i*inc+frame_length]
    data_semi = datas_semi[i*inc:i*inc+frame_length]
    
    #yuchuli
    mfccs = UrbanSoundModel.yuchuli_aco(data_aco)
    features_semi = MediumScaleModel.yuchuli_semi(data_semi)
    
    #throw into the models
    out_aco_BPA = model_aco(mfccs).data.numpy()
    out_aco_BPA = out_aco_BPA/np.sum(out_aco_BPA)
    out_aco_predlabel = np.argmax(out_aco_BPA)
    labels_pred[i,0] = out_aco_predlabel;
    #tmp_
    out_semi_BPA = model_semi(features_semi).data.numpy()
    #out_semi_BPA = [out_semi_BPA_tmp[0],out_semi_BPA_tmp[2],out_semi_BPA_tmp[3]]
    out_semi_BPA = out_semi_BPA[:,[0,2,3]]
    out_semi_BPA = out_semi_BPA/np.sum(out_semi_BPA)
    out_semi_predlabel = np.argmax(out_semi_BPA)
    labels_pred[i,1] = out_semi_predlabel;
    #ans before Fusion
    if out_aco_predlabel!=true_label:
        count_wrong_aco_be4_fusion+=1
    if out_semi_predlabel!=true_label:
        count_wrong_semi_be4_fusion+=1
    
    #reset the BPA
    acc_aco = 0.72;
    acc_semi =0.84;
    #set the trusty
    if i < 5:
        trusty_aco_yangben[i]=out_aco_predlabel;
        trusty_semi_yangben[i]=out_semi_predlabel;
    else:
        acc_aco=trustyCalculate(trusty_aco_yangben,out_aco_predlabel,acc_aco)
        acc_semi=trustyCalculate(trusty_semi_yangben,out_semi_predlabel,acc_semi)
        #update yangben
        trusty_aco_yangben[0:3]=trusty_aco_yangben[1:4]
        trusty_aco_yangben[4]=out_aco_predlabel;
        trusty_semi_yangben[0:3]=trusty_semi_yangben[1:4]
        trusty_semi_yangben[4]=out_semi_predlabel;
        
    if 1:
        out_aco_BPA = resetTheBpa(out_aco_BPA, acc=acc_aco) 
        out_semi_BPA = resetTheBpa(out_semi_BPA, acc=acc_semi) 
    
    #Fusion
    con_matrix = np.zeros((3,3))+1-np.diag((1,1,1))
    k = np.dot(np.dot(out_aco_BPA, con_matrix),out_semi_BPA.T)
    con_mass = 1/(1-k)
    mass = con_mass * np.multiply(out_aco_BPA, out_semi_BPA)
    fuse_ans = mass.argmax()
    labels_pred[i,2] = fuse_ans;
    #ans after Fusion
    if fuse_ans != true_label:
        count_wrong_fused_after_fusion+=1;
        
#summer and printf
print('aco  accurate before fusion is ',(1-count_wrong_aco_be4_fusion/num_length))
print('semi accurate before fusion is ',(1-count_wrong_semi_be4_fusion/num_length))
print('fuse accurate after  fusion is ',(1-count_wrong_fused_after_fusion/num_length))