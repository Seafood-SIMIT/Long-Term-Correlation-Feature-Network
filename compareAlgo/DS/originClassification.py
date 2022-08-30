import torch
import os
from pathlib import Path
import numpy as np
def originClassify(logger,data_aco,data_seis,model_aco,model_seis,frame_length):
    frame_number = len(data_aco)//frame_length
    predicts_aco,predicts_seis=[],[]
    for i in range(frame_number-1):
        frame_data_aco = data_aco[i*frame_length:(i+1)*frame_length]
        frame_data_seis = data_seis[i*frame_length:(i+1)*frame_length]

        frame_data_aco = (frame_data_aco-min(frame_data_aco))/(max(frame_data_aco)-min(frame_data_aco))
        frame_data_seis = (frame_data_seis-min(frame_data_seis))/(max(frame_data_seis)-min(frame_data_seis))

        predicts_aco.append(model_aco(torch.tensor(frame_data_aco.reshape(1,-1),dtype=torch.float)).data.numpy())
        predicts_seis.append(model_seis(torch.tensor(frame_data_seis.reshape(1,-1),dtype=torch.float)).data.numpy())
    return predicts_aco,predicts_seis,frame_number
def labelRead(file):
    if 'smallwheel' in file:
        label=0
    elif 'largewheel' in file:
        label=1
    elif 'track' in file:
        label=2
    else:
        label=3
    return label

def readData(aco_data_dir,seis_data_dir,aco_file):
    seis_file = aco_file.replace('[A]','[S]')
    now_aco_file = aco_data_dir+'/'+aco_file
    now_seis_file = seis_data_dir+'/'+seis_file
    check = Path(now_seis_file)
    data_aco=None
    data_seis=None
    label=None
    flag=0
    if check.exists():
        flag=1
        data_aco = np.loadtxt(now_aco_file)[:,0]
        data_aco = data_aco[::8]
        data_seis = np.loadtxt(now_seis_file)[::8]
        label=labelRead(aco_file)
    return flag,data_aco,data_seis,label