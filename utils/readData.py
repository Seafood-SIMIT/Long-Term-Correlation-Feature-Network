# -*- coding: utf-8 -*-
'''
@20220210
@1.0
@Seafood
@读取原始txt音频文件
'''
import os
import pywt
import numpy as np
import librosa

def waveletPreprocess(data):
    wp = pywt.WaveletPacket(data=data, 
                            wavelet='db3',
                            mode='symmetric',
                            maxlevel=8)
    re = []
    for i in [node.path for node in wp.get_level(8, 'freq')]:
        re.append(wp[i].data)
    #能量特征
    energy=[]
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None), 2))
    energy = np.array(energy[0:64])
    energy = energy/np.sum(energy)
    #energy = energy/energy.sum
    #energy = energy/np.sqrt(np.dot(energy,energy.T))
    return energy    
def preProcessAco(data_aco,input_size=32):
    data_aco=data_aco-np.mean(data_aco)
    mfccs = librosa.feature.mfcc(y=data_aco, sr = 22050, S=None, norm = 'ortho', n_mfcc=input_size)
    return np.mean(mfccs.T,axis = 0)


def normaLization(data):
    return (data-min(data))/(max(data)-min(data))

def readDataFilelists(data_dir):
    
    frame_length=1024
    aco_dir = os.path.join(data_dir,'aco_01')
    seis_dir = os.path.join(data_dir,'seis_01')
    
    aco_filelist = os.listdir(aco_dir)
    seis_filelist = os.listdir(seis_dir)
    assert len(aco_filelist) !=0, \
            "No training file found"
    return aco_dir, seis_dir, aco_filelist, seis_filelist

def readDataInFile(aco_dir, seis_dir,aco_filelist, seis_filelist,index):
    aco_file = aco_filelist[index]
    seis_file = aco_file.replace('[A]', '[S]')
    aco_file = os.path.join(aco_dir,aco_file)
    seis_file = os.path.join(seis_dir,seis_file)
    if os.path.isfile(seis_file):
        if 'smallwheel' in aco_file:
            label=0
        elif 'largewheel' in aco_file:
            label=1
        elif 'track' in aco_file:
            label=2
        else:
            label=3
        
        origin_signal_aco = np.loadtxt(aco_file)
        origin_signal_aco = origin_signal_aco[:,0].reshape(-1)
        origin_signal_aco = origin_signal_aco[::8]
        origin_signal_seis = np.loadtxt(seis_file)
        origin_signal_seis = origin_signal_seis.reshape(-1)
        origin_signal_seis = origin_signal_seis[::8]
        return True, label,origin_signal_aco,origin_signal_seis
    else:
        return False,False,False,False