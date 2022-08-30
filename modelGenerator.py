# -*- coding: utf-8 -*-
from itertools import count
from os import system
from statistics import mode
import sys
from compareAlgo import *
import torch
import torchsummary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def modelSize(model):
    print("model_size:  ",count_parameters(model))

def modelGenerator(system_name):
    if system_name == "wsl":
        model_file = "/home/seafood/workdir/vehicleClassification/results/models/"
    else:
        model_file = "/Users/sunlin/Documents/workdir/vehicleClassification/results/models/"
    #加载声音模型
    model_aco = ACOClassifierLSTM(1,128,2)
    chckt = torch.load('models/aco_model/acoAlexNet0322-1705moreepoch_checkout_86step[71.56].pt',map_location=torch.device('cpu'))
    model_aco.load_state_dict(chckt['model'])
    model_aco.eval()
    modelSize(model_aco)
    #加载震动模型
    model_seis = SEISClassifierLSTM(1,128,2)
    chckt = torch.load('models/seis_model/seisAlexNet0322-1134_checkout_38step[83.35].pt',map_location=torch.device('cpu'))
    model_seis.load_state_dict(chckt['model'])
    model_seis.eval()
    modelSize(model_seis)
    #加载LSTM模型
    model_lstm = DeepDS(input_size = 6,hidden_size=16)
    chckt = torch.load(model_file+'lstm_model/DeepDSltcfn0323-1116form19step[0.97].pt',map_location=torch.device('cpu'))
    model_lstm.load_state_dict(chckt['model'])
    model_lstm.eval()
    modelSize(model_lstm)
    return model_aco, model_seis, model_lstm

def modelCompareGenerate():
    model_seis=SeismicNet()
    model_seis.load_state_dict(torch.load('models/medium_model/seismicNet0322-2252form_checkout_70step[78.32].pt',map_location='cpu')['model'])
    model_seis.eval()
    modelSize(model_seis)
    
    model_aco=UrbanSoundModel()
    model_aco.load_state_dict(torch.load('models/mfcc_model/acoMFCC0322-2257form_checkout_26step[80.30].pt',map_location='cpu')['model'])
    model_aco.eval()
    modelSize(model_aco)
    
    model_aco_wavalet = WaveletAcoModel()
    model_aco_wavalet.load_state_dict(torch.load('models/wavelet_aco_model/waveletAco0322-1807_checkout_196step[61.51].pt',map_location='cpu')['model'])
    model_aco_wavalet.eval()
    modelSize(model_aco_wavalet)
    
    model_seis_wavalet = WaveletSeisModel()
    model_seis_wavalet.load_state_dict(torch.load('models/wavelet_seis_model/waveletSeis0322-1821_checkout_196step[80.13].pt',map_location='cpu')['model'])
    model_seis_wavalet.eval()
    modelSize(model_seis_wavalet)
    return model_aco, model_seis,model_aco_wavalet, model_seis_wavalet
