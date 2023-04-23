# -*- coding: utf-8 -*-
from itertools import count
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
    elif system_name == "macos":
        model_file = "/Users/sunlin/Documents/workdir/vehicleClassification/vehicleClassification/results/models/"
        #model_file = "/Users/sunlin/Documents/workdir/vehicleClassification/results/models/"
    #加载声音模型
    model_aco = ACOAlexNetBiLSTM(3)
    chckt = torch.load('models/aco_model/aco20220917alexNetcheckout78step[0.76].pt',map_location=torch.device('cpu'))
    model_aco.load_state_dict(chckt['model'])
    model_aco.eval()
    modelSize(model_aco)
    #加载震动模型
    model_seis = SEISAlexNetBiLSTM(3)
    chckt = torch.load('models/seis_model/seis20220919AlexNetcheckout78step[0.77].pt',map_location=torch.device('cpu'))
    model_seis.load_state_dict(chckt['model'])
    model_seis.eval()
    modelSize(model_seis)
    #加载LSTM模型
    model_lstm = DeepDS(input_size = 6,hidden_size=16)
    chckt = torch.load(model_file+'lstm_model/lstm.pt',map_location=torch.device('cpu'))
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
