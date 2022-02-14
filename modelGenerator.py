# -*- coding: utf-8 -*-
import sys
import torch
sys.path.append('/Users/sunlin/Documents/workdir/vehicleClassification')
from results.models.aco_model.acoClassifier import ACOClassifier
from results.models.seis_model.seisClassifier import SEISClassifier
from compareAlgo.MFCC.MediumScaleModel import SeismicNet
from compareAlgo.MFCC.UrbanSoundModel import UrbanSound8KModel
from deepdsTrainer.DeepDS import DeepDS
from deepdsTrainer.others import HParam
def modelGenerator():
    hp_lstm = HParam('/Users/sunlin/Documents/workdir/vehicleClassification/deepdsTrainer/config.yaml')
    #加载声音模型
    model_aco = ACOClassifier()
    chckt = torch.load('/Users/sunlin/Documents/workdir/vehicleClassification/results/models/aco_model/model_aco.pt',map_location=torch.device('cpu'))
    model_aco.load_state_dict(chckt['model'])
    model_aco.eval()
    #加载震动模型
    model_seis = SEISClassifier()
    chckt = torch.load('/Users/sunlin/Documents/workdir/vehicleClassification/results/models/seis_model/model_seis.pt',map_location=torch.device('cpu'))
    model_seis.load_state_dict(chckt['model'])
    model_seis.eval()
    #加载LSTM模型
    model_lstm = DeepDS(hp_lstm)
    chckt = torch.load('/Users/sunlin/Documents/workdir/vehicleClassification/results/models/lstm_model/DeepDStest39step[1.00].pt',map_location=torch.device('cpu'))
    model_lstm.load_state_dict(chckt['model'])
    model_lstm.eval()
    return model_aco, model_seis, model_lstm

def modelCompareGenerate():
    model_seis=SeismicNet(class_num=5)
    model_seis.load_state_dict(torch.load('/Users/sunlin/Documents/workdir/vehicleClassification/results/compareAlgo/MFCC/model_seis.pth',map_location='cpu'))
    model_seis.eval()
    
    model_aco=UrbanSound8KModel()
    model_aco.load_state_dict(torch.load('/Users/sunlin/Documents/workdir/vehicleClassification/results/compareAlgo/MFCC/model_aco.pth',map_location='cpu'))
    model_aco.eval()
    return model_aco, model_seis
