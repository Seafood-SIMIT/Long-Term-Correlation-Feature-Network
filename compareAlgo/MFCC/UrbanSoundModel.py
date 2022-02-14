#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:17:51 2020

@author: seafood
"""

import torch
import math
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import librosa.display
input_size=20
output_size=3

def yuchuli_aco(data_aco):
    data_aco=data_aco-np.mean(data_aco)
    mfccs = librosa.feature.mfcc(data_aco, sr = 22050, S=None, norm = 'ortho', n_mfcc=input_size)
    return torch.tensor(np.mean(mfccs.T,axis = 0))

def accuracy(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class AudioClassificationBase(nn.Module):
    def training_step(self,batch):
        images,labels = batch
        out = self(images)              #Generate prediction
        #print(out)
        loss = F.cross_entropy(out,labels)      #Calculate loss
        return loss
    
    def get_score_step(self, batch):
        inputs= batch
        out = self(inputs)
        return out
    
    def validation_step(self,batch):
        inputs , labels = batch
        out = self.get_score_step(batch)              #Generate predictions
        loss = F.cross_entropy(out,labels)  #Calculate loss
        acc = accuracy(out, labels)         #Calculate accuary
        return {'val_loss':loss.detach(),'val_acc':acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]     #combine loss
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]        #combine acuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
    
class UrbanSound8KModel(AudioClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.Linear(64, output_size),
            nn.Sigmoid()
            )
    def forward(self, xb):
        xb = xb.clone().detach().float()
        #xb = torch.tensor(xb, dtype=torch.float32)
        return self.network(xb)