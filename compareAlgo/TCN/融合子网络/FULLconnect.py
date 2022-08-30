# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:46:05 2018

@author: Administrator
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class FULLconnect(nn.Module):
    def __init__(self, num_classes=4):
        super(FULLconnect, self).__init__()


        self.layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(32, num_classes))

    def forward(self, x):

        x = self.layer2(x)
        x = self.layer3(x)

        return F.log_softmax(x,dim = 1)
