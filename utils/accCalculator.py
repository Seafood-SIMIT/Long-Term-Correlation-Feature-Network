# -*- coding: utf-8 -*-

'''
@20220210
@1.0
@Seafood
@计算各种准确率
'''
import numpy as np
def accCalculateFrame(pred,label,frame_length):
    #pred.shape:frame_number,3
    return np.sum(np.argmax(np.array(pred)) == np.array(label))/frame_length