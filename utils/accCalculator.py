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
    print(np.sum(pred == np.array(label)))
    return np.sum(pred == np.array(label))/frame_length

def theOutputfromArray(bpa):
    return np.argmax(np.array(bpa).reshape(-1,3),axis=1)