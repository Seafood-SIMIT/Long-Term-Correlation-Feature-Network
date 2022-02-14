# -*- coding: utf-8 -*-
'''
20220210
@Seafood
@混淆矩阵生成
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
def confusionMatrixGenerator(pred, labels):
    #print(np.argmax(np.array(pred),axis=1))
    cm = confusion_matrix(pred, np.array(labels))
    return cm


def plot_confusion_matrix(cm,number_fig,name):
    fig = plt.figure(number_fig)
    labels_name = ['smallWheel','largeWheel','Track','Unknow']
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(name)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):    #第几行
        for second_index in range(len(cm[first_index])):    #第几列
            plt.text(first_index, second_index, cm[first_index][second_index])
    plt.show()
    plt.savefig(os.path.join('output',str(number_fig)+name+'.png'),format='png')