# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:33:19 2018

@author: Administrator
"""
from __future__ import division
import torch
import tqdm
import torch.nn.functional as F

import os
import torch.nn as nn
import DataSet_sound
import dense
import torch.cuda as cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import numpy as np

from temp import model_lasted_path,lasted,weights_123

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


epochs=200

cuda=True
batch_size=1024
num_classes=4
lr=0.002
def load_data():
    trainset=DataSet_sound.DataSet()#调用类，生成对象
    loader=DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

    return loader
# for i in loader:
    #  i  一个批次的样本



def train():
    model=dense.TCN(input_size=4,output_size=4, num_channels= [12,12,24,32,48,64] , kernel_size=7, dropout=0.2) #这句话是用之前定义好的类SeismicNet，生成model对象

    if cuda:
        model.cuda()
    traindata=load_data()#数据生成
    len_epoch = len(traindata)
    #testdata=load_data()
    #optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.001)
    #scheduler=MultiStepLR(optimizer,milestones=[20,35,50],gamma=0.3)
    
    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    if model_lasted_path !=0:
        print('The lasted model is:{}'.format(model_lasted_path))
        checkpoint = torch.load(model_lasted_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.eval()
    
    for epoch in range(int(lasted),epochs):
        model.train() #生成
        correct=0
        total=0
        train_loss=0
        
        acc=np.zeros([num_classes])
        subcorrect = np.zeros([num_classes])
        subtotal = np.zeros([num_classes])
        
        traindata=tqdm.tqdm(traindata)
        for ii,(data,label) in enumerate(traindata):
            inputdata=Variable(data)
            target=Variable(label)
#            print(target)
            if cuda:
                inputdata=inputdata.cuda()
                target=target.cuda()
            output=model(inputdata)  #输出 = 对象 (输入)
           # output=torch.log(output)
            total+=target.size(0)
#            print(total)
            pred = output.data.max(1, keepdim=True)[1]
            #print(pred)
            res = pred.eq(target.data.view_as(pred)).cpu().numpy()
#            print(list(res))
            for label_idx in range(len(target)):
                label_single = target[label_idx]
                subcorrect[label_single] += res[label_idx]
                subtotal[label_single] += 1
            #print(subcorrect)
            #print(subtotal)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().numpy()
            #print(correct)
#            loss=F.nll_loss(output,target)
            loss=F.nll_loss(output,target,weight=Variable(torch.Tensor([1,1.076,1.215,1.068]).cuda()))
            train_loss+=loss.cpu().data.numpy()
            #print(train_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss/=len_epoch
        correct/=total
        for i in range(num_classes):
            acc[i] = subcorrect[i]/subtotal[i]
        #print('learning_rate={}'.format(optimizer.param_groups[0]["lr"]))
        #print(time.strftime('%Y.%m.%d.%H.%M.%S',time.localtime(time.time())))
        print('epoch={},train_loss={},correct={},subclassacc={}'.format(epoch,train_loss,correct,acc))
        
        #np.savetxt('correct.txt',correct,delimiter="\n")
        f1 = open("train_correct.txt","a+")
        f2 = open("train_loss.txt","a+")
        correctstr = str(correct)+"\n"
        f1.write(correctstr)
        loss_str = str(train_loss)+"\n"
        f2.write(loss_str)
        f1.close()
        f2.close()
        #testcorrect=test(model,testdata)
        #print('epoch={},testcorrect={}'.format(epoch,testcorrect))
        
#        torch.save(model.state_dict(),os.path.join(weights_123,'small_model_')+repr(epoch)+'.pth') 
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                    os.path.join(weights_123,'z_model')+repr(epoch)+'.pth') 
   
        scheduler.step()





   
train() #先看train.py文件，首先执行的就是这个train()函数  
