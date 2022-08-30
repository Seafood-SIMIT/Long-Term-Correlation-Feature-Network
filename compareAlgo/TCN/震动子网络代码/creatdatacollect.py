# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:02:41 2018

@author: Administrator
"""

##对原始数据分帧保存

import numpy as np
import math
import os

#path=r'D:\震动数据\按要求截取数据\老系统数据'
path='/home/fanyuying/two_streams数据库/训练A'
framelength=1024
inc=256
numlargewheel=0
numtrack=0
numsmallwheel=0
numperson = 0


def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files
files=list_all_files(path)  #得到path下所有的目录
for j in range(len(files)):
    if files[j].endswith('.txt'):
        data=np.loadtxt(files[j])
        #data=data[2000:len(data)-5000]
       # data=data[:,1]
        data=data[::8]
        num=math.floor((len(data)-framelength+inc)/inc)  #下取整，num指帧数
       
        
        if files[j].lower().find('largewheel')!=-1:
             #data=data[::8]
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='largewheel'
             for i in range(int(num)):
                 waves=data[i*inc:i*inc+framelength]
                 numlargewheel+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numperson)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams数据库/测试集A_分帧',target+'_'+str(numlargewheel)+'.txt')
                 np.savetxt(savepath,waves)
#                 
                 
        if files[j].lower().find('smallwheel')!=-1:
             #data=data[::8]
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='smallwheel'
             for i in range(int(num)):
                 waves=data[i*inc:i*inc+framelength]
                 numsmallwheel+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numwheel)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams数据库/测试集A_分帧',target+'_'+str(numsmallwheel)+'.txt')
                 np.savetxt(savepath,waves)
                 
                 
        if files[j].lower().find('track')!=-1:
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='track'
             for i in range(int(num)):
                 waves=data[i*inc:i*inc+framelength]
                 numtrack+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numtrack)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams数据库/测试集A_分帧',target+'_'+str(numtrack)+'.txt')
                 np.savetxt(savepath,waves)
         

        if files[j].lower().find('person')!=-1:
            num=math.floor((len(data)-framelength+inc)/inc)
            target='person'
            for i in range(int(num)):
                waves=data[i*inc:i*inc+framelength]
                numperson+=1
                #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numheightcar)+'.txt')
                savepath=os.path.join('/home/fanyuying/two_streams数据库/测试集A_分帧',target+'_'+str(numperson)+'.txt')
                np.savetxt(savepath,waves)
