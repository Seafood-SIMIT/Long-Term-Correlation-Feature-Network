# -*- coding:UTF-8 -*
##对原始数据分帧保存

import numpy as np
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#path=r'D:\震动数据\按要求截取数据\老系统数据'
path='/home/fanyuying/two_streams_data/未降噪测试'
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
    prefix = files[j].split('[')[0]
    ednfix = files[j].split('[')[1]
    if ednfix[0] == 'S': #震动
        dataz=np.loadtxt(files[j],encoding="GBK")
        print(dataz.shape)
        dataz=dataz[3000:len(dataz)-3000]
       # data=data[:,1]
        dataz=dataz[::8]
        S_name = prefix + '[A' + ednfix[1:]
        datas = np.loadtxt(S_name)
        datas=datas[3000:len(datas)-3000,:]
        print(datas.shape)
        datas = datas[::8]
        
        num=(len(dataz)-framelength+inc)//inc  
        print(num)      
        if files[j].lower().find('largewheel')!=-1:
             #data=data[::8]
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='largewheel'
             for i in range(int(num)):
                 wavez=dataz[i*inc:i*inc+framelength]
                 waves = datas[i*inc:i*inc+framelength]
                 numlargewheel+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numperson)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams_data/未降噪训练震动分帧',target+'_'+str(numlargewheel)+'.txt')
                 np.savetxt(savepath,wavez) 
                 savepaths=os.path.join('/home/fanyuying/two_streams_data/未降噪训练声音分帧',target+'_'+str(numlargewheel)+'.txt')
                 np.savetxt(savepaths,waves)
        if files[j].lower().find('smallwheel')!=-1:
             #data=data[::8]
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='smallwheel'
             for i in range(int(num)):
                 wavez=dataz[i*inc:i*inc+framelength]
                 waves = datas[i*inc:i*inc+framelength]
                 numsmallwheel+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numwheel)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams_data/未降噪训练震动分帧',target+'_'+str(numsmallwheel)+'.txt')
                 np.savetxt(savepath,wavez) 
                 savepaths=os.path.join('/home/fanyuying/two_streams_data/未降噪训练声音分帧',target+'_'+str(numsmallwheel)+'.txt')
                 np.savetxt(savepaths,waves)
        if files[j].lower().find('track')!=-1:
             #num=math.floor((len(data)-framelength+inc)/inc)
             target='track'
             for i in range(int(num)):
                 wavez=dataz[i*inc:i*inc+framelength]
                 waves = datas[i*inc:i*inc+framelength]
                 numtrack+=1
                 #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numtrack)+'.txt')
                 savepath=os.path.join('/home/fanyuying/two_streams_data/未降噪训练震动分帧',target+'_'+str(numtrack)+'.txt')
                 np.savetxt(savepath,wavez)
                 savepaths=os.path.join('/home/fanyuying/two_streams_data/未降噪训练声音分帧',target+'_'+str(numtrack)+'.txt')
                 np.savetxt(savepaths,waves)
        if files[j].lower().find('person')!=-1:
            target='person'
            for i in range(int(num)):
                wavez=dataz[i*inc:i*inc+framelength]
                waves = datas[i*inc:i*inc+framelength]
                numperson+=1
                #savepath=os.path.join(r'D:\微系统所\mydl\traindata\data(四类randomcrop)\老系统',target+'_'+str(numheightcar)+'.txt')
                savepath=os.path.join('/home/fanyuying/two_streams_data/未降噪训练震动分帧',target+'_'+str(numperson)+'.txt')
                np.savetxt(savepath,wavez)
                savepaths=os.path.join('/home/fanyuying/two_streams_data/未降噪训练声音分帧',target+'_'+str(numperson)+'.txt')
                np.savetxt(savepaths,waves)
      
       
