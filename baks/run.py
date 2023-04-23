from aco_model.acoClassifier import ACOClassifier
from seis_model.seisClassifier import SEISClassifier
from utils.originClassification import readData
from utils.originClassification import originClassify
from utils.analise import originAccuracy,Result
from utils.classicDS import classicDSFusion
from utils.hu2014DS import hu2014DSFusion
from utils.xiao2020DS import xiao2020DSFusion
import os
import time
import torch
import pickle
import logging
import numpy as np
from tqdm import tqdm
log_dir = 'logs'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, '%d.log' % (time.time()))),
        logging.StreamHandler()
        ]
    )
logger = logging.getLogger()



frame_length = 1024
#加载声音模型
model_aco = ACOClassifier()
chckt = torch.load('aco_model/model_aco.pt',map_location=torch.device('cpu'))
model_aco.load_state_dict(chckt['model'])
model_aco.eval()
#加载震动模型
model_seis = SEISClassifier()
chckt = torch.load('seis_model/model_seis.pt',map_location=torch.device('cpu'))
model_seis.load_state_dict(chckt['model'])
model_seis.eval()

#读取时序数据
#aco_data_dir = '/Volumes/UNTITLED/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/aco_01'
#seis_data_dir = '/Volumes/UNTITLED/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/seis_01'

aco_data_dir = r'D:\Database\Acoustic-and-seismic-synchronous-signal\20200805ASSSFromLYan/aco_01'
seis_data_dir = r"D:\Database\Acoustic-and-seismic-synchronous-signal\20200805ASSSFromLYan/seis_01"
save_file = 'output/result.pkl'
aco_filelist = os.listdir(aco_data_dir)
result_all=[]
for i in tqdm(range(len(aco_filelist))):
    result=Result()
    aco_file = aco_filelist[i]
    if aco_file.startswith('.'):
        continue
    
    flag,data_aco,data_seis,label= readData(aco_data_dir,seis_data_dir,aco_file)
    if flag == 0:
        continue
    result.filename.append(aco_file)
    logger.info("Filename: %s, withlabel: %d" % (aco_file,label))
    pba_aco,pba_seis,frame_number = originClassify(logger,data_aco,data_seis,model_aco,model_seis,frame_length)
    #logger.info("")
    result.pba_origin_aco=pba_aco
    result.pba_origin_seis=pba_seis
    result.frame_number=frame_number
    acc_ori_aco, acc_ori_seis = originAccuracy(logger,pba_aco,pba_seis,label)
    result.acc_origin_aco=acc_ori_aco
    result.acc_origin_seis=acc_ori_seis
    result.label=label
    logger.info("Origin Accuracy: aco_%.2f, seis_%.2f"%(acc_ori_aco, acc_ori_seis))
    #--------------------------Classic DS Theory
    acc_classic_DS = classicDSFusion(np.array(pba_aco),np.array(pba_seis),label)
    logger.info("Classic D-S Theory: acc: %.2f"%(acc_classic_DS))
    result.acc_classic_DS=acc_classic_DS
    #-------------------------Hu2014
    acc_Hu_2014 = hu2014DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    logger.info("Hu 2014  D-S Theory: acc: %.2f"%(acc_Hu_2014))
    result.acc_classic_DS=acc_Hu_2014
    #-------------------------Xiao2020
    acc_Xiao_2020 = xiao2020DSFusion(np.array(pba_aco),np.array(pba_seis),label)
    logger.info("Xiao 2020  D-S Theory: acc: %.2f"%(acc_Xiao_2020))
    result.acc_classic_DS=acc_Xiao_2020
    result_all.append(result)
#store result
pickle.dump(result_all, open(save_file,'wb'))
    