import numpy as np
def chen_improvedds(out_aco_BPA, out_semi_BPA):
    #print(out_aco_BPA.shape, out_semi_BPA.shape)
    distance = np.zeros(len(out_aco_BPA))
    # 距离矩阵
    for i in range(len(out_aco_BPA)):
        #distance[i]=1-abs(out_aco_BPA[0,i]-out_semi_BPA[0,i])/max([out_aco_BPA[0,i],out_semi_BPA[0,i]])
        distance[i]=1-abs(out_aco_BPA[0,i]-out_semi_BPA[0,i])/max([out_aco_BPA[0,i],out_semi_BPA[0,i]])

    distance = distance/np.sum(distance)
    #out_aco_BPA = np.multiout_aco_BPA
    out_aco_BPA = np.multiply(out_aco_BPA, distance)
    out_aco_BPA = out_aco_BPA/np.sum(out_aco_BPA)
    out_semi_BPA = np.multiply(out_semi_BPA, distance)
    out_semi_BPA = out_semi_BPA/np.sum(out_semi_BPA)
    return out_aco_BPA,out_semi_BPA
def fuseResultCal(out_aco_BPA,out_semi_BPA):
    out_aco_BPA_hu,out_semi_BPA_hu=chen_improvedds(out_aco_BPA, out_semi_BPA)
    con_matrix = np.zeros((3,3))+1-np.diag((1,1,1))
    k_improved_hu = np.dot(np.dot(out_aco_BPA_hu, con_matrix),out_semi_BPA_hu.T)
    con_mass_improved_hu = 1/(1-k_improved_hu)
    mass_improved_hu = con_mass_improved_hu * np.multiply(out_aco_BPA_hu, out_semi_BPA_hu)
    fuse_ans = np.argmax(mass_improved_hu,axis=1)
    return fuse_ans
def hu2014DSFusion(pba_aco,pba_seis,label):
    count =0 
    for i in range(len(label)):
        fuse_ans = fuseResultCal(pba_aco[i],pba_seis[i])
        if fuse_ans == label[i]:
            count +=1
    return count/len(label)
    