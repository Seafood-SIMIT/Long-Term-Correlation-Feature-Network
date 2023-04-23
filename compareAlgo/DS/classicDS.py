from doctest import OutputChecker
import numpy as np

def fuseResultCal(out_aco_BPA,out_semi_BPA):
    con_matrix = np.zeros((3,3))+1-np.diag((1,1,1))
    k_origin = np.dot(np.dot(out_aco_BPA, con_matrix),out_semi_BPA.T)
    con_mass = 1/(1-k_origin)
    mass = con_mass * np.multiply(out_aco_BPA, out_semi_BPA)
    fuse_ans = mass
    return fuse_ans
def classicDSFusion(pba_aco,pba_seis,label):
    fuse_ans = []
    for i in range(len(label)):
        fuse_ans.append( fuseResultCal(pba_aco[i],pba_seis[i]))

    return fuse_ans
    count =0 
    for i in range(len(label)):
        fuse_ans = fuseResultCal(pba_aco[i],pba_seis[i])
        if fuse_ans == label[i]:
            count +=1
    return count/len(label)