import numpy as np
def generateBPAXiao(out_aco_BPA, out_semi_BPA):
    B_12=0
    for i in range(len(out_aco_BPA)):
        B_12+=out_aco_BPA[i]*np.log(out_aco_BPA[i]/(0.5*out_aco_BPA[i]+0.5*out_semi_BPA[i]))
    for i in range(len(out_semi_BPA)):
        B_12+=out_semi_BPA[i]*np.log(out_semi_BPA[i]/(0.5*out_aco_BPA[i]+0.5*out_semi_BPA[i]))
    RB12 = np.sqrt(np.abs(B_12))
    RB_hat_1 = RB12
    RB_hat_2 = RB12
    
    S_1 = 1/RB_hat_1
    S_2 = 1/RB_hat_2
    
    c_1 = 0.5
    c_2 = 0.5
    
    m_new = c_1*out_aco_BPA + c_2*out_semi_BPA
    
    return m_new
def fuseResultCal(out_aco_BPA,out_semi_BPA):
    out_fuse_BPA_xiao = generateBPAXiao(out_aco_BPA, out_semi_BPA)
    con_matrix = np.zeros((3,3))+1-np.diag((1,1,1))
    k_origin = np.dot(np.dot(out_aco_BPA, con_matrix),out_semi_BPA.T)
    con_mass = 1/(1-k_origin)
    mass = con_mass * np.multiply(out_aco_BPA, out_semi_BPA)
    
    k_improved_xiao = np.dot(np.dot(mass, con_matrix),out_fuse_BPA_xiao.T)
    con_mass_improved_xiao = 1/(1-k_improved_xiao)
    mass_improved_xiao = con_mass_improved_xiao * np.multiply(mass, out_fuse_BPA_xiao)
    fuse_ans =mass_improved_xiao
    return fuse_ans
def xiao2020DSFusion(pba_aco,pba_seis,label):
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