import numpy as np
import math
def addNoise(x, snr):
    '''
    
    add noise 
    snr: snr
    '''
    Nx = len(x)  # 求出信号的长度
    noise = np.random.randn(Nx)# 用randn产生正态分布随机数 
    signal_power = np.sum(x*x)/Nx# 求信号的平均能量
    noise_power = np.sum(noise*noise)/Nx# 求信号的平均能量
    noise_variance = signal_power/(math.pow(10., (snr/10)))#计算噪声设定的方差值
    noise = math.sqrt(noise_variance/noise_power)*noise# 按照噪声能量构成相应的白噪声
    y=x+noise
    return y



if __name__=='__main__':
    x = np.array([1.,2.,3.,4.])
    y=addNoise(x,-5)
    print('y=',y)
