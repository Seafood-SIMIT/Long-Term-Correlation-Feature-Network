# Long-Term-Correlation-Feature-Network
LTCFN
是使用帧间特征进行一维声震同步信号进行融合分类的方法，同时还包括了部分比较方法如MFCC，Wavelet，CNN的Pytorch实现。借助[训练工具](https://github.com/Seafood-SIMIT/snake_spear_with_102_inch-A_DL_Trainer).

## 论文
论文链接：https://ieeexplore.ieee.org/document/10093799

![基本Idea](LTCFN.png)

    @INPROCEEDINGS{9624499,
    author={Sun, Lin and Liu, Jianpo and Liu, Yuanqing and Li, Baoqing},
    booktitle={2021 International Conference on Control, Automation and Information Sciences (ICCAIS)}, 
    title={HRRP Target Recognition Based On Soft-Boundary Deep SVDD With LSTM}, 
    year={2021},
    volume={},
    number={},
    pages={1047-1052},
    doi={10.1109/ICCAIS52680.2021.9624499}}

## 结果介绍
![valid confusion matrix](cm_ltcfn.png)

