# GWAN
This is the code library for the implementation of GWAN. [A Kolmogorov-Arnold-informed Interpretable Graph Wavelet Activation Network for Machine Fault Diagnosis](https://ieeexplore.ieee.org/abstract/document/10079151).

![GWAN-based-fault diagnosis](https://github.com/HazeDT/GWAN/blob/main/GWAN.jpg)


# Note
IIn GWAN, two critical components are designed, that is, graph wavelet activation convolution (GWAConv) layer and wavelet attention (WavAtt) layer. In GWAConv, the graph message passing is achieved using the wavelet Kolmogorov-Arnold layer with learnable scale and translation parameters to capture the robust fault features. While WavAtt layer decomposes the raw signal into low-frequency and high-frequency components to force the model to focus on the low-frequency components which is helpful for fault diagnosis.
# sample data
The data for running this code can be found in [PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark)

# Implementation
python ./GWAN_master/train_graph.py --model_name GWAN  --data_name XJTUSpurgearRadius --data_dir ./datasets/XJTUSpurgearRadius_constant.pkl


# Citation

GWAN:
@ARTICLE{10079151,
  author={Li, Tianfu and Sun, Chuang and Zhao, Zhibin and Liu, Tao and Chen, Xuefeng and Yan, Ruqiang},
  journal={IEEE Transactions on Systems, Man and Cybernetics, Systems}, 
  title={A Kolmogorov-Arnold-informed Interpretable Graph Wavelet Activation Network for Machine Fault Diagnosis}, 
  year={206},
  volume={},
  number={},
  pages={},
  doi={}}


