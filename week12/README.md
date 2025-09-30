# **Self2Self与Deep Image Prior图像去噪对比实验**

## 实验目标
本实验主要比较Self2Self和Deep Image Prior两种去噪算法。这两种方法的核心思想都是利用神经网络本身的结构作为正则化器，而无需使用任何干净的训练数据。根据现有结果可得到，Self2Self在去噪性能上通常优于DIP，但其计算成本远高于DIP；而DIP虽然速度更快，但其性能高度依赖于早停策略，且结果稳定性较差。

##  安装依赖
```bash
pip install numpy opencv-python scikit-image pywavelets bm3d matplotlib tqdm scipy keras tensorflow
```
**主要依赖库**：
- OpenCV
- NumPy
- PyWavelets
- scikit-image
- BM3D
- Matplotlib
- Tqdm
- Scipy
- Keras
- Tensorflow

##  安装环境
GPU:服务器
CUDA: 12.6
cuDNN: 8.9.7
Python:3.9


##  快速开始
1. **下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   解压到代码目录下的"project/week6/Set14"文件夹


2. **运行实验**：
   ```python
   python new.py(self2self)
   python 2.py(DIP)
   ```


##  项目结构
```
project
├── week6/
│   ├──Deep/                                       # DIP去噪算法
│   │   ├──data/                                   # 测试数据集
│   │   │   ├── denoising/                           
│   │   │   │   ├── ppt3.png
│   │   │   │   ├── ppt3/                           # 结果图
│   │   ├──models/                                 # 预训练模型 
│   │   ├──utils/       
│   │   ├── 2.py                                   # 主程序   
│   ├──self2self/
│   │   ├──network/                                
│   │   ├──testsets/                               # 测试数据集
│   │   │   ├──Set9/
│   │   │   │   ├──5/                              # 结果图
│   │   ├──demo_denoising.py                       # 图像去噪
│   │   ├──demo_inpainting.py                      # 图像补全
│   │   ├──util.py                       
│   │   ├──new.py                                          
│   └── README.md                 
```

##  实验结果示例

1)DIP去噪
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week12/DIP/1.jpg?raw=true)

2)S2S去噪
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week12/S2S/4.jpg?raw=true)

##  实验结论
DIP 在去噪实验中表现出良好的噪声抑制能力，背景干净、数字边缘清晰，整体视觉效果自然，计算效率高，适合快速处理；但存在轻微过平滑，细节略有损失。相比之下，Self2Self 理论上具备更强的细节保持能力和稳定性，但在本次实验中由于掩膜概率设置不当、图像空白区域占比大，导致网络学习到错误的灰度均值，去噪结果中残留明显盐霜噪声，背景出现灰斑，整体对比度下降，视觉效果反而不如 DIP。总体而言，DIP 更适合结构简单、纹理较少的图像去噪任务，而 S2S 需在参数调优和图像特性匹配上进一步改进。


---