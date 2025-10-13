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
   python demo_denoising.py(self2self)
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
├──self2self/
│   ├──network/                                
│   ├──testsets/                                   # 测试数据集
│   │   ├──Set9/
│   │   │   ├──5/                                  # 结果图
│   │   ├──BSD68/
│   │   │   ├──11/                                 # 结果图
│   │   ├──PolyU/
│   │   │   ├──45/                                 # 结果图
│   │   ├──Set14/
│   │   │   ├──ppt3/                               # 结果图
│   ├──demo_denoising.py                           # 图像去噪
│   ├──demo_inpainting.py                          # 图像补全
│   ├──util.py                                                               
│   └──README.md                 
```

##  实验结果示例

1)DIP去噪
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week13/DIP/ppt3_results/output3.png?raw=true)

2)S2S去噪
- 对Set9-5.png添加高斯噪声（25）并进行去噪---dropout=0.3：
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week13/Self2Self/results/Set9/1.jpg?raw=true)

- 对BSD68-45.png添加高斯噪声（25、50）进行去噪---σ=25时dropout=0.2   σ= 50时dropout=0.3 ：
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week13/Self2Self/results/BSD68/2.jpg?raw=true)

- 对真实噪声PolyU-45.jpg进行去噪：
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week13/Self2Self/results/PolyU/3-s2s.jpg?raw=true)

- 对ppt3添加高斯噪声（15、25、50）并进行去噪---dropout=0.3：
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week13/Self2Self/results/Set14/results-ppt3.jpg?raw=true)



| Algorithm | testsets       | σ = 15     | σ = 25     | σ = 50     | real noise |
|-----------|----------------|------------|------------|------------|------------|
| self2self | Set9-5.png     | -          | 33.2729dB  | -          | -          |
| self2self | BSD68-11.png   | -          | 28.8336dB  | 25.5846dB  | -          |
| self2self | PolyU-45.jpg   | -          | -          | -          | 38.1538dB  |
| self2self | Set14-ppt3.png | 36.5425 dB | 35.2191 dB | 32.2397 dB | -          |
| DIP       | Set14-ppt3.png | -          | 28.1230dB  | -          | -          |


##  实验结论

Self2Self 与 Deep Image Prior（DIP）虽同属“无需干净样本”的自监督框架，但内在机制截然不同：
前者在网络训练阶段通过多次采样平均抑制噪声，同时保留图像固有结构；后者则把噪声本身视为网络拟合目标，依靠早期停止防止过拟合，藉由卷积结构先验自然滤除高频扰动。

对比结果可见，Self2Self 的两组输出几乎与干净图像无异，边缘锐利、纹理完整，仅在 σ=50 的暗区有极轻微平滑，证明其随机采样策略对细节保持更友好；
而 DIP 结果整体亮度正确，却伴随明显划痕式伪影与边缘振铃，尤其在平坦区域出现低频“油画”斑块，反映出卷积网络在欠拟合与过拟合临界点附近对噪声残留与结构重建的权衡更为敏感，且缺乏像素级统计平均，导致高频误差未被充分抵消。

综上，Self2Self 以“随机采样+平均”显式抑制随机噪声，保真度更高；DIP 则依赖网络结构先验与早期停止，虽能重建主结构，却易遗留伪影，细节稳健性略逊。

---