# **Neighbor2Neighbor去噪算法**

## 实验内容

Neighbor2Neighbor是一种无需干净目标图像即可训练深度去噪网络的自监督学习方法。该方法的关键创新在于一种新颖的邻域下采样器，它可以从单张噪声图像中构建训练对，以及一个协同损失函数，该函数强制这些训练对去噪后输出之间的一致性。

##  安装依赖
```bash
pip install numpy opencv-python scikit-image pywavelets bm3d matplotlib tqdm scipy keras tensorflow torch
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
- torch

##  安装环境
GPU：NVIDIA RTX 4060 (8GB)
CUDA: 12.6
cuDNN: 8.9.7
Python:3.9


##  快速开始
1. **下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   解压到代码目录下的"project/week6/data/Test/Set14"文件夹
2. **下载DIV2K数据集**：
   ```bash
   https://www.kaggle.com/datasets/takihasan/div2k-dataset-for-super-resolution
   ```
   解压到代码目录下的"project/week6/data/DIV2K_train(DIV2K_valid)"文件夹

   噪声类型：高斯噪声（σ=15/25/50） 

2. **运行实验**：
   ```python
   python new.py
   ```

##  项目结构
```
project
├── week6/
│   ├──data/
│   │   │   ├──Test/                                   # 测试数据集
│   │   │   │   ├── Set14/                           
│   │   │   ├──DIV2K_train/                            # 训练数据集
│   │   │   ├──DIV2K_valid/                            # 验证数据集
│   ├──Neighbor2Neighbor
│   │   ├──neighbor_results/                           # 预训练模型
│   │   │   ├──unet_gauss25_b4e200r02  
│   │   │   │   ├──2025-09-18-07-22
│   │   │   │   │   ├──epoh_model_020.pth                            
│   │   ├── results/                                   # 输出结果       
│   │   ├── arch_unet.py                                    
│   │   ├── train.py                                   # 训练代码
│   │   ├── models.py
│   │   └── new.py                                     # 主程序
               
```

##  实验结果示例

1）noise2noise去噪结果
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week11/results/ppt3.jpg?raw=true)

2）高斯噪声在测试集上的统计量化值


| Algorithm          | σ = 15         |                |       | σ = 25         |                |       | σ = 50         |                |       |
| :----------------- | :------------- | :------------- | :---- | :------------- | :------------- | :---- | :------------- | :------------- | :---- |
|                    | **PSNR**       | **SSIM**       | **Time** | **PSNR**       | **SSIM**       | **Time** | **PSNR**       | **SSIM**       | **Time** |
| BM3D               | 30.68          | 0.8897         | 2.33   | 27.91          | 0.8173         | 2.38   | 24.09          | 0.6805         | 2.42   |
| ISTA               | 26.59          | 0.7298         | 0.49   | 22.38          | 0.5417         | 0.49   | 16.28          | 0.2971         | 0.50   |
| FISTA              | 28.73          | 0.8395         | 0.18   | 26.42          | 0.7650         | 0.23   | 22.97          | 0.6160         | 0.22   |
| ADMM               | 29.00          | 0.8452         | 0.17   | 26.37          | 0.7555         | 0.18   | 22.50          | 0.5758         | 0.21   |
| DnCNN              | 24.16          | 0.5612         | 0.13   | 21.29          | 0.4454         | 0.12   | 16.35          | 0.2915         | 0.14   |
| FFDNet             | 30.44          | 0.8830         | 0.01   | 28.20          | 0.8095         | 0.01   | 24.76          | 0.6729         | 0.01   |
| UNet               | 27.30          | 0.8147         | 0.01   | 27.63          | 0.8301         | 0.01   | 21.73          | 0.5517         | 0.01   |
| Noise2Noise        | 22.68          | 0.6332         | 0.07   | 22.24          | 0.6106         | 0.09   | 20.69          | 0.533          | 0.07   |
| Neighbor2Neighbor  | 28.91          | 0.8159         | 0.05   | 25.33          | 0.6822         | 0.05   | 20.07          | 0.4605         | 0.05   |
##  实验结论

Neighbor2Neighbor框架为自监督图像去噪提供了一个有效且优雅的解决方案。通过利用图像的空间局部一致性先验，在保持图像细节和结构完整性方面超越了Noise2Noise，为自监督去噪提供了一条更有效的技术路径。通过形式化地定义邻域下采样器和协同损失函数，我们为在缺乏干净真值数据的情况下训练去噪模型建立了数学上合理的基础。一致性损失和正则化损失之间的协同作用使网络能够有效地分离信号和噪声，实现了与监督方法相竞争的性能。


---
