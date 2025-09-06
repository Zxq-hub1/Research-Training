# **深度学习图像去噪算法对比**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. ADMM (交替方向乘子法) 
2. BM3D (块匹配3D滤波)
3. 面向任意噪声水平的快速灵活图像去噪FFDNet
4. U-Net神经网络架构
5. N2N：让网络强行学习从一个带噪图片到另一个带噪图片的映射。

Noise2Noise（简称 N2N）是一种无需干净图像即可训练去噪网络的自监督策略,只要给网络输入两张独立的噪声图，输出逼近其中任意一张，就能学会去除噪声。

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
GPU：NVIDIA RTX 4060 (8GB)
CUDA: 12.6
cuDNN: 8.9.7
Python:3.9
关键库：TensorFlow 2.13, OpenCV 4.7


##  快速开始
1. **下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   解压到代码目录下的"project/week6/data/Test/Set14"文件夹

   噪声类型：高斯噪声（σ=15/25/50） 

2. **运行实验**：
   ```python
   python n2n_new2.py
   ```

3. **查看结果**：
   - 去噪效果图：`N2N_results2/image`
   - 量化指标：`N2N_results2/metrics`


##  参数配置
在`n2n_new2.py`中修改实验参数：

| 算法类别  | 代表算法       | 参数设置                                                 | 
|-------|------------|------------------------------------------------------|
| 传统方法  | BM3D       | σ=15、25、50                                           | 
| 深度学习  | DnCNN      | 预训练模型 (final_model.keras)                            | 
| 深度学习  | FFDNet     | 预训练模型 (best_model.pth)                               | 
| 深度学习  | UNet       | 预训练模型 (unet_gray_sigma25.pth/unet_color_sigma25.pth) | 
| 深度学习  | Noise2Noise | 预训练模型 (best.pth)                                     | 
| 迭代算法  | ISTA       | λ=15, max_iter=100                                   | 
| 正则化方法 | FISTA/ADMM | 自适应参数                                                | 



##  项目结构
```
project
├── week6/
│   ├──data/
│   │   ├──Test/                                   # 测试数据集
│   │   │   ├── Set14/                           
│   │   ├──DIV2K_train/                            # 训练数据集
│   │   ├──DIV2K_valid/                            # 验证数据集
│   ├──models/                                     # 预训练模型
│   │   ├──N2N_DIV2K  
│   │   │   ├──best.pth
│   │   │   ├──epoch.pth                            
│   ├── N2N_results2/                              # 输出结果
│   │   ├──image
│   │   ├──metrics                                 # 汇总统计        
│   ├── n2n_new2.py                                # 主程序(彩色）    
│   ├── train_Unet.py                              # 模型训练
│   └── UNet_README.md                 
```

##  实验结果示例

1）添加高斯(σ=15、σ=25、σ=50)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week9/results/ppt3.jpg?raw=true)

2）高斯噪声在测试集上的统计量化平均值（PSNR 、SSIM 、Time)

| Algorithm    | σ = 15          | σ = 25          | σ = 50          |
|--------------|-----------------|-----------------|-----------------|
|              | PSNR | SSIM  | Time | PSNR | SSIM  | Time | PSNR | SSIM  | Time |
| BM3D         | 30.68| 0.8899| 1.36 | 27.91| 0.8173| 1.37 | 24.09| 0.6805| 1.38 |
| ISTA         | 26.59| 0.7298| 0.28 | 22.38| 0.5418| 0.29 | 16.28| 0.2974| 0.29 |
| FISTA        | 28.76| 0.8403| 0.09 | 26.40| 0.7650| 0.12 | 22.98| 0.6160| 0.10 |
| ADMM         | 29.02| 0.8453| 0.12 | 26.35| 0.7545| 0.13 | 22.50| 0.5757| 0.15 |
| DnCNN        | 24.16| 0.5612| 0.13 | 21.29| 0.4454| 0.12 | 16.35| 0.2915| 0.14 |
| FFDNet       | 30.47| 0.8836| 0.02 | 28.16| 0.8086| 0.02 | 24.77| 0.6733| 0.02 |
| UNet         | 27.31| 0.8155| 0.11 | 27.61| 0.8295| 0.10 | 21.74| 0.5520| 0.10 |
| Noise2Noise  | 22.75| 0.6571| 0.06 | 22.69| 0.6488| 0.06 | 21.80| 0.5975| 0.06 |
##  实验结论

BM3D和FFDNet在三个噪声水平下都表现出最优异的性能， FFDNet在处理时间上具有显著优势，仅需0.02秒。

传统优化算法中，FISTA和ADMM表现较为稳定，在不同噪声水平下保持了较好的去噪效果。

UNet在σ=25噪声水平下表现最佳，但在噪声σ=50下性能下降明显。

Noise2Noise方法虽然处理时间较短，但去噪效果明显不如其他先进方法，PSNR和SSIM指标均较低。还需进一步对模型进行修改训练。

---
