# **深度学习图像去噪算法对比**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)
5. 基于深度学习的DnCNN
6. 面向任意噪声水平的快速灵活图像去噪FFDNet
7. U-Net神经网络架构

U-Net是一种最初为生物医学图像分割而设计的卷积神经网络（CNN）架构。它由Olaf Ronneberger等人于2015年在论文《U-Net: Convolutional Networks for Biomedical Image Segmentation》中提出。由于其独特的U型对称结构和出色的性能，它迅速成为图像分割领域的标杆性模型，并被广泛Adapt到其他图像任务中，如图像去噪、超分辨率、缺陷检测等。


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
   彩色：python Unet.py
   灰度：python Unet_gray.py
   ```

3. **查看结果**：
   - 去噪效果图：`Unet_results/image``Unet_color_results/image`
   - 量化指标：`Unet_results/summary.csv``Unet_color_results/metrics/`


##  参数配置
在`FFDNet.py`中修改实验参数：

| 算法类别  | 代表算法       | 参数设置                                                 | 
|-------|------------|------------------------------------------------------|
| 传统方法  | BM3D       | σ=15、25、50                                           | 
| 深度学习  | DnCNN      | 预训练模型 (final_model.keras)                            | 
| 深度学习  | FFDNet     | 预训练模型 (best_model.pth)                               | 
| 深度学习  | UNet       | 预训练模型 (unet_gray_sigma25.pth/unet_color_sigma25.pth) | 
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
│   │   ├──UDNet/  
│   │   │   ├──checkpoints
│   │   │   │   ├──best_model_unet_color.pth
│   │   │   │   ├──best_model_unet_gray.pth
│   │   │   ├──unet_gray_sigma25.pth               # 训练结果（灰度）
│   │   │   ├──unet_color_sigma25.pth              # 训练结果（彩色）
│   ├── Unet_results/                              # 输出结果（灰度）
│   │   ├──image
│   │   ├──summary.csv                             # 汇总统计表格    
│   ├── Unet_color_results/                        # 输出结果（彩色）     
│   ├── Unet.py                                    # 主程序(彩色）  
│   ├── Unet_gray.py                               # 主程序(灰度）  
│   ├── train_Unet.py                              # 模型训练
│   └── UNet_README.md                 
```

##  实验结果示例

1）添加高斯(σ=15、σ=25、σ=50)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week8/gray.jpg?raw=true)

![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week8/color.jpg?raw=true)


2）高斯噪声在测试集上的统计量化平均值（PSNR 、SSIM 、Time)

| Algorithm | σ = 15 | σ = 15   |σ = 15 | σ = 25 |σ = 25 |σ = 25 | σ = 50 |  σ = 50 |  σ = 50 |
|:---|:---|:---------|:---|:---|:---|:---|:---|:---|:---|
| | **PSNR** | **SSIM** | **Time** | **PSNR** | **SSIM** | **Time** | **PSNR** | **SSIM** | **Time** |
| BM3D | 30.68 | 0.8898   | 1.31 | 27.94 | 0.8184 | 1.33 | 24.09 | 0.6804 | 1.34 |
| ISTA | 26.59 | 0.7297   | 0.29 | 22.38 | 0.5418 | 0.29 | 16.27 | 0.2971 | 0.29 |
| FISTA | 28.74 | 0.8399   | 0.13 | 26.42 | 0.7652 | 0.09 | 22.98 | 0.6165 | 0.12 |
| ADMM | 29.00 | 0.8453   | 0.11 | 26.37 | 0.7547 | 0.12 | 22.49 | 0.5760 | 0.15 |
| DnCNN | 24.16 | 0.5609   | 0.13 | 21.29 | 0.4460 | 0.11 | 16.34 | 0.2199 | 0.11 |
| FFDNet | 30.44 | 0.8832   | 0.02 | 28.20 | 0.8094 | 0.02 | 24.77 | 0.6733 | 0.02 |
| UNet | 27.29 | 0.8152   | 0.11 | 27.62 | 0.8295 | 0.11 | 21.72 | 0.5513 | 0.11 |   | 0.6616    | 0.02      |

##  实验结论

从去噪对于需要高质量去噪的应用，BM3D仍然是可靠性最高的选择；对于实时性要求较高的场景，FFDNet提供了最佳的速度-质量权衡；UNet在特定噪声水平下表现良好，但泛化能力有待提高；DnCNN适用于处理特定噪声水平的图像。

---
