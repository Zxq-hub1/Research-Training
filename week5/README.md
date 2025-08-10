# **深度学习图像去噪算法对比实验报告**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)
5. 基于深度学习的DnCNN

对比传统算法（BM3D）与基于深度学习的DnCNN在图像去噪任务中的性能差异,分析迭代优化算法（ISTA/FISTA）与TV正则化方法的效果,评估不同噪声类型（高斯/椒盐）对算法鲁棒性的影响。

本文针对高斯噪声与椒盐噪声污染下的图像恢复问题，系统对比了传统算法（BM3D）、深度学习模型（DnCNN）及迭代优化方法（ISTA/FISTA）的性能差异。根据现有结果的得知，深度学习DnCNN应均优于传统方法，且GPU加速下处理速度会提升；TV正则化方法在强噪声场景下展现更好的边缘保持能力。本研究为不同噪声环境下的算法选择提供了实证依据。

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
   解压到代码目录下的"project/week5/Set14"文件夹

   噪声类型：高斯噪声（σ=15/25/50） 椒盐噪声（密度=5%/10%/20%）

2. **运行实验**：
   ```python
   python algorithm5.py
   ```

3. **查看结果**：
   - 去噪效果图及收敛曲线：`results/convergence`、`results/image`
   - 细节结果：`results/detailed_results.xlsx`
   - 量化指标：`results/summary.csv`


##  参数配置
在`algorithm5.py`中修改实验参数：

| 算法类别  | 代表算法  | 参数设置               | 
|-------|-------|--------------------|
| 传统方法  | BM3D  | σ=噪声标准差            | 
| 深度学习  | DnCNN | 预训练模型              | 
| 迭代算法  | ISTA/FISTA (L1小波) | λ=15, max_iter=100 | 
| 正则化方法 | FISTA-TV/ADMM-TV | λ=0.1, ρ=1.0       | 



##  项目结构
```
project
├── week5/
│   ├── Set14/                                    # 测试数据集
│   ├── results/                                  # 输出结果
│   │   ├──image
│   │   │   ├──gaussian                           #高斯噪声
│   │   │   │   ├── 15
│   │   │   │   │   ├── denoising_baboon.png      # 每张图像的去噪结果
│   │   │   │   │   │── denoising_barbara.png             
│   │   │   │   │   │── ...
│   │   │   │   ├── 25
│   │   │   │   │   ├── ...
│   │   │   │   ├── 50
│   │   │   │   │   │── ...
│   │   │   ├──salt_pepper                        #椒盐噪声
│   │   │   │   ├── 0.1
│   │   │   │   ├── ...
│   │   ├──convergence                            # 收敛曲线
│   │   │   ├──gaussian  
│   │   │   │   ├──...
│   │   │   ├──salt_pepper
│   │   │   │   ├──...
│   │   ├──detailed_results.xlsx                  # 所有图像的详细结果
│   │   ├──summary.csv                            # 汇总统计表格         
│   ├── algorithm4.py                             # 主程序   
│   └── README.md                 
```

##  实验结果示例

1)添加高斯噪声(σ=25)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week5/gaussian-denoising_ppt3.png?raw=true)

2)添加椒盐噪声(p=1)，并进行去噪:
![添加椒盐噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week5/denoising_ppt3.png?raw=true)

3)高斯噪声（25）和椒盐噪声（0.1）在测试集上的统计量化结果（PSNR 、SSIM 、Time)

| NoiseType   | Param | Algorithm | PSNR  | SSIM   | Time  |
|-------------|-------|-----------|-------|--------|-------|
| gaussian    | 25.0  | BM3D      | 28.07 | 0.8238 | 0.57  |
| gaussian    | 25.0  | ISTA      | 22.33 | 0.5374 | 0.11  |
| gaussian    | 25.0  | FISTA     | 22.19 | 0.5328 | 0.50  |
| gaussian    | 25.0  | FISTA-TV  | 26.17 | 0.7490 | 0.72  |
| gaussian    | 25.0  | ADMM      | 23.74 | 0.6987 | 0.36  |
| gaussian    | 25.0  | DnCNN     | 6.44  | 0.0429 | 0.04  |
| salt_pepper | 0.1   | BM3D      | 18.99 | 0.4707 | 0.57  |
| salt_pepper | 0.1   | ISTA      | 16.70 | 0.3418 | 0.12  |
| salt_pepper | 0.1   | FISTA     | 16.55 | 0.3402 | 0.35  |
| salt_pepper | 0.1   | FISTA-TV  | 20.98 | 0.4812 | 0.87  |
| salt_pepper | 0.1   | ADMM      | 22.17 | 0.5854 | 0.37  |
| salt_pepper | 0.1   | DnCNN     | 6.45  | 0.0469 | 0.05  |


##  实验结论

BM3D算法在高斯噪声环境下表现最优，展现了传统非局部方法对高斯噪声的有效性；椒盐噪声条件下，其性能显著下降，反映了非局部方法对离散噪声的局限性。其次，ADMM算法在椒盐噪声场景下表现突出，验证了全变分正则化对脉冲噪声的适应性。

DnCNN在所有测试条件下均表现异常，这种系统性失效表明可能存在模型实现错误、训练数据不匹配或超参数设置不当等根本性问题.

实验结果还揭示了SSIM指标对椒盐噪声更为敏感的特性，所有算法在该条件下的SSIM值均出现显著下降，表明离散噪声对图像结构信息的破坏更为严重。

这些发现强调了在实际应用中开展噪声特性分析的重要性，并为不同场景下的算法选择提供了实证依据：推荐使用BMSD处理高斯噪声，ADMM处理椒盐噪声，后期要对DnCNN实现方案进行全面的诊断和优化。


---
