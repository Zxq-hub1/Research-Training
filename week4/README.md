# **FISTA、ADMM与BM3D、ISTA算法在图像去噪中的对比**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)

通过在Set14标准数据集上添加不同噪声强度的高斯噪声（15、25、50）和椒盐噪声（0.1、0.2、0.05），采用PSNR、SSIM和计算时间作为评价指标。实验结果表明，BM3D算法在高斯噪声环境下表现最为突出，在椒盐噪声场景中，ADMM展现出更强的适应性，即对质量要求高的场景可选用BM3D（高斯噪声）或ADMM（椒盐噪声），而实时性要求高的场景则可考虑ISTA。

##  安装依赖
```bash
pip install numpy opencv-python scikit-image pywavelets bm3d matplotlib tqdm scipy
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

##  快速开始
1. **下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   解压到代码目录下的"project/week4/Set14"文件夹

2. **运行实验**：
   ```python
   python algorithm4.py
   ```

3. **查看结果**：
   - 去噪效果图及收敛曲线：`results/gaussian`、`results/salt_pepper`
   - 细节结果：`results/detailed_results.xlsx`
   - 量化指标：`results/summary.csv`


##  参数配置
在`test.py`中修改实验参数：
```python
# 定义噪声配置
    noise_configs = [
        ('gaussian', 15),
        ('gaussian', 25),
        ('gaussian', 50),
        ('salt_pepper', 0.05),
        ('salt_pepper', 0.1),
        ('salt_pepper', 0.2)
    ]

# ISTA参数
ista_params = {'lambda_'= 15,'max_iter'= 100,'tol'= 1e-6,'wavelet': 'db8','level' = 4}

# FISTA_L1参数
fista_l1_params = {'lambda_'= 15,'max_iter'= 100,'tol'= 1e-6,'wavelet': 'db8','level' = 4}

# FISTA_TV参数
fista_tv_params = {'lambda_'=0.1,'max_iter'=100, 'tol'=1e-6}

# ADMM_TV参数
admm_tv_params = {'lambda_'= 0.1,'rho'=1.0, 'max_iter'=100, 'tol'=1e-6}
```

##  项目结构
```
project
├── week4/
│   ├── Set14/                                # 测试数据集
│   ├── results/                              # 输出结果
│   │   ├──gaussian                           #高斯噪声
│   │   │   ├── 15
│   │   │   │   ├── denoising_baboon.png      # 每张图像的去噪结果
│   │   │   │   │── denoising_barbara.png             
│   │   │   │   │── ...
│   │   │   │   │── convergence_baboon.png    # 收敛曲线
│   │   │   │   │── convergence_barbara.png
│   │   │   │   ├──...
│   │   │   ├── 25
│   │   │   │   ├── ...
│   │   │   ├── 50
│   │   │   │   │── ...
│   │   ├──salt_pepper                        #椒盐噪声
│   │   │   ├── 0.1
│   │   │   ├── ...
│   │   ├──detailed_results.xlsx              # 所有图像的详细结果
│   │   ├──summary.csv                        # 汇总统计表格         
│   ├── algorithm4.py                         # 主程序   
│   └── README.md                 
```

##  实验结果示例

1)添加高斯噪声(σ=25)，并进行BM3D、ISTA、FISTA、ADMM去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week4/ppt3/gaussian_ppt3.png?raw=true)

2)添加椒盐噪声(p=1)，并进行BM3D、ISTA、FISTA、ADMM去噪:
![添加椒盐噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week4/ppt3/salt_pepper_ppt3.png?raw=true)

3)高斯噪声（25）和椒盐噪声（0.1）的去噪结果均值（选取部分噪声强度的结果展示）

| NoiseType   | Param | Algorithm | Avg PSNR(dB) | Avg SSIM | Avg Time(s) |
|-------------|-------|-----------|--------------|----------|-------------|
| gaussian    | 25.0  | BM3D      | 28.07        | 0.8238   | 0.57        |
| gaussian    | 25.0  | ISTA      | 22.33        | 0.5374   | 0.11        |
| gaussian    | 25.0  | FISTA     | 22.19        | 0.5328   | 0.50        |
| gaussian    | 25.0  | FISTA-TV  | 26.17        | 0.7490   | 0.72        |
| gaussian    | 25.0  | ADMM      | 23.74        | 0.6987   | 0.36        |
| salt_pepper | 0.1   | BM3D      | 18.99        | 0.4707   | 0.57        |
| salt_pepper | 0.1   | ISTA      | 16.70        | 0.3418   | 0.12        |
| salt_pepper | 0.1   | FISTA     | 16.55        | 0.3402   | 0.35        |
| salt_pepper | 0.1   | FISTA-TV  | 20.98        | 0.4812   | 0.87        |
| salt_pepper | 0.1   | ADMM      | 22.17        | 0.5854   | 0.37        |


##  实验结论

本次实验系统性地评估了四种经典图像去噪算法在Set14数据集上的性能表现，通过定量指标和收敛性分析揭示了关键发现。

可以得出在算法性能方面，不同去噪方法展现出明显的噪声特异性。BM3D算法在高斯噪声环境下表现最为突出，在噪声强度σ=15时取得30.83dB的PSNR和0.8946的SSIM，显著优于其他算法。特别是在强高斯噪声（σ=50）条件下，BM3D仍能保持24.29dB的PSNR，比次优算法ADMM高出约2.2dB。然而，在椒盐噪声场景中，ADMM展现出更强的适应性，在噪声密度0.05时以23.24dB的PSNR和0.6622的SSIM领先，而BM3D的性能则随噪声密度增加急剧下降。这一对比验证了必须根据噪声特性进行针对性去噪方法选择的结果。

在计算效率方面，ISTA始终保持最快速度，但其去噪质量往往最差；BM3D的时间成本相对稳定，而ADMM和FISTA_TV则需要更多计算资源。这种性能-效率的权衡关系为实际应用提供了明确指导：对质量要求高的场景可选用BM3D（高斯噪声）或ADMM（椒盐噪声），实时性要求高的场景则可考虑ISTA。

---
