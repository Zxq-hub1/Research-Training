# **FISTA、ADMM与BM3D、ISTA算法在图像去噪中的对比**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12%2B-green)

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)

通过在Set14标准数据集上添加σ=25的高斯噪声，采用PSNR、SSIM和计算时间作为评价指标。实验结果表明，BM3D算法在去噪效果上表现最优，而ADMM算法在效率与效果的平衡性上最具优势。本研究为不同应用场景下的算法选择提供了实证依据。

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
   解压到代码目录下的"project/week3/Set14"文件夹

2. **运行实验**：
   ```python
   python test.py
   ```

3. **查看结果**：
   - 去噪效果图及收敛曲线：`results`
   - 细节结果：`results/detailed_results。csv`
   - 量化指标：`results/summary.csv`


##  参数配置
在`test.py`中修改实验参数：
```python
noise_configs = [
    # 高斯噪声
    gaussian_noise(sigma=25)

# ISTA参数
ista_params = {'lambda_'= 10,'max_iter': 100,'tol'= 1e-5,'wavelet': 'db4','level' = 3}

# FISTA_L1参数
fista_l1_params = {'lambda_'= 10,'max_iter': 100,'tol'= 1e-5,'wavelet': 'db4','level' = 3}

# FISTA_TV参数
fista_tv_params = {'lambda_'=0.1,'max_iter'=100, 'tol'=1e-5}

# ADMM_TV参数
admm_tv_params = {'lambda_'= 0.1,'rho'=1.0, 'max_iter'=100, 'tol'=1e-5}
```

##  项目结构
```
project
├── week3/
│   ├── Set14/                        # 测试数据集
│   ├── results/                      # 输出结果
│   │   ├── denoising_baboon.png      # 每张图像的去噪结果
│   │   │── denoising_barbara.png             
│   │   │── ...
│   │   │── convergence_baboon.png    # 收敛曲线
│   │   │── convergence_barbara.png
│   │   │── ...
│   │   ├──detailed_results.csv        # 所有图像的详细结果
│   │   ├──summary.csv                # 汇总统计表格         
│   ├── test.py                       # 主程序   
│   └── README.md                 
```

##  实验结果示例

![添加高斯噪声，并进行BM3D、ISTA、FISTA、ADMM去噪](https://github.com/Zxq-hub1/Research-Training/blob/main/week3/ppt3/denoising.png?raw=true)

| Algorithm    | PSNR | SSIM | Time        |
|---------------|-----------|-----------|-------------|
| **BM3D**    | 26.56189499      | 0.759581036      | 0.508974075 |
| **ISTA**    | 16.7846345     | 0.316112214     | 0.15196228  |
| **FISTA**    | 16.70023976     | 0.312860403   | 0.227494478 |
| **FISTA_TV** | 13.27797667     | 0.176061565    | 0.805597544      |
| **ADMM** |16.52426896      | 0.186083478     | 0.00409627      |


##  实验结论

本次实验系统性地评估了四种经典图像去噪算法在Set14数据集上的性能表现，通过定量指标和收敛性分析揭示了关键发现。实验数据明确显示，BM3D算法以26.56dB的PSNR和0.76的SSIM显著优于其他方法，但其0.5秒的计算耗时较ADMM（0.004秒）高出两个数量级，这种性能差异为不同应用场景下的算法选择提供了重要参考。实验结果表明FISTA-TV版本的异常表现表明当前TV正则化参数设置需要优化，而ADMM算法在SSIM指标上的不足则反映了其在结构保持方面的改进空间。

---
