# **ISTA与BM3D图像去噪算法对比实验**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12%2B-green)

本项目实现了**ISTA（迭代收缩阈值算法）**和**BM3D（三维块匹配滤波）**两种经典图像去噪算法，并在Set14数据集上进行了系统性对比实验。支持处理**高斯噪声**和**椒盐噪声**，提供PSNR、SSIM等量化评估指标。

##  项目特点
-  **完整实现**：包含ISTA和BM3D算法核心代码
-  **全面评估**：支持PSNR、SSIM和运行时间计算
-  **可视化对比**：生成噪声/去噪效果对比图
-  **参数可调**：灵活调整噪声强度、正则化系数等

##  安装依赖
```bash
pip install numpy opencv-python matplotlib scikit-image pywavelets bm3d tqdm
```
**主要依赖库**：
- OpenCV
- NumPy
- PyWavelets
- scikit-image
- BM3D
- Matplotlib
- Tqdm

##  快速开始
1. **下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   解压到代码目录下的"project/week2/Set14"文件夹

2. **运行实验**：
   ```python
   python algorithm.py
   ```

3. **查看结果**：
   - 去噪效果图：`denoising_results/per_image_results/`
   - 收敛曲线：`denoising_results/convergence/`
   - 量化指标：`denoising_results/summary_table.csv`
   - 步长分析：`denoising_results/step_size_impact.png`

##  参数配置
在`algorithm.py`中修改实验参数：
```python
noise_configs = [
    # 高斯噪声
    {'type': 'gaussian', 'intensity': 10, 25, 50}
    # 椒盐噪声
    {'type': 'salt_pepper', 'intensity': 0.05, 0.1, 0.2}
]

# ISTA参数
ista_params = {
    'max_iter': 100,
    'step_size': 0.8,
    'wavelet': 'db4'
}
```

##  项目结构
```
project
├── week2/
│   ├── Set14/                        # 测试数据集
│   ├── denoising_results/            # 输出结果
│   │   ├── per_image_results/        # 每张图像的去噪结果
│   │   │   ├── baboon/               # 示例图像目录
│   │   │   │   ├── Gaussian_15.png   # 去噪结果可视化
│   │   │   │   ├── Gaussian_25.png
│   │   │   │   └── ...
│   │   │   ├── barbara/
│   │   │   └── ...
│   │   │── convergence/              # 收敛曲线
│   │   │   ├── baboon_Gaussian_15_convergence.png
│   │   │   ├── baboon_Gaussian_25_convergence.png
│   │   │   └── ...
│   │   ├──summary_results.csv        # 所有图像的详细结果
│   │   ├──summary_table.csv          # 汇总统计表格
│   │   ├──summary_table.tex          # LaTeX格式汇总表格
│   │   ├──step_size_impact.png       # 步长影响分析图
│   ├── algorithm.py                  # 主程序   
│   └── README.md                 
```

##  实验结果示例

| Noise Type      | Intensity | Algorithm | PSNR (dB) | SSIM   | Time (s) | PSNR Imp. (%) | SSIM Imp. (%) |
|-----------------|-----------|-----------|-----------|--------|----------|---------------|---------------|
| **Gaussian**    | 15.0      | ISTA      | 28.12     | 0.7898 | 0.7810   | --            | --            |
|                 |           | BM3D      | 25.51     | 0.6894 | 1.1215   | -9.3%         | -12.7%        |
| **Gaussian**    | 25.0      | ISTA      | 25.48     | 0.6379 | 0.8072   | --            | --            |
|                 |           | BM3D      | 25.16     | 0.6780 | 1.1104   | -1.3%         | 6.3%          |
| **Gaussian**    | 50.0      | ISTA      | 18.47     | 0.3149 | 0.8237   | --            | --            |
|                 |           | BM3D      | 23.99     | 0.6240 | 1.1028   | 29.9%         | 98.2%         |
| **Salt-Pepper** | 0.05      | ISTA      | 22.83     | 0.5199 | 0.8199   | --            | --            |
|                 |           | BM3D      | 29.65     | 0.8670 | 1.1084   | 29.9%         | 66.7%         |
| **Salt-Pepper** | 0.1       | ISTA      | 20.31     | 0.3899 | 0.8268   | --            | --            |
|                 |           | BM3D      | 28.77     | 0.8392 | 1.1016   | 41.7%         | 115.2%        |
| **Salt-Pepper** | 0.2       | ISTA      | 16.76     | 0.2421 | 0.8281   | --            | --            |
|                 |           | BM3D      | 25.69     | 0.7197 | 1.1307   | 53.3%         | 197.3%        |


##  实验结论
本项目完整实现了ISTA和BM3D去噪算法，通过系统性实验揭示了两种算法在不同噪声条件下的性能特点:ISTA在低强度高斯噪声下的计算效率优势，BM3D在高噪声场景（特别是椒盐噪声）的性能优越性。为不同应用场景下的算法选择提供了量化依据：实时低噪声处理可优先考虑ISTA，而高质量去噪需求建议采用BM3D。实验结果为指导实际应用中的算法选择提供了可靠依据，代码实现充分考虑了鲁棒性和可扩展性，为后续研究奠定了良好基础。

---
