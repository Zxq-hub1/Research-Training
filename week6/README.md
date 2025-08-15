# **深度学习图像去噪算法对比**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)
5. 基于深度学习的DnCNN

对比传统算法（BM3D）与基于深度学习的DnCNN在图像去噪任务中的性能差异,分析迭代优化算法（ISTA/FISTA）与TV正则化方法的效果,评估不同噪声类型（高斯/椒盐）对算法鲁棒性的影响。

本文针对高斯噪声与椒盐噪声污染下的图像恢复问题，系统对比了传统算法（BM3D）、深度学习模型（DnCNN）及迭代优化方法（ISTA/FISTA）的性能差异。根据结果得知，深度学习DnCNN应均优于传统方法，且GPU加速下处理速度会提升；TV正则化方法在强噪声场景下展现更好的边缘保持能力。本研究为不同噪声环境下的算法选择提供了实证依据。

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
   python new.py
   ```

3. **查看结果**：
   - 去噪效果图：`results2/image`
   - 细节结果：`results2/detailed_results.xlsx`
   - 量化指标：`results2/summary.csv`


##  参数配置
在`new.py`中修改实验参数：

| 算法类别  | 代表算法 | 参数设置                      | 
|-------|------|---------------------------|
| 传统方法  | BM3D | σ=15、25、50                | 
| 深度学习  | DnCNN | 预训练模型 (final_model.keras) | 
| 迭代算法  | ISTA | λ=15, max_iter=100        | 
| 正则化方法 | FISTA/ADMM | 自适应参数                     | 



##  项目结构
```
project
├── week6/
│   ├──data/
│   │   ├──Test/                                   # 测试数据集
│   │   │   ├── Set14/                           
│   │   │   ├── Set12/
│   │   │   ├── Set68/
│   │   ├──Train400/                               # 训练数据集
│   ├──models/                                     # 预训练模型
│   │   ├──DnCNN_sigma25/  
│   │   │   ├──final_model.keras
│   │   │   ├──final_model.h5
│   ├── results2/                                  # 输出结果
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
│   │   ├──detailed_results.xlsx                  # 所有图像的详细结果
│   │   ├──summary.csv                            # 汇总统计表格         
│   ├── new.py                                    # 主程序   
│   ├── main_test.py                              # 模型测试
│   ├── main_train.py                             # 模型训练
│   ├── data_generator.py                         # 图像块生成器
│   └── README.md                 
```

##  实验结果示例

1)添加高斯(σ=15)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week6/results/15_denoising_ppt3.png?raw=true)

2)添加高斯噪声(σ=25)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week6/results/25_denoising_ppt3.png?raw=true)

3)添加高斯(σ=50)，并进行去噪:
![添加椒盐噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week6/results/50_denoising_ppt3.png?raw=true)

3)高斯噪声在测试集上的统计量化平均值（PSNR 、SSIM 、Time)

| Algorithm | σ = 15 PSNR | σ = 15 SSIM | σ = 15 Time | σ = 25 PSNR | σ = 25 SSIM | σ = 25 Time | σ = 50 PSNR | σ = 50 SSIM | σ = 50 Time |
|-----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| BM3D      | 30.84      | 0.8949     | 0.43       | 28.07      | 0.8239     | 0.44       | 24.29      | 0.6879     | 0.44       |
| ISTA      | 26.60      | 0.7281     | 0.11       | 22.33      | 0.5374     | 0.11       | 16.15      | 0.2912     | 0.11       |
| FISTA     | 28.84      | 0.8427     | 0.06       | 26.49      | 0.7677     | 0.05       | 23.10      | 0.6179     | 0.04       |
| ADMM      | 29.08      | 0.8475     | 0.06       | 26.43      | 0.7556     | 0.06       | 22.60      | 0.5374     | 0.11       |
| DnCNN     | 28.25      | 0.7909     | 0.13       | 27.16      | 0.7954     | 0.12       | 18.11      | 0.3827     | 0.12       |


##  实验结论

本次实验针对不同强度的高斯噪声（σ=15、25、50）场景，系统评估了BM3D、DnCNN、ISTA、FISTA和ADMM五种主流去噪算法的性能表现。

实验结果表明，BM3D算法在各类噪声环境下均展现出卓越的去噪能力，特别是在低噪声（σ=15）条件下，显著优于其他对比算法。随着噪声强度的增加，虽然所有算法的性能均出现不同程度下降，但BM3D仍能保持相对优势。

FISTA和ADMM算法表现出较好的鲁棒性，在中低噪声水平下性能接近BM3D，且计算效率较高，可作为实时处理等对计算资源敏感场景的备选方案。

相比之下，DnCNN算法在低噪声条件下表现尚可，但在高噪声环境下性能急剧恶化，稳定性不足。

ISTA算法在所有测试场景中均表现最差，特别是在高噪声条件下PSNR仅16.88dB，应用价值有限。

综合来看，在实际应用中应根据具体需求选择合适的去噪算法：对于图像质量要求严格的场景推荐采用BM3D算法；在需要平衡处理速度和去噪效果的场合可考虑FISTA或ADMM算法；而DnCNN和ISTA算法则仅适用于特定低噪声场景或对图像质量要求不高的应用。

---