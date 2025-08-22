# **深度学习图像去噪算法对比**

## 实验目标
本实验对比分析以下图像去噪算法在Set14数据集上的表现：
1. FISTA (快速迭代收缩阈值算法)
2. ADMM (交替方向乘子法) 
3. BM3D (块匹配3D滤波)
4. ISTA (迭代收缩阈值算法)
5. 基于深度学习的DnCNN
6. 面向任意噪声水平的快速灵活图像去噪FFDNet

FFDNet 是首个在“单一网络、单一权重”条件下即可处理连续噪声强度（σ ∈ [0, 75]）和空间变化噪声的深度去噪模型。本实验通过对数据集Set14添加不同程度的高斯噪声，系统对比了传统算法（BM3D）、深度学习模型（DnCNN）、迭代优化方法（ISTA/FISTA）以及FFDNet去噪算法的性能差异。

FFDNet网络架构与流程:

| **阶段**    | **关键操作**                                                |
| --------- | ------------------------------------------------------- |
| **输入预处理** | 将噪声图像下采样为4个子图，与噪声水平图拼接为多通道输入（灰度图通道数为5，彩色图为7）。           |
| **特征提取**  | 通过15层（灰度）或12层（彩色）卷积网络提取特征，每层包含Conv+ReLU（中间层加BatchNorm）。 |
| **输出重建**  | 使用子像素卷积合并子图像，恢复原始分辨率，输出去噪图像。                            |


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
   python FFDNet.py
   ```

3. **查看结果**：
   - 去噪效果图：`ffdnet_results/image`
   - 细节结果：`ffdnet_results/detailed_results.xlsx`
   - 量化指标：`ffdnet_results/summary.csv`


##  参数配置
在`FFDNet.py`中修改实验参数：

| 算法类别  | 代表算法       | 参数设置                      | 
|-------|------------|---------------------------|
| 传统方法  | BM3D       | σ=15、25、50                | 
| 深度学习  | DnCNN      | 预训练模型 (final_model.keras) | 
| 深度学习  | FFDNet     | 预训练模型 (best_model.pth)    | 
| 迭代算法  | ISTA       | λ=15, max_iter=100        | 
| 正则化方法 | FISTA/ADMM | 自适应参数                     | 



##  项目结构
```
project
├── week6/
│   ├──data/
│   │   ├──Test/                                   # 测试数据集
│   │   │   ├── Set14/                           
│   │   ├──Train400/                               # 训练数据集
│   ├──models/                                     # 预训练模型
│   │   ├──FFDNet/  
│   │   │   ├──checkpoints
│   │   │   │   ├──best_model.pth
│   │   │   ├──ffdnet_gray_final.pth
│   ├── ffdnet_results/                            # 输出结果
│   │   ├──image
│   │   │   ├──gaussian                            #高斯噪声
│   │   │   │   ├── 15
│   │   │   │   │   ├── denoising_baboon.png       # 每张图像的去噪结果
│   │   │   │   │   │── denoising_barbara.png             
│   │   │   │   │   │── ...
│   │   │   │   ├── 25
│   │   │   │   │   ├── ...
│   │   │   │   ├── 50
│   │   │   │   │   │── ...
│   │   ├──detailed_results.xlsx                   # 所有图像的详细结果
│   │   ├──summary.csv                             # 汇总统计表格         
│   ├── FFDNet.py                                  # 主程序   
│   ├── train_ffdnet.py                            # 模型训练
│   └── README.md                 
```

##  实验结果示例

1）添加高斯(σ=15、σ=25、σ=50)，并进行去噪:
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week7/results/ppt3.jpg?raw=true)


2）高斯噪声在测试集上的统计量化平均值（PSNR 、SSIM 、Time)

| Algorithm | σ=15 PSNR | σ=15 SSIM | σ=15 Time | σ=25 PSNR | σ=25 SSIM | σ=25 Time | σ=50 PSNR | σ=50 SSIM | σ=50 Time |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| BM3D      | 30.84     | 0.8949    | 0.45      | 28.07     | 0.8239    | 0.44      | 24.29     | 0.6879    | 0.45      |
| ISTA      | 26.60     | 0.7281    | 0.12      | 22.33     | 0.5374    | 0.12      | 16.15     | 0.2912    | 0.12      |
| FISTA     | 28.84     | 0.8427    | 0.06      | 26.49     | 0.7677    | 0.05      | 23.10     | 0.6179    | 0.04      |
| ADMM      | 29.08     | 0.8475    | 0.06      | 26.43     | 0.7556    | 0.06      | 22.60     | 0.5767    | 0.7       |
| DnCNN     | 28.25     | 0.7909    | 0.14      | 27.46     | 0.7954    | 0.12      | 18.11     | 0.3827    | 0.12      |
| FFDNet    | 29.39     | 0.8679    | 0.02      | 27.29     | 0.7974    | 0.02      | 24.21     | 0.6616    | 0.02      |

##  实验结论

从去噪效果来看，传统算法BM3D展现了其作为标杆的强大实力，尤其在低噪声（σ=15）和极高噪声（σ=50）条件下，其PSNR和SSIM指标均最为优异，证明了其在广泛噪声水平下的卓越鲁棒性和恢复能力。

深度学习方法FFDNet显示出最佳的综合性能与实用性。它在所有噪声水平下都保持了接近顶尖的去噪效果，并且其表现极为稳定，没有出现类似DnCNN在高噪声下性能急剧衰退的情况。更为突出的是，FFDNet拥有压倒性的计算效率优势，其处理速度比其他算法快一个数量级，这使其成为处理速度与质量要求并重场景的理想选择。

综上所述，算法的选择取决于具体应用场景的需求：若追求极致的去噪质量且不计较计算成本，BM3D仍是可靠的选择；若需要在效果、速度和稳定性之间取得最佳平衡，FFDNet是综合性能最优的现代解决方案；而对于处理特定噪声水平的图像或追求算法新颖性，DnCNN亦有其用武之地。这些结果清晰地体现了图像去噪领域中传统模型、优化方法与深度学习模型各自的特点与优势。

---
