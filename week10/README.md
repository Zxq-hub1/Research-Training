# **noise2noise去噪算法**

## 实验内容

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
   python example.py
   ```

##  项目结构
```
project
├── week6/
│   ├──noise2noise
│   │   ├──data/
│   │   │   ├──Test/                                   # 测试数据集
│   │   │   │   ├── Set14/                           
│   │   │   ├──DIV2K_train/                            # 训练数据集
│   │   │   ├──DIV2K_valid/                            # 验证数据集
│   │   ├──runs/                                       # 预训练模型
│   │   │   ├──Noise2Noisegaussian  
│   │   │   │   ├──checkpoints
│   │   │   │   │   ├──model_at_epoch_100.dat                            
│   │   ├── results/                                   # 输出结果       
│   │   ├── dataset.py                                    
│   │   ├── train.py                              
│   │   ├── example.py
│   │   ├── models.py
│   │   ├── utils.py
│   │   └── requirements.txt
               
```

##  实验结果示例

1）noise2noise去噪结果
![添加高斯噪声](https://github.com/Zxq-hub1/Research-Training/blob/main/week10/11.jpg?raw=true)

2）高斯噪声在测试集上的统计量化值

| image       | Noisy image 1 | Noisy image 2 | Denoising result | Quality improvement |
|-------------|---------------|---------------|------------------|---------------------|
| zebra       | 20.482        | 20.464        | 26.323           | +5.859              |
| flowers     | 20.590        | 20.583        | 24.511           | +3.928              |
| face        | 21.002        | 21.010        | 30.035           | +9.033              |
| coastguard  | 20.362        | 20.365        | 28.747           | +8.385              |
| bridge      | 20.310        | 20.312        | 27.777           | +7.467              |
| man         | 19.868        | 19.880        | 27.539           | +7.659              |
| ppt3        | 21.760        | 21.736        | 17.342           | -4.418              |
| barbara     | 20.258        | 20.264        | 27.801           | +7.543              |
| lenna       | 20.117        | 20.138        | 23.386           | +3.248              |
##  实验结论

Noise2Noise算法在图像去噪任务中展现出了显著的有效性和实用性。在个别案例中出现了性能退化现象，去噪后PSNR反而下降了4.4dB，这暴露出算法对训练参数和图像内容的敏感性。这种不一致性表明，Noise2Noise方法的有效性在一定程度上依赖于适当的超参数设置和充分的训练收敛。 从技术层面来看，该方法的突出优势在于其自监督学习特性，无需干净的参考图像即可完成训练，极大降低了数据获取的难度。算法通过学习和建立不同噪声实例之间的映射关系，间接掌握了去噪变换的内在规律，为图像处理领域提供了一种新颖且实用的解决方案。 建议在实际应用中，通过增加训练迭代次数、优化网络架构、引入正则化技术和采用自适应学习率策略等措施来进一步提升算法的稳定性和泛化能力。

---