# **基于PnP-ADMM的自然图像去噪和磁共振重建**

## 实验目标
Plug-and-Play ADMM（PnP-ADMM）方法通过将优化模型与深度学习先验相结合，为图像恢复问题提供了一种兼具可解释性与高性能的解决方案。本实验基于 PnP-ADMM 框架，分别开展了自然图像去噪与磁共振成像（MRI）重建实验。在自然图像去噪任务中，验证了 PnP-ADMM 在高斯噪声下的收敛性和鲁棒性；在 MRI 重建实验中，基于 SIAT 数据集和 TRPA（Trainable Regularization Plug-and-Play Algorithm）模型实现了欠采样图像的高质量重建。结果表明，PnP-ADMM 能有效恢复图像细节并抑制噪声伪影，在 PSNR 与 SSIM 等指标上均显著优于传统方法。



##  快速开始
1. 自然图像去噪

**下载Set14数据集**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   将测试图像放入figs文件夹下

**运行实验**：
   ```python
   python 1.py
   ```

2. 磁共振重建

**数据集准备**：
   ```bash
   https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
   ```
   将测试图像放入figs文件夹下

**运行实验**：

    （1）第一阶段：使用代码准备训练和测试数据（根据数据集存放位置修改代码）并训练降噪器
         本阶段可使用https://github.com/Houruizhi/TRPA中已经训练好的预训练权重
   ```bash
   cd src/data
   ```
   ```python
   python prepare_data.py
   python main_train_denoiser.py --config_file ./options/SIAT_TRPA.yaml
   ```
    （2）第二阶段：运行测试代码 
        
   ```python
   python test_on_dataset_SIAT.py
   ```

    （2）第三阶段：可视化结果

   ```python
   python view_results_SIAT.py
   ```

##  项目结构
```
project
├── PnP-ADMM/
│   ├──figs/                                         # 测试数据集图像
│   │   ├──ppt3.png/
│   │   ├──test_image.png/                                                       
│   ├──results/                                      # 运行结果
│   ├──1.py                                          # 主程序
│   ├──denoiser.pth
│   ├──model.py
│   ├──pnp.py
│   ├──utils.py
├── TRPA/
│   ├──checkpoints/
│   │   ├──SIAT/                                     # 预训练权重                   
│   ├──data/                                        
│   ├──options/                                   
│   ├──src/                             
│   ├──main_train_dnoiser.py                        
│   ├──test_on_dataset_SIAT.py                       # 主程序
│   ├──view_results_SIAT.py                          # 可视化
│   └──README.md                 
```

##  实验结果示例

1）自然图像去噪
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week15/PnP-ADMM/results/1.jpg?raw=true)
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week15/PnP-ADMM/results/2.jpg?raw=true)

2)磁共振重建
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week15/TRPA/results/3.jpg?raw=true)

定量指标对比：

| Type | radial-010 | radial-020 | radia-l030 | spiral-010 | spiral-020 | spiral-030 |
|------|------------|------------|------------|------------|------------|------------|
| PSNR | 30.4794(±1.89) | 34.1175(±1.57) | 35.8674(±1.75) | 27.7233(±2.15) | 33.8454(±1.63) | 36.0629(±1.67) |
| SSIM | 0.8506(±0.03) | 0.9038(±0.02) | 0.9278(±0.02) | 0.7886(±0.04) | 0.8958(±0.02) | 0.9253(±0.01) |


##  实验结论

本实验基于 PnP-ADMM 框架，完成了自然图像去噪与磁共振重建实验。结果表明，PnP-ADMM 能有效结合优化理论与深度先验，在两类任务中均实现高质量恢复。尤其在 MRI 重建中，基于 TRPA 模型的 PnP-ADMM 在保持结构一致性和纹理清晰度方面具有明显优势。

---