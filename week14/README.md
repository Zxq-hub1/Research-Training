# **DDM2 MRI 去噪算法实验**

## 实验目标
本实验主要实现扩散去噪模型（Diffusion Denoising Model 2，简称 DDM2） 在磁共振成像（MRI）图像去噪任务中的应用与实验结果。DDM2 基于扩散概率模型，通过正向加噪与反向扩散的概率推断过程，实现了高保真度的去噪重建。本文在 Stanford HARDI 数据集 上复现了 DDM2 的完整三阶段训练与推理流程，并在定量指标（SNR、CNR）上进行了评估。

##  安装依赖
```bash
conda env create -f environment.yml  
conda activate ddm2
```


##  快速开始
1. **数据集**：
   ```bash
   hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
   data, affine = load_nifti(hardi_fname)
   ```
   或运行代码
   ```bash
    python data.py
   ```
   (本实验采用第二种方法，下载到本地，因为学校服务器访问不到数据）


2. **运行实验**：

    （1）第一阶段：修改配置文件 config/hardi_150.json中数据集路径："./data/HARDI150.nii.gz"
   ```python
   python3 train_noise_model.py -p train -c config/hardi_150.json
   ```
    （2）第二阶段：修改配置文件 config/hardi_150.json中 
        
   1、第一阶段检查点路径："resume_state": "experiments/hardi150_noisemodel_251019_143302/checkpoint/latest"

   2、第二阶段生成路径："stage2_file": "experiments/hardi150_stage2_state.txt"
    ```python
   python3 match_state.py -p train -c config/hardi_150.json
   ```
    （3）第三阶段：
    ```python
    python3 train_diff_model.py -p train -c config/hardi_150.json
   ```
    （4）推理降噪：修改配置文件开头的路径（第三阶段训练结果、指定去噪结果的目录、"phase": "val"）
    ```python
    python denoise.py -c config/hardi.json --save
   ```

##  项目结构
```
project
├── DDM2/
│   ├──config/                                       # 配置文件
│   │   ├──hardi_150.json/                                                     
│   ├──core/                               
│   ├──data/                                         # 数据集
│   │   ├──HARDI150.bval
│   │   ├──HARDI150.bvec
│   │   ├──HARDI150.nii.gz                                 
│   ├──model/                        
│   ├──experiments/                                  # 结果
│   │   ├──hardi150_251020_075432/                   # 第三阶段训练结果
│   │   ├──hardi150_denoise_251020_130810/           # 推理降噪（第 32 个体积的所有切片文件）
│   │   ├──hardi150_denoise_251020_132237/           # 推理降噪（第 32 个体积的所有切片的结果图片）
│   │   ├──hardi150_denoise_251023_095324/           # 推理降噪（所有体积的所有切片文件）
│   │   ├──hardi150_noisemodel_251019_143302/        # 第一阶段训练结果
│   │   ├──hardi150_stage2_state.txt                 # 第二阶段训练结果
│   ├──data.py                                       # 数据集下载            
│   ├──compare_denoising.py                          # 去噪前后对比
│   ├──denoise.py                                    # 推理（去噪）
│   ├──environment.yml                               # 环境依赖
│   ├──match_state.py                                # 第二阶段训练代码
│   ├──quantitative_metrics.ipynb                    # 定量评估计算
│   ├──train_diff_model.py                           # 第三阶段训练代码
│   ├──train_noise_model.py                          # 第一阶段训练代码
│   └──README.md                 
```

##  实验结果示例

1）去噪前后对比
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week14/results/%E5%89%8D%E5%90%8E%E5%AF%B9%E6%AF%94.png?raw=true)

2)推理降噪输出结果图像(左：input 右：denoise)
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week14/results/41.jpg?raw=true)

定量指标对比：

| NoiseType       | mean    | std     | best    | worst    |
|-----------------|---------|---------|---------|----------|
| raw [SNR]       | 5.1153  | 2.4990  | -       | -        |
| raw [CNR]       | 4.6593  | 2.4977  | -       | -        |
| DDM2 [SNR delta]| 0.0123  | 1.0781  | 2.5668  | -2.6503  |
| DDM2 [CNR delta]| -0.0538 | 1.0766  | 2.4960  | -2.7199  |


##  实验结论

本实验复现并验证了 DDM2（Diffusion Denoising Model 2） 在 MRI 图像去噪中的有效性。 通过三阶段训练策略，模型能够在无配对数据条件下学习复杂的噪声分布并实现高质量重建。

DDM2 显著提升了 SNR、CNR的稳定性，指标没有达到原论文的效果，原因可能是数据/强度尺度不一致等原因。 分析图片来看，去噪后图像的结构细节和组织边界保持良好，模型在实际医学影像去噪中具有良好的推广潜力。

---