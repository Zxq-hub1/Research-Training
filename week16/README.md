# **基于Stable Diffusion与LoRA模型的图像生成**

## 实验目标
Stable Diffusion是AI绘画领域的一个核心模型，能够进行文生图（txt2img）和图生图（img2img）等图像生成任务。本实验基于Stable Diffusion实现了SD以及Lora微调图像的生成。通过实验，我们分别完成了SD基础模型的训练以及基于LoRA（Low-Rank Adaptation）的轻量级微调训练，并生成了多张个性化图像。

##  快速开始
1. 数据集准备

* **SD模型数据集**：微信公众号WeThinkIn：宝可梦数据集

* **Lora模型数据集**：微信公众号WeThinkIn：小丑女数据集

* **SD微调训练底模型**：微信公众号WeThinkIn：SD_二次元模型

* **LoRA训练底模型**：微信公众号WeThinkIn：SD_真人模型


2. 使用Stable Diffusion WebUI搭建Stable Diffusion

* **安装Stable Diffusion WebUI框架**：
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   ```
* **修改文件中webui-user.bat(改为自己python3.10存放位置）**：
   ```bash
   set PYTHON="D:\python3.10.6\python.exe"
   set VENV_DIR=venv
   ```
* **启动Stable Diffusion WebUI**：
   ```bash
   .\webui-user.bat
   ```

3. 训练模型
* **SD-Train项目资源包**：微信公众号WeThinkIn：SD-Train

* **安装依赖**：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
   ```
* **检查ccelerate库版本**：  
   ```bash  
     pip install accelerate==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
   ```
* **设置ccelerate库、保存配置文件**：
   ```bash
  accelerate config
  In which compute environment are you running? 
  This machine
  Which type of machine are you using?  
  No distributed training
  Do you wish to use FP16 or BF16 (mixed precision)? 
  fp16   
  accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
   ```
* **数据集制作**：  
（1）使用BLIP自动标注caption：  
    ```bash
    python make_captions.py\  
      "/202521000855/ZXQ/project/SD-Train/datasets/pokemon_data" (训练集的路径)   
      --batch_size=8 （表示每次传入BLIP模型进行前向处理的数据数量）  
      --caption_weights="/202521000855/ZXQ/project/SDTrain/BLIP/model_large_caption.pth" (加载的本地BLIP模型)  
      --beam_search (设置为波束搜索，默认Nucleus采样)  
      --min_length=5 (设置caption标签的最短长度)  
      --max_length=75 (设置caption标签的最长长度)  
      --debug (设置后在BLIP前向处理过程中打印所有的图片路径与caption标签内容)  
      --caption_extension=".caption" （设置caption标签的扩展名）  
      --max_data_loader_n_workers=2 （设置大于等于2，加速数据处理）  
    ```  
  
 （2）使用Waifu Diffusion v1.4模型自动标注tag标签:    
   ```bash
   python tag_images_by_wd14_tagger.py 
     "/202521000855/ZXQ/project/SD-Train/datasets/pokemon_data" （训练集路径）
     --batch_size=8 （每次传入Waifu Diffusion v1.4模型进行前向处理的数据数量）
     --model_dir="../tag_models/wd-v1-4-moat-tagger-v2"(加载本地v1.4模型路径）
     --remove_underscore （开启会将输出tag关键词中的下划线替换为空格）
     --general_threshold=0.35 （设置常规tag关键词的筛选置信度）
     --character_threshold=0.35 （设置特定人物特征tag关键词的筛选置信度）
     --caption_extension=".txt" （设置tag关键词标签的扩展名）
     --max_data_loader_n_workers=2 （设置大于等于2，加速数据处理）
     --debug 
     --undesired_tags=""（设置不需要保存的tag关键词
   ```

（3）训练数据预处理  
* 对生成的.caption和.txt的标注文件进行整合存储为json文件：  
    ```bash
    python ./merge_all_to_metadata.py "../datasets/pokemon_data" "../datasets/pokemon_data/meta_clean.json"
    ```
* 下载SD1.5模型：
    ```bash
    huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./stable-diffusion-v1-5
    ```
* 对数据进行分桶与保存Latent特征、保存图片分辨率信息：
    ```bash
    python ./prepare_buckets_latents.py 
    "../datasets/pokemon_data" 
    "../datasets/pokemon_data/meta_clean.json" 
    "../datasets/pokemon_data/meta_lat.json" 
    "../models/SD/stable-diffusion-v1-5" 
    --min_bucket_reso=256 
    --max_bucket_reso=1024 
    --batch_size 4 
    --max_data_loader_n_workers=2 
    --max_resolution "1024,1024" 
    --mixed_precision="no"
    ```

4. SD模型微调训练参数设置：SD-Train/train_config/config/config_file.toml
5. Lora模型微调训练参数设置：SD-Train/train_config/Lora_config/config_file.toml
6. 修改SD_finetune.sh脚本：SD-Train/SD_finetune.sh（SD模型）
    ```bash
    sh SD_finetune.sh
    ```
7. 修改修改SD_finetune_LoRA.sh脚本：SD-Train/SD_finetune_LoRA.sh（Lora模型）
    ```bash
    sh SD_finetune_LoRA.sh
    ```


##  实验结果示例

1）SD模型出图结果
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week16/output/3.jpg?raw=true)

2)Lora模型样例输出(图片为模型训练过程中的过程sample图片，使用模型生成图片时效果不是很好，还需要调整）
![](https://github.com/Zxq-hub1/Research-Training/blob/main/week16/output/lora.jpg?raw=true)



##  实验结论

通过本实验深入理解了Stable Diffusion的潜在扩散机制及其条件生成能力。  
SD模型在大规模数据集上训练后表现出强大的通用生成能力，而LoRA技术则为我们提供了一种高效的领域自适应方法。 


---
