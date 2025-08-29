import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import random


# UNet模型定义
class UNet(nn.Module):
    """UNet去噪模型实现"""

    def __init__(self, in_channels=3):
        super(UNet, self).__init__()

        # 编码器 (下采样)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = self._block(512, 1024)

        # 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)

        # 最终卷积层
        self.final = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))

        # 解码路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


# 自定义数据集类 - 固定噪声强度25
class DIV2KDatasetFixedNoise(Dataset):
    def __init__(self, image_paths, patch_size=128, is_train=True, noise_sigma=25):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.is_train = is_train
        self.noise_sigma = noise_sigma  # 固定噪声强度

    def __len__(self):
        return len(self.image_paths) * 10  # 每个epoch使用10倍的数据增强

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]

        # 读取图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Failed to read image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # 如果读取失败，创建随机图像
            image = np.random.randint(0, 256, (self.patch_size, self.patch_size, 3), dtype=np.uint8)

        # 调整大小或裁剪
        h, w = image.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size))
        elif self.is_train:
            # 训练时随机裁剪
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            image = image[i:i + self.patch_size, j:j + self.patch_size]
        else:
            # 验证时中心裁剪
            i = (h - self.patch_size) // 2
            j = (w - self.patch_size) // 2
            image = image[i:i + self.patch_size, j:j + self.patch_size]

        # 数据增强（仅训练时）
        if self.is_train:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)  # 水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 0)  # 垂直翻转
            if random.random() > 0.5:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # 归一化到 [0, 1]
        clean_image = image.astype(np.float32) / 255.0

        # 添加固定强度的高斯噪声
        noise = np.random.normal(0, self.noise_sigma / 255.0, clean_image.shape)
        noisy_image = clean_image + noise
        noisy_image = np.clip(noisy_image, 0, 1)

        # 转换为张量 (HWC to CHW)
        clean_tensor = torch.from_numpy(clean_image.transpose(2, 0, 1)).float()
        noisy_tensor = torch.from_numpy(noisy_image.transpose(2, 0, 1)).float()

        return noisy_tensor, clean_tensor


# 灰度数据集类 - 固定噪声强度25
class DIV2KDatasetGrayFixedNoise(DIV2KDatasetFixedNoise):
    def __getitem__(self, idx):
        noisy_tensor, clean_tensor = super().__getitem__(idx)

        # 转换为灰度 (使用加权平均)
        noisy_gray = 0.299 * noisy_tensor[0] + 0.587 * noisy_tensor[1] + 0.114 * noisy_tensor[2]
        clean_gray = 0.299 * clean_tensor[0] + 0.587 * clean_tensor[1] + 0.114 * clean_tensor[2]

        return noisy_gray.unsqueeze(0), clean_gray.unsqueeze(0)


# 获取数据路径函数
def get_image_paths(directory):
    """获取指定目录下的所有图像路径"""
    image_paths = glob.glob(os.path.join(directory, '*.png')) + \
                  glob.glob(os.path.join(directory, '*.jpg')) + \
                  glob.glob(os.path.join(directory, '*.bmp')) + \
                  glob.glob(os.path.join(directory, '*.jpeg')) + \
                  glob.glob(os.path.join(directory, '*.tiff'))
    return image_paths


# 训练函数
def train_unet(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4, model_name="unet"):
    """训练UNet模型"""

    # 创建模型保存目录
    os.makedirs('models/UNet/checkpoints', exist_ok=True)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for noisy_imgs, clean_imgs in train_bar:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 前向传播
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_imgs.size(0)
            train_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')

        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_bar:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)

                val_loss += loss.item() * noisy_imgs.size(0)
                val_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, f'models/UNet/checkpoints/best_model_{model_name}.pth')
            print(f'✅ Saved best {model_name} model with validation loss: {val_loss:.6f}')

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f'models/UNet/checkpoints/checkpoint_epoch_{epoch + 1}_{model_name}.pth')

    # 保存最终模型
    final_model_path = f'models/UNet/{model_name}_sigma25.pth'
    torch.save(model.state_dict(), final_model_path)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name} (σ=25)')
    plt.legend()
    plt.savefig(f'models/UNet/training_loss_{model_name}.png')
    plt.close()

    print(f"✅ {model_name} training completed! Model saved to {final_model_path}")

    return model


# 灰度UNet训练 - 固定噪声强度25
def train_unet_gray():
    """训练灰度UNet模型，固定噪声强度25"""
    print("🚀 Starting UNet Gray Training with fixed noise σ=25...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取训练集和验证集路径
    train_data_path = 'data/DIV2K_train'
    valid_data_path = 'data/DIV2K_valid'

    # 获取训练集图像路径
    train_paths = get_image_paths(train_data_path)
    if not train_paths:
        print(f"❌ No images found in {train_data_path}/")
        print("Please make sure the DIV2K training dataset is downloaded and placed in the correct directory.")
        return

    # 获取验证集图像路径
    val_paths = get_image_paths(valid_data_path)
    if not val_paths:
        print(f"❌ No images found in {valid_data_path}/")
        print("Please make sure the DIV2K validation dataset is downloaded and placed in the correct directory.")
        return

    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images")

    # 创建数据集 - 固定噪声强度25
    train_dataset = DIV2KDatasetGrayFixedNoise(train_paths, patch_size=128, is_train=True, noise_sigma=25)
    val_dataset = DIV2KDatasetGrayFixedNoise(val_paths, patch_size=128, is_train=False, noise_sigma=25)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)

    # 创建模型
    model = UNet(in_channels=1).to(device)

    # 训练模型
    trained_model = train_unet(model, train_loader, val_loader, device, num_epochs=30, lr=1e-4, model_name="unet_gray")

    return trained_model


# 彩色UNet训练 - 固定噪声强度25
def train_unet_color():
    """训练彩色UNet模型，固定噪声强度25"""
    print("🚀 Starting UNet Color Training with fixed noise σ=25...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取训练集和验证集路径
    train_data_path = 'data/DIV2K_train'
    valid_data_path = 'data/DIV2K_valid'

    # 获取训练集图像路径
    train_paths = get_image_paths(train_data_path)
    if not train_paths:
        print(f"❌ No images found in {train_data_path}/")
        return

    # 获取验证集图像路径
    val_paths = get_image_paths(valid_data_path)
    if not val_paths:
        print(f"❌ No images found in {valid_data_path}/")
        return

    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images")

    # 创建数据集 - 固定噪声强度25
    train_dataset = DIV2KDatasetFixedNoise(train_paths, patch_size=128, is_train=True, noise_sigma=25)
    val_dataset = DIV2KDatasetFixedNoise(val_paths, patch_size=128, is_train=False, noise_sigma=25)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    # 创建模型
    model = UNet(in_channels=3).to(device)

    # 训练模型
    trained_model = train_unet(model, train_loader, val_loader, device, num_epochs=30, lr=1e-4, model_name="unet_color")

    return trained_model


# 测试函数 - 验证固定噪声强度25的效果
def test_unet_model():
    """测试训练好的UNet模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试灰度模型
    gray_model_path = 'models/UNet/unet_gray_sigma25.pth'
    if os.path.exists(gray_model_path):
        model_gray = UNet(in_channels=1).to(device)
        model_gray.load_state_dict(torch.load(gray_model_path, map_location=device))
        model_gray.eval()

        # 使用验证集图像进行测试
        valid_data_path = 'data/DIV2K_valid'
        val_paths = get_image_paths(valid_data_path)

        if val_paths:
            # 测试第一张验证集图像
            test_img_path = val_paths[0]
            image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # 调整大小
                image = cv2.resize(image, (256, 256))
                clean_img = image.astype(np.float32) / 255.0

                # 添加固定噪声强度25
                noise = np.random.normal(0, 25 / 255.0, clean_img.shape)
                noisy_img = clean_img + noise
                noisy_img = np.clip(noisy_img, 0, 1)

                # 转换为张量并预测
                noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    denoised_tensor = model_gray(noisy_tensor)

                denoised_img = denoised_tensor.squeeze().cpu().numpy()

                print("✅ Gray model test completed!")
                print(f"Test image: {os.path.basename(test_img_path)}")
                print(f"Noisy PSNR: {10 * np.log10(1 / np.mean((clean_img - noisy_img) ** 2)):.2f} dB")
                print(f"Denoised PSNR: {10 * np.log10(1 / np.mean((clean_img - denoised_img) ** 2)):.2f} dB")

    print("✅ Testing completed!")


if __name__ == "__main__":
    # 创建模型目录
    os.makedirs('models/UNet/checkpoints', exist_ok=True)

    # 训练灰度UNet - 固定噪声强度25
    train_unet_gray()

    # 训练彩色UNet - 固定噪声强度25
    train_unet_color()

    # 测试模型
    test_unet_model()