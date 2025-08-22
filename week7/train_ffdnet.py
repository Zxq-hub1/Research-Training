import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from datetime import datetime
import glob
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# FFDNet模型定义
class FFDNet(nn.Module):
    def __init__(self, in_channels=1):
        super(FFDNet, self).__init__()

        # 第一层：卷积 + ReLU
        self.conv1 = nn.Conv2d(in_channels + 1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 中间层：多个卷积 + ReLU
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # 最后层：卷积
        self.conv5 = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, noise_map):
        # 拼接噪声图和输入图像
        x = torch.cat([x, noise_map], dim=1)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)

        return x


# 自定义数据集 - 针对Train400优化
class Train400Dataset(Dataset):
    def __init__(self, data_dir, patch_size=64, augment=True, num_patches_per_image=100):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.augment = augment
        self.num_patches_per_image = num_patches_per_image

        # 加载图像路径
        self.image_paths = glob.glob(os.path.join(data_dir, '*.png')) + \
                           glob.glob(os.path.join(data_dir, '*.jpg')) + \
                           glob.glob(os.path.join(data_dir, '*.bmp')) + \
                           glob.glob(os.path.join(data_dir, '*.tif'))

        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths) * self.num_patches_per_image

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]

        # 加载图像
        try:
            clean_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if clean_img is None:
                raise ValueError(f"Could not read image: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个默认图像
            clean_img = np.ones((self.patch_size, self.patch_size), dtype=np.uint8) * 128

        # 确保图像足够大
        H, W = clean_img.shape
        if H < self.patch_size or W < self.patch_size:
            # 如果图像太小，进行填充
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            H, W = clean_img.shape

        # 随机裁剪patch
        x = random.randint(0, H - self.patch_size)
        y = random.randint(0, W - self.patch_size)
        patch = clean_img[x:x + self.patch_size, y:y + self.patch_size]

        # 数据增强
        if self.augment:
            # 随机翻转
            if random.random() > 0.5:
                patch = np.flipud(patch)
            if random.random() > 0.5:
                patch = np.fliplr(patch)

            # 随机旋转
            rotation = random.choice([0, 1, 2, 3])
            if rotation > 0:
                patch = np.rot90(patch, rotation)

        # 归一化到[0, 1]
        patch = patch.astype(np.float32) / 255.0

        # 随机选择噪声水平 (5-75)
        sigma = random.uniform(5, 75) / 255.0

        # 添加高斯噪声
        noise = np.random.normal(0, sigma, patch.shape).astype(np.float32)
        noisy_patch = patch + noise
        noisy_patch = np.clip(noisy_patch, 0, 1)

        # 转换为tensor
        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
        clean_tensor = torch.from_numpy(patch).unsqueeze(0)
        noise_map = torch.full_like(noisy_tensor, sigma)

        return noisy_tensor, clean_tensor, noise_map


# 验证数据集
class ValidationDataset(Dataset):
    def __init__(self, data_dir, patch_size=64):
        self.data_dir = data_dir
        self.patch_size = patch_size

        # 加载图像路径
        self.image_paths = glob.glob(os.path.join(data_dir, '*.png')) + \
                           glob.glob(os.path.join(data_dir, '*.jpg')) + \
                           glob.glob(os.path.join(data_dir, '*.bmp'))

        if not self.image_paths:
            # 如果没有验证集，使用训练集的前10%作为验证
            train_paths = glob.glob(os.path.join('data/Train400', '*.png')) + \
                          glob.glob(os.path.join('data/Train400', '*.jpg')) + \
                          glob.glob(os.path.join('data/Train400', '*.bmp'))
            self.image_paths = train_paths[:max(1, len(train_paths) // 10)]

        print(f"Using {len(self.image_paths)} images for validation")

        # 预加载验证图像
        self.images = []
        for path in tqdm(self.image_paths, desc="Loading validation images"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 调整图像大小以适应验证
                if max(img.shape) > 256:
                    scale = 256 / max(img.shape)
                    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    img = cv2.resize(img, new_size)
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean_img = self.images[idx]
        H, W = clean_img.shape

        # 如果图像太小，进行填充
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            H, W = clean_img.shape

        # 归一化
        clean_img = clean_img.astype(np.float32) / 255.0

        # 固定噪声水平用于验证
        sigma = 25 / 255.0  # 固定25噪声水平

        # 添加噪声
        noise = np.random.normal(0, sigma, clean_img.shape).astype(np.float32)
        noisy_img = clean_img + noise
        noisy_img = np.clip(noisy_img, 0, 1)

        # 转换为tensor
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
        noise_map = torch.full_like(noisy_tensor, sigma)

        return noisy_tensor, clean_tensor, noise_map


# 验证函数
def validate(model, val_loader, device, epoch):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for noisy, clean, noise_map in tqdm(val_loader, desc="Validating"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            noise_map = noise_map.to(device)

            output = model(noisy, noise_map)
            loss = F.mse_loss(output, clean)

            # 计算PSNR和SSIM
            for i in range(output.size(0)):
                output_np = output[i].squeeze().cpu().numpy()
                clean_np = clean[i].squeeze().cpu().numpy()

                psnr_val = psnr(clean_np, output_np, data_range=1.0)
                ssim_val = ssim(clean_np, output_np, data_range=1.0)

                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1

            total_loss += loss.item() * noisy.size(0)

    return total_loss / num_samples, total_psnr / num_samples, total_ssim / num_samples


# 自定义学习率调度器（兼容旧版本PyTorch）
class CustomReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0

        if mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")

        self.mode_worse = float('inf') if mode == 'min' else -float('inf')

    def step(self, metrics):
        self.last_epoch += 1

        if self.best is None:
            self.best = metrics
            return

        if self.mode == 'min':
            is_better = metrics < self.best - self.threshold
        else:
            is_better = metrics > self.best + self.threshold

        if is_better:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.factor


# 训练函数
def train_ffdnet(train_dir, output_dir, epochs=100, batch_size=32, lr=1e-3, val_dir=None):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # 加载数据
    print("Loading training data...")
    train_dataset = Train400Dataset(train_dir, patch_size=64, augment=True, num_patches_per_image=100)

    if val_dir and os.path.exists(val_dir):
        val_dataset = ValidationDataset(val_dir)
    else:
        print("No validation directory provided, using subset of training data for validation")
        val_dataset = ValidationDataset(train_dir)  # 使用训练集的一部分作为验证

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 初始化模型
    model = FFDNet(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 使用自定义的学习率调度器（兼容旧版本PyTorch）
    scheduler = CustomReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, 'logs', f'training_log_{timestamp}.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_psnr,val_ssim,lr\n')

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnrs = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for noisy, clean, noise_map in progress_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            noise_map = noise_map.to(device)

            optimizer.zero_grad()
            output = model(noisy, noise_map)
            loss = F.mse_loss(output, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss / num_batches:.4f}'
            })

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # 验证
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device, epoch)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录日志
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file, 'a') as f:
            f.write(f'{epoch + 1},{avg_train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{current_lr:.6f}\n')

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}, LR: {current_lr:.6f}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f"Saved best model with val loss: {val_loss:.6f}")

        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch + 1}.pth'))

    # 保存最终模型
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_psnr': val_psnrs[-1]
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'ffdnet_gray_final.pth'))

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs, label='Val PSNR', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('Validation PSNR')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logs', 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation PSNR: {max(val_psnrs):.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FFDNet model on Train400 dataset')
    parser.add_argument('--train_dir', type=str, default='data/Train400', help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation images (optional)')
    parser.add_argument('--output_dir', type=str, default='models/FFDNet', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()

    # 检查训练目录是否存在
    if not os.path.exists(args.train_dir):
        print(f"Training directory {args.train_dir} does not exist!")
        print("Please make sure the Train400 dataset is available at data/Train400/")
        exit(1)

    train_ffdnet(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_dir=args.val_dir
    )