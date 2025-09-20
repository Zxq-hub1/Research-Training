import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
import pywt
import time
import os
import bm3d
from scipy.fftpack import dct, idct
import glob
import pandas as pd
from tqdm import tqdm
import scipy.fft
from skimage.util import random_noise
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input, Conv2D, Activation
from keras.models import Model
from keras.saving import register_keras_serializable
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def psnr_color(img1, img2, data_range=255):
    """彩色/灰度通用的 PSNR，兼容旧版 skimage"""
    if img1.ndim == 3 and img1.shape[-1] == 3:          # 彩色
        return np.mean([_psnr(img1[..., c], img2[..., c], data_range=data_range)
                        for c in range(3)])
    else:                                               # 灰度
        return _psnr(img1, img2, data_range=data_range)

def ssim_color(img1, img2, data_range=255, win_size=7):
    """彩色/灰度 SSIM，兼容极旧 skimage"""
    if img1.ndim == 3 and img1.shape[-1] == 3:          # 彩色
        return np.mean([_ssim(img1[..., c], img2[..., c], data_range=data_range, win_size=win_size)
            for c in range(3)
        ])
    else:                                               # 灰度
        return _ssim(img1, img2, data_range=data_range, win_size=win_size)

# 1. 定义并注册自定义损失函数
@register_keras_serializable(package="Custom")
def sum_squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))


# 2. 加载模型时指定自定义对象
try:
    model = tf.keras.models.load_model(
        '/202521000855/ZXQ/project/week6/models/DnCNN_sigma25/final_model.keras',
        custom_objects={'sum_squared_error': sum_squared_error}
    )
except Exception as e:
    print(f"⚠️  DnCNN 权重加载失败，已跳过该算法。详情：{e}")
    model = None


# FFDNet模型定义
class FFDNet(nn.Module):
    """FFDNet模型实现"""

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


# U-Net模型定义 - 修改为支持彩色图像
class UNet(nn.Module):
    """U-Net去噪模型实现（彩色版本）"""

    def __init__(self, in_channels=3, out_channels=3):
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
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

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


def load_ffdnet_model(device='cpu'):
    """加载预训练的FFDNet模型，支持从完整checkpoint中提取模型权重"""
    model = FFDNet(in_channels=1)

    # 尝试从多个位置加载预训练权重
    possible_paths = [
        os.path.join('/202521000855/ZXQ/project/week6/models/FFDNet', 'ffdnet_gray.pth'),
        os.path.join('/202521000855/ZXQ/project/week6/models/FFDNet/checkpoints', 'best_model.pth'),
        os.path.join('/202521000855/ZXQ/project/week6/models/FFDNet', 'ffdnet_gray_final.pth'),
        'ffdnet_gray_final.pth'
    ]

    model_loaded = False
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                # 如果是完整的checkpoint，提取模型权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # 移除可能的 'module.' 前缀（如果是DataParallel训练的）
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict)
                print(f"✅ 成功加载 FFDNet 模型权重：{model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ 加载模型失败：{model_path}，错误：{e}")
                continue

    if not model_loaded:
        print("⚠️ 未找到可用的FFDNet权重，使用随机初始化的模型。")

    model.to(device)
    model.eval()
    return model


def load_unet_model(device='cpu'):
    """加载预训练的U-Net模型（彩色版本）"""
    model = UNet(in_channels=3, out_channels=3)

    # 尝试从多个位置加载预训练权重
    possible_paths = [
        os.path.join('/202521000855/ZXQ/project/week6/models/UNet', 'unet_color_sigma25.pth'),
        os.path.join('/202521000855/ZXQ/project/week6/models/UNet/checkpoints', 'best_model_unet_color.pth'),
        os.path.join('/202521000855/ZXQ/project/week6/models/UNet', 'unet_color_final.pth'),
        'unet_color_sigma25.pth'
    ]

    model_loaded = False
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                # 如果是完整的checkpoint，提取模型权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # 移除可能的 'module.' 前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict)
                print(f"✅ 成功加载 U-Net 彩色模型权重：{model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ 加载彩色U-Net模型失败：{model_path}，错误：{e}")
                continue

    if not model_loaded:
        print("⚠️ 未找到可用的彩色U-Net权重，使用随机初始化的模型。")

    model.to(device)
    model.eval()
    return model


# 添加DnCNN模型加载函数（保持灰度，因为DnCNN通常是灰度模型）
def load_dncnn_model():
    """加载预训练的DnCNN模型（灰度）"""
    model_path = os.path.join('/202521000855/ZXQ/project/week6/models/DnCNN_sigma25', 'final_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DnCNN model not found at {model_path}. Please download the pretrained model.")
    return load_model(model_path)


# 添加Neighbor2Neighbor模型定义
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1

    # 根据当前设备创建对应的生成器
    if torch.cuda.is_available():
        g_generator = torch.Generator(device="cuda")
    else:
        g_generator = torch.Generator(device="cpu")

    g_generator.manual_seed(operation_seed_counter)
    return g_generator


operation_seed_counter = 0          # 全局计数器
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


# 1. 噪声 augment
class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            # 修改这里：使用与输入x相同的设备
            noise = torch.zeros(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            # 修改这里：使用与输入x相同的设备
            noise = torch.zeros(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


# 2. mask 生成
def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


# 3. 数据集（单张图用）
class SingleImageDataset(Dataset):
    def __init__(self, img_np, patch_size=256):
        """
        img_np: (H,W,3) uint8
        """
        self.patch_size = patch_size
        self.img = img_np

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        im = self.img
        H, W, _ = im.shape
        # 如果图太小就 pad
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            im = np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')
            H, W, _ = im.shape

        # 随机 crop
        xx = np.random.randint(0, H - self.patch_size + 1)
        yy = np.random.randint(0, W - self.patch_size + 1)
        patch = im[xx:xx + self.patch_size, yy:yy + self.patch_size, :]

        # 转 Tensor
        patch = transforms.ToTensor()(patch)  # (3,H,W)
        return patch


# 全局加载DnCNN模型，避免重复加载
try:
    dncnn_model = load_dncnn_model()
except Exception as e:
    print(f"Warning: Could not load DnCNN model. DnCNN will be disabled. Error: {e}")
    dncnn_model = None

# 全局加载模型，避免重复加载
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ffdnet_model = load_ffdnet_model(device=device)
    unet_model = load_unet_model(device=device)
except Exception as e:
    print(f"Warning: Could not load models. Error: {e}")
    ffdnet_model = None
    unet_model = None



def add_gaussian_noise_color(image, sigma=25):
    """添加高斯噪声（彩色图像）"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def soft_threshold(x, threshold):
    """软阈值函数，处理复数情况"""
    if np.iscomplexobj(x):
        magnitude = np.abs(x)
        phase = np.angle(x)
        thresholded = np.maximum(magnitude - threshold, 0)
        return thresholded * np.exp(1j * phase)
    else:
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def ista_l1_color(noisy_img, clean_img, lambda_=15, max_iter=100, tol=1e-6, wavelet='db8', level=3):
    """ISTA算法（L1小波正则化，彩色版本）"""
    noisy = noisy_img.astype(np.float32)
    x = noisy.copy()
    last_x = x.copy()
    psnrs = []
    ssims = []

    # 对每个通道分别处理
    for c in range(3):
        channel_noisy = noisy[:, :, c]
        channel_x = x[:, :, c].copy()
        channel_last_x = last_x[:, :, c].copy()

        # 预计算小波分解结构
        coeffs = pywt.wavedec2(channel_noisy, wavelet, level=level)
        coeffs_shape = [coeffs[0].shape]
        for j in range(1, len(coeffs)):
            coeffs_shape.append(tuple(c.shape for c in coeffs[j]))

        # 学习率设置
        L = 4.0  # Lipschitz常数估计

        for i in range(max_iter):
            # 小波分解
            coeffs = pywt.wavedec2(channel_x, wavelet, level=level)

            # 阈值处理高频系数
            coeffs_thresh = [coeffs[0]]  # 保留低频分量

            for j in range(1, len(coeffs)):
                # 对每个方向的高频系数进行阈值处理
                threshold = lambda_ * (0.5 ** j)  # 尺度相关的阈值
                cH, cV, cD = [soft_threshold(c, threshold) for c in coeffs[j]]
                coeffs_thresh.append((cH, cV, cD))

            # 小波重构
            x_new = pywt.waverec2(coeffs_thresh, wavelet)

            # 数据保真项更新
            x_new = x_new + (channel_noisy - x_new) / L

            # 投影到有效范围
            x_new = np.clip(x_new, 0, 255)

            # 确保数值稳定性
            x_new = np.nan_to_num(x_new, nan=0.0, posinf=255, neginf=0)

            # 检查收敛性
            if np.linalg.norm(x_new - channel_last_x) / (np.linalg.norm(channel_last_x) + 1e-10) < tol:
                break

            channel_last_x = x_new.copy()
            channel_x = x_new.copy()

        x[:, :, c] = channel_x

    # 计算最终评估指标
    denoised_uint8 = x.astype(np.uint8)
    psnr_val = psnr_color(clean_img, denoised_uint8, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_uint8, data_range=255)
    psnrs.append(psnr_val)
    ssims.append(ssim_val)

    return denoised_uint8, psnrs, ssims


def fista_tv_color(noisy_img, clean_img, lambda_=None, max_iter=100, tol=1e-4):
    """改进的 FISTA-TV 去噪算法（彩色版本）"""
    noisy = noisy_img.astype(np.float32) / 255.0
    h, w, c = noisy.shape

    # 对每个通道分别处理
    x = np.zeros_like(noisy)
    for channel in range(3):
        channel_noisy = noisy[:, :, channel]

        # 自适应 λ 设置
        if lambda_ is None:
            sigma_est = estimate_sigma(noisy_img[:, :, channel], average_sigmas=True)
            lambda_channel = 0.8 * sigma_est / 255.0
        else:
            lambda_channel = lambda_

        # 预去噪 warm-start
        x_channel = denoise_tv_chambolle(channel_noisy, weight=lambda_channel, max_num_iter=100)
        y = x_channel.copy()
        t = 1.0
        last_x = x_channel.copy()

        # 估计 Lipschitz 常数 L
        L = 8.0  # TV 算子的最大特征值约为 8（经验值）

        def grad_tv(img):
            return denoise_tv_chambolle(img, weight=lambda_channel / L, max_num_iter=1)

        for i in range(max_iter):
            # 梯度下降步
            grad_step = grad_tv(y)
            x_new = y + (channel_noisy - y) / L

            # TV 去噪步
            x_new = denoise_tv_chambolle(x_new, weight=lambda_channel / L, max_num_iter=5)

            # FISTA 加速
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y_new = x_new + ((t - 1) / t_new) * (x_new - last_x)

            # 投影到 [0,1]
            y_new = np.clip(y_new, 0, 1)

            # 收敛检查
            if np.linalg.norm(x_new - last_x) / (np.linalg.norm(last_x) + 1e-8) < tol:
                break

            last_x = x_new.copy()
            x_channel = x_new.copy()
            y = y_new.copy()
            t = t_new

        x[:, :, channel] = x_channel

    # 计算评估指标
    denoised_uint8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    psnr_val = psnr_color(clean_img, denoised_uint8, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_uint8, data_range=255)

    return denoised_uint8, [psnr_val], [ssim_val]


def admm_tv_color(noisy_img, clean_img, lambda_=None, rho=None, max_iter=200, tol=1e-4, min_iter=20):
    """改进的ADMM-TV去噪算法（彩色版本）"""
    h, w, c = noisy_img.shape
    x = noisy_img.astype(np.float32)

    # 对每个通道分别处理
    for channel in range(3):
        channel_noisy = noisy_img[:, :, channel]

        # 自适应参数设置
        if lambda_ is None:
            sigma_est = estimate_sigma(channel_noisy, average_sigmas=True)
            lambda_channel = 0.5 * sigma_est
        else:
            lambda_channel = lambda_

        if rho is None:
            rho_channel = 1.0
        else:
            rho_channel = rho

        # 初始化变量
        z_x = np.zeros_like(channel_noisy, dtype=np.float32)
        z_y = np.zeros_like(channel_noisy, dtype=np.float32)
        u_x = np.zeros_like(channel_noisy, dtype=np.float32)
        u_y = np.zeros_like(channel_noisy, dtype=np.float32)
        last_x = x[:, :, channel].copy()
        eps = 1e-6

        # 构造拉普拉斯算子的FFT表示
        laplacian = np.zeros((h, w), dtype=np.float32)
        laplacian[0, 0] = 4.0
        if w > 1:
            laplacian[0, 1] = -1.0
            laplacian[0, -1] = -1.0
        if h > 1:
            laplacian[1, 0] = -1.0
            laplacian[-1, 0] = -1.0
        laplacian_fft = scipy.fft.fft2(laplacian)

        # 定义梯度算子（循环边界）
        def grad(img):
            gx = np.roll(img, -1, axis=1) - img
            gy = np.roll(img, -1, axis=0) - img
            return gx, gy

        # 定义散度算子（循环边界）
        def div(gx, gy):
            gx_roll = np.roll(gx, 1, axis=1) - gx
            gy_roll = np.roll(gy, 1, axis=0) - gy
            return gx_roll + gy_roll

        for i in range(max_iter):
            # x-更新：求解线性系统 (I - ρΔ)x = b
            div_zu = div(z_x - u_x, z_y - u_y)
            b = channel_noisy + rho_channel * div_zu
            b_fft = scipy.fft.fft2(b)
            x_fft = b_fft / (1 + rho_channel * (laplacian_fft + eps))
            x_new = np.real(scipy.fft.ifft2(x_fft))
            x_new = np.clip(x_new, 0, 255)

            # z-更新：软阈值
            dx, dy = grad(x_new)
            z_x_new = soft_threshold(dx + u_x, lambda_channel / rho_channel)
            z_y_new = soft_threshold(dy + u_y, lambda_channel / rho_channel)

            # u-更新：对偶变量
            u_x += dx - z_x_new
            u_y += dy - z_y_new

            # 检查收敛性
            if i >= min_iter and np.linalg.norm(x_new - last_x) / (np.linalg.norm(last_x) + eps) < tol:
                break

            last_x = x_new.copy()
            x[:, :, channel] = x_new
            z_x, z_y = z_x_new, z_y_new

    # 计算评估指标
    denoised_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    psnr_val = psnr_color(clean_img, denoised_uint8, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_uint8, data_range=255)

    return denoised_uint8, [psnr_val], [ssim_val]


def bm3d_denoise_color(noisy_img, clean_img, sigma):
    """BM3D去噪算法（彩色版本）"""
    start_time = time.time()

    # 对每个通道分别处理
    denoised_channels = []
    for c in range(3):
        channel_denoised = bm3d.bm3d(noisy_img[:, :, c], sigma)
        denoised_channels.append(channel_denoised)

    denoised = np.stack(denoised_channels, axis=-1)
    elapsed_time = time.time() - start_time

    psnr_val = psnr_color(clean_img, denoised, data_range=255)
    ssim_val = ssim_color(clean_img, denoised, data_range=255)

    return denoised, psnr_val, ssim_val, elapsed_time


def dncnn_denoise_color(noisy_img, clean_img):
    """DnCNN去噪算法（保持灰度处理，因为DnCNN通常是灰度模型）"""
    if dncnn_model is None:
        raise RuntimeError("DnCNN model is not available")

    start_time = time.time()

    # 转换为YUV颜色空间，只对Y通道处理
    noisy_yuv = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2YUV)
    y_channel = noisy_yuv[:, :, 0].astype(np.float32) / 255.0

    # 预处理图像
    noisy_input = np.expand_dims(y_channel, axis=[0, -1])  # 添加batch和channel维度

    # 预测
    denoised_output = dncnn_model.predict(noisy_input, verbose=0)

    # 后处理
    denoised_y = np.clip(denoised_output[0, ..., 0] * 255, 0, 255).astype(np.uint8)

    # 合并通道
    denoised_yuv = noisy_yuv.copy()
    denoised_yuv[:, :, 0] = denoised_y
    denoised_img = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2BGR)

    elapsed_time = time.time() - start_time

    # 计算评估指标
    psnr_val = psnr_color(clean_img, denoised_img, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_img, data_range=255)

    return denoised_img, psnr_val, ssim_val, elapsed_time


def ffdnet_denoise(noisy_img, clean_img, sigma=25):
    """FFDNet去噪算法"""
    if ffdnet_model is None:
        raise RuntimeError("FFDNet model is not available")

    start_time = time.time()

    # 转换为PyTorch张量
    noisy_tensor = torch.from_numpy(noisy_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    # 创建噪声图（与输入图像相同大小的常数噪声水平图）
    noise_map = torch.full_like(noisy_tensor, sigma / 255.0)

    # 移动到设备
    noisy_tensor = noisy_tensor.to(device)
    noise_map = noise_map.to(device)

    # 去噪
    with torch.no_grad():
        denoised_tensor = ffdnet_model(noisy_tensor, noise_map)

    # 转换回numpy数组
    denoised_img = denoised_tensor.squeeze().cpu().numpy()
    denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)

    elapsed_time = time.time() - start_time

    # 计算评估指标
    psnr_val = psnr_color(clean_img, denoised_img, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_img, data_range=255)

    return denoised_img, psnr_val, ssim_val, elapsed_time


def unet_denoise_color(noisy_img, clean_img):
    """U-Net去噪算法（彩色版本）"""
    if unet_model is None:
        raise RuntimeError("U-Net model is not available")

    start_time = time.time()

    # 转换为PyTorch张量
    noisy_tensor = torch.from_numpy(noisy_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    # 移动到设备
    noisy_tensor = noisy_tensor.to(device)

    # 去噪
    with torch.no_grad():
        denoised_tensor = unet_model(noisy_tensor)

    # 转换回numpy数组
    denoised_img = denoised_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)

    elapsed_time = time.time() - start_time

    # 计算评估指标
    psnr_val = psnr_color(clean_img, denoised_img, data_range=255)
    ssim_val = ssim_color(clean_img, denoised_img, data_range=255)

    return denoised_img, psnr_val, ssim_val, elapsed_time

# ====================== Noise2Noise 彩色推理 ======================
def noise2noise_denoise_color(noisy_img_np, clean_img_np, model_weight_path, patch_size=256, batch_size=1, device='cuda'):
    """
    纯推理：输入 uint8 彩色图 -> 输出 uint8 彩色去噪图
    """
    from models import UNet
    import torch
    import time

    start = time.time()

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 1. 建网络（彩色）
    model = UNet(in_channels=3, out_channels=3)

    # 加载模型权重
    ckpt = torch.load(model_weight_path, map_location='cpu', weights_only=False)

    # 兼容不同的保存格式
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    # 移除可能的 'module.' 前缀
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    model.load_state_dict(new_state, strict=True)
    model.eval().to(device)

    # 2. 准备输入数据
    H, W, _ = noisy_img_np.shape

    # 如果需要，进行填充
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        noisy_img_np = np.pad(noisy_img_np, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')

    # 3. 转换为Tensor并归一化
    img_tensor = torch.from_numpy(noisy_img_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 4. 生成两个不同的噪声版本作为源图像
    sigma = 25 / 255.0
    noise1 = torch.randn_like(img_tensor) * sigma
    noise2 = torch.randn_like(img_tensor) * sigma

    src1 = (img_tensor + noise1).clamp(0, 1)
    src2 = (img_tensor + noise2).clamp(0, 1)

    # 5. 使用第一个源图像进行去噪
    with torch.no_grad():
        pred = model(src1).clamp(0, 1)

    # 6. 转换回numpy并去除填充
    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)

    src1_np = src1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    src1_np = np.clip(src1_np * 255, 0, 255).astype(np.uint8)

    src2_np = src2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    src2_np = np.clip(src2_np * 255, 0, 255).astype(np.uint8)

    if pad_h or pad_w:
        pred_np = pred_np[:H, :W, :]
        src1_np = src1_np[:H, :W, :]
        src2_np = src2_np[:H, :W, :]

    # 7. 计算指标
    psnr_val = psnr_color(clean_img_np, pred_np, data_range=255)
    ssim_val = ssim_color(clean_img_np, pred_np, data_range=255)

    # 计算源图像的指标
    src1_psnr = psnr_color(clean_img_np, src1_np, data_range=255)
    src1_ssim = ssim_color(clean_img_np, src1_np, data_range=255)
    src2_psnr = psnr_color(clean_img_np, src2_np, data_range=255)
    src2_ssim = ssim_color(clean_img_np, src2_np, data_range=255)

    elapsed = time.time() - start

    return pred_np, psnr_val, ssim_val, elapsed, src1_np, src2_np, src1_psnr, src1_ssim, src2_psnr, src2_ssim


# Neighbor2Neighbor 彩色推理函数
def n2n_denoise_color(noisy_img_np, clean_img_np, model_weight_path,
                      patch_size=256, batch_size=1, device='cuda'):
    """
    noisy_img_np : (H,W,3)  uint8  带噪图
    clean_img_np : (H,W,3)  uint8  干净参考图（仅算指标）
    model_weight_path : 1.py 训练保存的 **完整 checkpoint** 或 **权重文件**
    return : 去噪后 (H,W,3) uint8  +  psnr  +  ssim
    """
    from arch_unet import UNet
    import torch
    import cv2
    import time

    start = time.time()

    # 1. 建网络（彩色）
    model = UNet(in_nc=3, out_nc=3, n_feature=48)
    if os.path.isfile(model_weight_path):
        ckpt = torch.load(model_weight_path, map_location='cpu')
        # 兼容两种保存方式
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            state = ckpt
        # 去 DataParallel 前缀
        from collections import OrderedDict
        new_state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state.items())
        model.load_state_dict(new_state, strict=True)
        print(f'=> loaded Neighbor2Neighbor model: {model_weight_path}')
    else:
        raise FileNotFoundError(f'pretrain weight not found: {model_weight_path}')
    model.eval().to(device)

    # 2. 准备数据（整图滑动窗口或单张 crop）
    H, W, _ = noisy_img_np.shape
    # 如果图太小就 pad 到 patch_size 倍数
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        noisy_img_np = np.pad(noisy_img_np, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')

    # 3. 切块推理（overlap 可省，为简单用不 overlap）
    # 3. 切块推理 ── 保证输入 0~1
    patches = []
    coords = []
    for y in range(0, noisy_img_np.shape[0], patch_size):
        for x in range(0, noisy_img_np.shape[1], patch_size):
            patch = noisy_img_np[y:y + patch_size, x:x + patch_size, :]
            patch = patch.astype(np.float32) / 255.0  # ✅ 归一化
            patches.append(torch.from_numpy(patch).permute(2, 0, 1))  # (3,H,W)
            coords.append((y, x))
    patches = torch.stack(patches)  # (N,3,H,W)

    # 4. 逐 batch 推理
    denoised_patches = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size].to(device)
            out = model(batch)
            denoised_patches.append(out.cpu())
    denoised_patches = torch.cat(denoised_patches)  # (N,3,pH,pW)

    # 5. 拼回整图 + 反归一化
    denoised_img = np.zeros_like(noisy_img_np, dtype=np.float32)
    count = np.zeros_like(noisy_img_np, dtype=np.float32)
    for (y, x), patch in zip(coords, denoised_patches):
        patch = patch.permute(1, 2, 0).numpy()  # (H,W,3)
        patch = np.clip(patch * 255, 0, 255)  # ✅ 反归一化
        denoised_img[y:y + patch_size, x:x + patch_size, :] += patch
        count[y:y + patch_size, x:x + patch_size, :] += 1
    denoised_img /= count
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    # 6. 去掉 pad
    if pad_h or pad_w:
        denoised_img = denoised_img[:H, :W, :]

    # 7. 计算指标
    psnr_val = psnr_color(clean_img_np, denoised_img, data_range=255)
    ssim_val = ssim_color(clean_img_np, denoised_img, data_range=255)
    elapsed = time.time() - start

    return denoised_img, psnr_val, ssim_val, elapsed
    print('model out range:', out.min().item(), out.max().item())  # 应该 0~1
    print('after *255     :', (out * 255).min().item(), (out * 255).max().item())


def process_image_color(image_path, noise_type='gaussian', noise_param=25, max_iter=100, resize=True):
    """处理单个彩色图像"""
    clean_img = cv2.imread(image_path)
    if clean_img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None

    if resize:
        clean_img = cv2.resize(clean_img, (256, 256))

    # 确保是numpy数组
    if not isinstance(clean_img, np.ndarray):
        clean_img = np.array(clean_img)

    # 添加噪声
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise_color(clean_img, noise_param)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # 确保噪声图像也是numpy数组
    if not isinstance(noisy_img, np.ndarray):
        noisy_img = np.array(noisy_img)

    results = {}
    times = {}

    # BM3D
    start_time = time.time()
    bm3d_denoised, bm3d_psnr, bm3d_ssim, bm3d_time = bm3d_denoise_color(
        noisy_img, clean_img, noise_param)
    results['BM3D'] = {'img': bm3d_denoised, 'psnr': bm3d_psnr, 'ssim': bm3d_ssim}
    times['BM3D'] = bm3d_time

    # DnCNN (仅在模型可用时运行)
    if dncnn_model is not None:
        try:
            start_time = time.time()
            dncnn_denoised, dncnn_psnr, dncnn_ssim, dncnn_time = dncnn_denoise_color(noisy_img, clean_img)
            results['DnCNN'] = {
                'img': dncnn_denoised,
                'psnr': dncnn_psnr,
                'ssim': dncnn_ssim
            }
            times['DnCNN'] = dncnn_time
        except Exception as e:
            print(f"Error running DnCNN: {e}")

    # FFDNet (仅在模型可用时运行)
    # ---------- 1. FFDNet 灰度专用 ----------
    if ffdnet_model is not None:
        try:
            # 彩色 → 灰度
            gray_clean = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            gray_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

            start_time = time.time()
            ffdnet_gray, ffdnet_psnr, ffdnet_ssim, ffdnet_time = ffdnet_denoise(
                gray_noisy, gray_clean, sigma=noise_param)
            # 把灰度结果再复制成 3 通道，方便后续可视化
            ffdnet_color = cv2.cvtColor(ffdnet_gray, cv2.COLOR_GRAY2BGR)
            results['FFDNet'] = {
                'img': ffdnet_color,
                'psnr': ffdnet_psnr,
                'ssim': ffdnet_ssim
            }
            times['FFDNet'] = ffdnet_time
        except Exception as e:
            print(f"Error running FFDNet: {e}")

    # U-Net (仅在模型可用时运行)
    if unet_model is not None:
        try:
            start_time = time.time()
            unet_denoised, unet_psnr, unet_ssim, unet_time = unet_denoise_color(noisy_img, clean_img)
            results['UNet'] = {
                'img': unet_denoised,
                'psnr': unet_psnr,
                'ssim': unet_ssim
            }
            times['UNet'] = unet_time
        except Exception as e:
            print(f"Error running U-Net: {e}")

    # ISTA
    start_time = time.time()
    ista_denoised, ista_psnrs, ista_ssims = ista_l1_color(
        noisy_img, clean_img, lambda_=20,
        max_iter=max_iter)
    results['ISTA'] = {
        'img': ista_denoised,
        'psnr': psnr_color(clean_img, ista_denoised, data_range=255),
        'ssim': ssim_color(clean_img, ista_denoised, data_range=255)
    }
    times['ISTA'] = time.time() - start_time

    # FISTA
    start_time = time.time()
    fista_tv_denoised, fista_tv_psnrs, fista_tv_ssims = fista_tv_color(
        noisy_img, clean_img, max_iter=150)
    results['FISTA'] = {
        'img': fista_tv_denoised,
        'psnr': psnr_color(clean_img, fista_tv_denoised, data_range=255),
        'ssim': ssim_color(clean_img, fista_tv_denoised, data_range=255)
    }
    times['FISTA'] = time.time() - start_time

    # ADMM
    start_time = time.time()
    admm_denoised, admm_psnrs, admm_ssims = admm_tv_color(
        noisy_img, clean_img, max_iter=200)
    results['ADMM'] = {
        'img': admm_denoised,
        'psnr': psnr_color(clean_img, admm_denoised, data_range=255),
        'ssim': ssim_color(clean_img, admm_denoised, data_range=255)
    }
    times['ADMM'] = time.time() - start_time


    # Noise2Noise (彩色)
    start_time = time.time()
    try:
        n2n_pred, n2n_psnr, n2n_ssim, n2n_time, n2n_src1, n2n_src2, src1_psnr, src1_ssim, src2_psnr, src2_ssim = noise2noise_denoise_color(
            noisy_img, clean_img,
            model_weight_path='/202521000855/ZXQ/project/week6/noise2noise/runs/Noise2Noisegaussian/checkpoints/model_at_epoch_100.dat',
            patch_size=256,
            batch_size=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        results['Noise2Noise'] = {
            'img': n2n_pred,
            'psnr': n2n_psnr,
            'ssim': n2n_ssim,
            'src1': n2n_src1,  # 源图像1
            'src2': n2n_src2,  # 源图像2
            'src1_psnr': src1_psnr,  # 源图像1的PSNR
            'src1_ssim': src1_ssim,  # 源图像1的SSIM
            'src2_psnr': src2_psnr,  # 源图像2的PSNR
            'src2_ssim': src2_ssim  # 源图像2的SSIM
        }
        times['Noise2Noise'] = n2n_time
    except Exception as e:
        print(f"Error running Noise2Noise: {e}")


    # Neighbor2Neighbor (彩色)
    start_time = time.time()
    try:
        n2n_img, n2n_psnr, n2n_ssim, n2n_time = n2n_denoise_color(
            noisy_img, clean_img,
            model_weight_path='./neighbor_results/unet_gauss25_b4e100r02/2025-09-18-07-22/epoch_model_020.pth',
            patch_size=256,
            batch_size=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        results['Neighbor2Neighbor'] = {
            'img': n2n_img,
            'psnr': n2n_psnr,
            'ssim': n2n_ssim
        }
        times['Neighbor2Neighbor'] = n2n_time
    except Exception as e:
        print(f"Error running Neighbor2Neighbor: {e}")

    # 在所有算法处理完之后，统一转 uint8
    for algo in results:
        results[algo]['img'] = np.clip(results[algo]['img'], 0, 255).astype(np.uint8)

    return clean_img, noisy_img, results, times


def plot_results_color(clean_img, noisy_img, results, image_name, noise_type, noise_param,save_dir='results/image'):
    """可视化结果并保存（彩色版本）"""
    plt.figure(figsize=(12, 4))

    algorithms = ['BM3D', 'FFDNet', 'UNet', 'ADMM', 'Neighbor2Neighbor']
    num_algorithms = len([algo for algo in algorithms if algo in results])
    n_algo = len(algorithms) + 3               # +3 = Source1 / Source2 / Denoised
    plt.figure(figsize=(3 * (n_algo + 2), 4))  # +2 = Clean & Noisy

    # Clean & Noisy
    plt.subplot(1, n_algo + 2, 1)
    plt.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(1, n_algo + 2, 2)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Noisy\n({noise_type}={noise_param})')
    plt.axis('off')

    # 显示去噪结果
    for i, algo in enumerate(algorithms):
        plt.subplot(1, n_algo + 2, i + 3)
        if algo in results:
            plt.imshow(cv2.cvtColor(results[algo]['img'], cv2.COLOR_BGR2RGB))
            plt.title(f"{algo}\nPSNR={results[algo]['psnr']:.2f}\nSSIM={results[algo]['ssim']:.4f}")
        else:
            plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 追加 Noise2Noise 三图（仅当存在时）
    if 'Noise2Noise' in results:
        # Source1
        plt.subplot(1, n_algo + 2, len(algorithms) + 3)
        plt.imshow(cv2.cvtColor(results['Noise2Noise']['src1'], cv2.COLOR_BGR2RGB))
        plt.title(
            f"N2N Source1\nPSNR={results['Noise2Noise']['src1_psnr']:.2f}\nSSIM={results['Noise2Noise']['src1_ssim']:.4f}")
        plt.axis('off')

        # Source2
        plt.subplot(1, n_algo + 2, len(algorithms) + 4)
        plt.imshow(cv2.cvtColor(results['Noise2Noise']['src2'], cv2.COLOR_BGR2RGB))
        plt.title(
            f"N2N Source2\nPSNR={results['Noise2Noise']['src2_psnr']:.2f}\nSSIM={results['Noise2Noise']['src2_ssim']:.4f}")
        plt.axis('off')

        # Denoised
        plt.subplot(1, n_algo + 2, len(algorithms) + 5)
        plt.imshow(cv2.cvtColor(results['Noise2Noise']['img'], cv2.COLOR_BGR2RGB))
        plt.title(f"N2N Denoised\nPSNR={results['Noise2Noise']['psnr']:.2f}\nSSIM={results['Noise2Noise']['ssim']:.4f}")
        plt.axis('off')

    plt.suptitle(f'Image: {image_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'denoising_{image_name}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main_color():
    """彩色图像去噪主函数"""
    # 创建结果目录
    os.makedirs('results/image', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    # 获取测试图像
    dataset_path = '/202521000855/ZXQ/project/week6/data/Test/Set14'
    image_paths = glob.glob(os.path.join(dataset_path, '*.png')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.bmp'))

    if not image_paths:
        print(f"No images found in {dataset_path}. Using sample image.")
        clean_img = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.putText(clean_img, 'Sample Image', (50, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imwrite('sample.png', clean_img)
        image_paths = ['sample.png']

    # 噪声参数
    noise_params = [15, 25, 50]
    max_iter = 100

    # 存储所有结果
    all_results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        image_name = os.path.basename(image_path).split('.')[0]

        for noise_param in noise_params:
            print(f"\nProcessing {image_name} with noise level {noise_param}")

            # 处理图像
            clean_img, noisy_img, results, times = process_image_color(
                image_path, noise_type='gaussian', noise_param=noise_param, max_iter=max_iter)

            if clean_img is None:
                continue

            # 保存结果
            plot_results_color(clean_img, noisy_img, results, image_name, 'gaussian', noise_param)

            # 记录指标
            for algo, result in results.items():
                all_results.append({
                    'image': image_name,
                    'noise_level': noise_param,
                    'algorithm': algo,
                    'psnr': result['psnr'],
                    'ssim': result['ssim'],
                    'time': times.get(algo, 0)
                })

    # 保存结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv('results/metrics/color_denoising_results.csv', index=False)

    # 计算平均指标
    avg_metrics = df.groupby(['algorithm', 'noise_level']).agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'time': 'mean'
    }).reset_index()

    print("\nAverage Performance Metrics:")
    print(avg_metrics.to_string(index=False))

    # 保存平均指标
    avg_metrics.to_csv('results/metrics/color_avg_metrics.csv', index=False)

    print("\nProcessing completed! Results saved in 'results/' directory")


if __name__ == "__main__":
    main_color()