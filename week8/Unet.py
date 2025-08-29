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
model = tf.keras.models.load_model(
    'models/DnCNN_sigma25/final_model.keras',
    custom_objects={'sum_squared_error': sum_squared_error}
)


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
        os.path.join('models/FFDNet', 'ffdnet_gray.pth'),
        os.path.join('models/FFDNet/checkpoints', 'best_model.pth'),
        os.path.join('models/FFDNet', 'ffdnet_gray_final.pth'),
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
        os.path.join('models/UNet', 'unet_color_sigma25.pth'),
        os.path.join('models/UNet/checkpoints', 'best_model_unet_color.pth'),
        os.path.join('models/UNet', 'unet_color_final.pth'),
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


# 全局加载模型，避免重复加载
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ffdnet_model = load_ffdnet_model(device=device)
    unet_model = load_unet_model(device=device)
except Exception as e:
    print(f"Warning: Could not load models. Error: {e}")
    ffdnet_model = None
    unet_model = None


# 添加DnCNN模型加载函数（保持灰度，因为DnCNN通常是灰度模型）
def load_dncnn_model():
    """加载预训练的DnCNN模型（灰度）"""
    model_path = os.path.join('models/DnCNN_sigma25', 'final_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DnCNN model not found at {model_path}. Please download the pretrained model.")
    return load_model(model_path)


# 全局加载DnCNN模型，避免重复加载
try:
    dncnn_model = load_dncnn_model()
except Exception as e:
    print(f"Warning: Could not load DnCNN model. DnCNN will be disabled. Error: {e}")
    dncnn_model = None



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

    # 在所有算法处理完之后，统一转 uint8
    for algo in results:
        results[algo]['img'] = np.clip(results[algo]['img'], 0, 255).astype(np.uint8)

    return clean_img, noisy_img, results, times


def plot_results_color(clean_img, noisy_img, results, image_name, noise_type, noise_param,save_dir='Unet_color_results/image'):
    """可视化结果并保存（彩色版本）"""
    plt.figure(figsize=(12, 4))

    algorithms = ['BM3D', 'DnCNN', 'FFDNet', 'UNet']
    num_algorithms = len([algo for algo in algorithms if algo in results])

    # 显示原始图像
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
    plt.title('Clean Image')
    plt.axis('off')

    # 显示噪声图像
    plt.subplot(1, 6, 2)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Noisy Image\n({noise_type}, param={noise_param})')
    plt.axis('off')

    # 显示去噪结果
    for i, algo in enumerate(algorithms):
        if algo in results:
            plt.subplot(1, 6, i + 3)
            plt.imshow(cv2.cvtColor(results[algo]['img'], cv2.COLOR_BGR2RGB))
            plt.title(f"{algo}\nPSNR: {results[algo]['psnr']:.2f} dB\nSSIM: {results[algo]['ssim']:.4f}")
            plt.axis('off')

    plt.suptitle(f'Image: {image_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'denoising_{image_name}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main_color():
    """彩色图像去噪主函数"""
    # 创建结果目录
    os.makedirs('Unet_color_results/image', exist_ok=True)
    os.makedirs('Unet_color_results/metrics', exist_ok=True)

    # 获取测试图像
    dataset_path = '/project/week6/data/Test/Set14'
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
    df.to_csv('Unet_color_results/metrics/color_denoising_results.csv', index=False)

    # 计算平均指标
    avg_metrics = df.groupby(['algorithm', 'noise_level']).agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'time': 'mean'
    }).reset_index()

    print("\nAverage Performance Metrics:")
    print(avg_metrics.to_string(index=False))

    # 保存平均指标
    avg_metrics.to_csv('Unet_color_results/metrics/color_avg_metrics.csv', index=False)

    print("\nProcessing completed! Results saved in 'Unet_color_results/' directory")


if __name__ == "__main__":
    main_color()