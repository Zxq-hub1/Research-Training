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

# 1. 定义并注册自定义损失函数
@register_keras_serializable(package="Custom")
def sum_squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))

# 2. 加载模型时指定自定义对象
model = tf.keras.models.load_model(
    'models/DnCNN_sigma25/final_model.keras',
    custom_objects={'sum_squared_error': sum_squared_error}
)

# 添加DnCNN模型加载函数
def load_dncnn_model():
    """加载预训练的DnCNN模型"""
    # 模型文件位于当前目录下的models/DnCNN_sigma25文件夹中
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


def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, amount=0.05, s_vs_p=0.5):
    """添加椒盐噪声"""
    noisy = np.copy(image)
    # 盐噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    # 椒噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
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


def ista_l1(noisy_img, clean_img, lambda_=15, max_iter=100, tol=1e-6, wavelet='db8', level=3):
    """ISTA算法（L1小波正则化）"""
    noisy = noisy_img.astype(np.float32)
    x = noisy.copy()
    last_x = x.copy()
    psnrs = []
    ssims = []

    # 预计算小波分解结构
    coeffs = pywt.wavedec2(noisy, wavelet, level=level)
    coeffs_shape = [coeffs[0].shape]
    for j in range(1, len(coeffs)):
        coeffs_shape.append(tuple(c.shape for c in coeffs[j]))

    # 学习率设置
    L = 4.0  # Lipschitz常数估计

    for i in range(max_iter):
        # 小波分解
        coeffs = pywt.wavedec2(x, wavelet, level=level)

        # 阈值处理高频系数
        coeffs_thresh = [coeffs[0]]  # 保留低频分量

        for j in range(1, len(coeffs)):
            # 对每个方向的高频系数进行阈值处理
            threshold = lambda_ * (0.5 ** j)   # 尺度相关的阈值
            cH, cV, cD = [soft_threshold(c, threshold) for c in coeffs[j]]
            coeffs_thresh.append((cH, cV, cD))

        # 小波重构
        x_new = pywt.waverec2(coeffs_thresh, wavelet)

        # 数据保真项更新
        x_new = x_new + (noisy - x_new) / L

        # 投影到有效范围
        x_new = np.clip(x_new, 0, 255)

        # 确保数值稳定性
        x_new = np.nan_to_num(x_new, nan=0.0, posinf=255, neginf=0)

        # 计算评估指标
        if i % 5 == 0 or i == max_iter - 1:
            denoised_uint8 = x_new.astype(np.uint8)
            psnr_val = psnr(clean_img, denoised_uint8, data_range=255)
            ssim_val = ssim(clean_img, denoised_uint8, data_range=255)
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

        # 检查收敛性
        if np.linalg.norm(x_new - last_x) / (np.linalg.norm(last_x) + 1e-10) < tol:
            break

        last_x = x_new.copy()
        x = x_new.copy()

    return x.astype(np.uint8), psnrs, ssims


def fista_tv(noisy_img, clean_img, lambda_=None, max_iter=100, tol=1e-4):
    """改进的 FISTA-TV 去噪算法，带背追踪线搜索与自适应 λ"""
    noisy = noisy_img.astype(np.float32) / 255.0
    h, w = noisy.shape

    # 自适应 λ 设置
    if lambda_ is None:
        sigma_est = estimate_sigma(noisy_img, average_sigmas=True)
        lambda_ = 0.8 * sigma_est / 255.0

    # 预去噪 warm-start
    x = denoise_tv_chambolle(noisy, weight=lambda_, max_num_iter=100)
    y = x.copy()
    t = 1.0
    last_x = x.copy()
    psnrs = []
    ssims = []

    # 估计 Lipschitz 常数 L
    L = 8.0  # TV 算子的最大特征值约为 8（经验值）

    def grad_tv(img):
        # 计算 TV 正则项的梯度（近似）
        return denoise_tv_chambolle(img, weight=lambda_ / L, max_num_iter=1)

    for i in range(max_iter):
        # 梯度下降步
        grad_step = grad_tv(y)
        x_new = y + (noisy - y) / L

        # TV 去噪步
        x_new = denoise_tv_chambolle(x_new, weight=lambda_ / L, max_num_iter=5)

        # FISTA 加速
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y_new = x_new + ((t - 1) / t_new) * (x_new - last_x)

        # 投影到 [0,1]
        y_new = np.clip(y_new, 0, 1)

        # 记录
        if i % 5 == 0 or i == max_iter - 1:
            denoised_uint8 = (x_new * 255).astype(np.uint8)
            psnr_val = psnr(clean_img, denoised_uint8, data_range=255)
            ssim_val = ssim(clean_img, denoised_uint8, data_range=255)
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

        # 收敛检查
        if np.linalg.norm(x_new - last_x) / (np.linalg.norm(last_x) + 1e-8) < tol:
            break

        last_x = x_new.copy()
        x = x_new.copy()
        y = y_new.copy()
        t = t_new

    return (np.clip(x, 0, 1) * 255).astype(np.uint8), psnrs, ssims


def admm_tv(noisy_img, clean_img, lambda_=None, rho=None, max_iter=200, tol=1e-4, min_iter=20):
    """改进的ADMM-TV去噪算法，自动调参，增强稳定性"""
    # 自适应参数设置
    if lambda_ is None:
        sigma_est = estimate_sigma(noisy_img, average_sigmas=True)
        lambda_ = 0.5 * sigma_est  # 经验公式，可微调
    if rho is None:
        rho = 1.0  # 初始惩罚参数，后续可自适应调整

    # 初始化变量
    x = noisy_img.astype(np.float32)
    z_x = np.zeros_like(noisy_img, dtype=np.float32)
    z_y = np.zeros_like(noisy_img, dtype=np.float32)
    u_x = np.zeros_like(noisy_img, dtype=np.float32)
    u_y = np.zeros_like(noisy_img, dtype=np.float32)
    last_x = x.copy()
    psnrs = []
    ssims = []

    h, w = noisy_img.shape
    eps = 1e-6  # 数值稳定性常数

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
        b = noisy_img + rho * div_zu
        b_fft = scipy.fft.fft2(b)
        x_fft = b_fft / (1 + rho * (laplacian_fft + eps))
        x_new = np.real(scipy.fft.ifft2(x_fft))
        x_new = np.clip(x_new, 0, 255)

        # z-更新：软阈值
        dx, dy = grad(x_new)
        z_x_new = soft_threshold(dx + u_x, lambda_ / rho)
        z_y_new = soft_threshold(dy + u_y, lambda_ / rho)

        # u-更新：对偶变量
        u_x += dx - z_x_new
        u_y += dy - z_y_new

        # 记录指标
        if i % 5 == 0 or i == max_iter - 1:
            denoised_uint8 = np.clip(x_new, 0, 255).astype(np.uint8)
            psnr_val = psnr(clean_img, denoised_uint8, data_range=255)
            ssim_val = ssim(clean_img, denoised_uint8, data_range=255)
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

        # 检查收敛性
        if i >= min_iter and np.linalg.norm(x_new - last_x) / (np.linalg.norm(last_x) + eps) < tol:
            break

        last_x = x_new.copy()
        x = x_new.copy()
        z_x, z_y = z_x_new, z_y_new

    return np.clip(x, 0, 255).astype(np.uint8), psnrs, ssims


def bm3d_denoise(noisy_img, clean_img, sigma):
    """BM3D去噪算法"""
    start_time = time.time()
    denoised = bm3d.bm3d(noisy_img, sigma)
    elapsed_time = time.time() - start_time
    psnr_val = psnr(clean_img, denoised, data_range=255)
    ssim_val = ssim(clean_img, denoised, data_range=255)
    return denoised, psnr_val, ssim_val, elapsed_time


def dncnn_denoise(noisy_img, clean_img):
    """DnCNN去噪算法"""
    if dncnn_model is None:
        raise RuntimeError("DnCNN model is not available")

    start_time = time.time()

    # 预处理图像
    noisy_norm = noisy_img.astype(np.float32) / 255.0
    noisy_input = np.expand_dims(noisy_norm, axis=[0, -1])  # 添加batch和channel维度

    # 预测
    denoised_output = dncnn_model.predict(noisy_input, verbose=0)

    # 后处理
    denoised_img = np.clip(denoised_output[0, ..., 0] * 255, 0, 255).astype(np.uint8)
    elapsed_time = time.time() - start_time

    # 计算评估指标
    psnr_val = psnr(clean_img, denoised_img, data_range=255)
    ssim_val = ssim(clean_img, denoised_img, data_range=255)

    return denoised_img, psnr_val, ssim_val, elapsed_time


def process_image(image_path, noise_type='gaussian', noise_param=25, max_iter=100, resize=True):
    """处理单个图像"""
    clean_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if clean_img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None

    if resize:
        clean_img = cv2.resize(clean_img, (256, 256))

    # 添加噪声
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(clean_img, noise_param)
    elif noise_type == 'salt_pepper':
        noisy_img = add_salt_pepper_noise(clean_img, amount=noise_param)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    results = {}
    times = {}

    # BM3D
    start_time = time.time()
    bm3d_denoised, bm3d_psnr, bm3d_ssim, bm3d_time = bm3d_denoise(
        noisy_img, clean_img, noise_param if noise_type == 'gaussian' else 25)
    results['BM3D'] = {'img': bm3d_denoised, 'psnr': bm3d_psnr, 'ssim': bm3d_ssim}
    times['BM3D'] = bm3d_time

    # DnCNN (仅在模型可用时运行)
    if dncnn_model is not None:
        try:
            start_time = time.time()
            dncnn_denoised, dncnn_psnr, dncnn_ssim, dncnn_time = dncnn_denoise(noisy_img, clean_img)
            results['DnCNN'] = {
                'img': dncnn_denoised,
                'psnr': dncnn_psnr,
                'ssim': dncnn_ssim
            }
            times['DnCNN'] = dncnn_time
        except Exception as e:
            print(f"Error running DnCNN: {e}")

    # ISTA
    start_time = time.time()
    ista_denoised, ista_psnrs, ista_ssims = ista_l1(
        noisy_img, clean_img, lambda_=20 if noise_type == 'gaussian' else 25,
        max_iter=max_iter)
    results['ISTA'] = {
        'img': ista_denoised,
        'psnr': psnr(clean_img, ista_denoised, data_range=255),
        'ssim': ssim(clean_img, ista_denoised, data_range=255),
        'psnrs': ista_psnrs,
        'ssims': ista_ssims
    }
    times['ISTA'] = time.time() - start_time

    # FISTA
    start_time = time.time()
    fista_tv_denoised, fista_tv_psnrs, fista_tv_ssims = fista_tv(
        noisy_img, clean_img, max_iter=150)
    results['FISTA'] = {
        'img': fista_tv_denoised,
        'psnr': psnr(clean_img, fista_tv_denoised, data_range=255),
        'ssim': ssim(clean_img, fista_tv_denoised, data_range=255),
        'psnrs': fista_tv_psnrs,
        'ssims': fista_tv_ssims
    }
    times['FISTA'] = time.time() - start_time

    # ADMM
    start_time = time.time()
    admm_denoised, admm_psnrs, admm_ssims = admm_tv(
        noisy_img, clean_img, max_iter=200)
    results['ADMM'] = {
        'img': admm_denoised,
        'psnr': psnr(clean_img, admm_denoised, data_range=255),
        'ssim': ssim(clean_img, admm_denoised, data_range=255),
        'psnrs': admm_psnrs,
        'ssims': admm_ssims
    }
    times['ADMM'] = time.time() - start_time

    return clean_img, noisy_img, results, times


def plot_results(clean_img, noisy_img, results, image_name, noise_type, noise_param, save_dir='results2/image'):
    """可视化结果并保存"""
    plt.figure(figsize=(15, 4))

    algorithms = ['BM3D', 'DnCNN', 'ISTA', 'FISTA', 'ADMM']
    num_algorithms = len([algo for algo in algorithms if algo in results])

    plt.subplot(1, 7, 1)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(1, 7, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Noisy Image\n({noise_type}, param={noise_param})')
    plt.axis('off')

    for i, algo in enumerate(algorithms):
        if algo in results:
            plt.subplot(1, 7, i + 3)
            plt.imshow(results[algo]['img'], cmap='gray')
            plt.title(f"{algo}\nPSNR: {results[algo]['psnr']:.2f} dB\nSSIM: {results[algo]['ssim']:.4f}")
            plt.axis('off')

    plt.suptitle(f'Image: {image_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'denoising_{image_name}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence(results, image_name, noise_type, noise_param, save_dir='results2/convergence'):
    """绘制收敛曲线并保存"""
    plt.figure(figsize=(12, 6))

    algorithms = ['ISTA', 'FISTA', 'ADMM']
    for algo in algorithms:
        if algo in results:
            psnrs = results[algo]['psnrs']
            ssims = results[algo]['ssims']
            iterations = np.arange(5, 5 * (len(psnrs) + 1), 5)

            plt.subplot(1, 2, 1)
            plt.plot(iterations, psnrs, label=algo, marker='o', markersize=4)

            plt.subplot(1, 2, 2)
            plt.plot(iterations, ssims, label=algo, marker='o', markersize=4)

    plt.subplot(1, 2, 1)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Convergence')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.title('SSIM Convergence')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Convergence: {image_name} ({noise_type}, param={noise_param})')

    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'convergence_{image_name}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_to_csv(all_results, save_path='results2/summary.csv'):
    """保存所有结果到CSV文件"""
    data = []
    for (image_name, noise_type, noise_param), results in all_results.items():
        for algo, metrics in results.items():
            data.append({
                'Image': image_name,
                'NoiseType': noise_type,
                'NoiseParam': noise_param,
                'Algorithm': algo,
                'PSNR': metrics['psnr'],
                'SSIM': metrics['ssim'],
                'Time': metrics['time']
            })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def print_summary_table(df):
    """打印汇总表格"""
    avg_results = df.groupby(['NoiseType', 'NoiseParam', 'Algorithm']).agg({
        'PSNR': 'mean',
        'SSIM': 'mean',
        'Time': 'mean'
    }).reset_index()

    print("\n" + "=" * 120)
    print("Average Performance Across All Images")
    print("=" * 120)
    print(
        f"{'NoiseType':<15}{'NoiseParam':<15}{'Algorithm':<15}{'Avg PSNR (dB)':<15}{'Avg SSIM':<15}{'Avg Time (s)':<15}")
    print("-" * 120)

    for _, row in avg_results.iterrows():
        print(f"{row['NoiseType']:<15}{row['NoiseParam']:<15}{row['Algorithm']:<15}"
              f"{row['PSNR']:<15.2f}{row['SSIM']:<15.4f}{row['Time']:<15.2f}")

    print("=" * 120)


def main():
    # 设置随机种子以确保结果可重现
    np.random.seed(42)

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

    # 定义噪声配置
    noise_configs = [
        ('gaussian', 15),
        ('gaussian', 25),
        ('gaussian', 50),
        ('salt_pepper', 0.05),
        ('salt_pepper', 0.1),
        ('salt_pepper', 0.2)
    ]

    all_results = {}

    # 处理每张图像和每种噪声配置
    for img_path in tqdm(image_paths, desc="Processing images"):
        image_name = os.path.basename(img_path)

        for noise_type, noise_param in tqdm(noise_configs, desc="Noise types", leave=False):
            clean_img, noisy_img, results, times = process_image(
                img_path,
                noise_type=noise_type,
                noise_param=noise_param,
                max_iter=150,
                resize=True
            )

            if clean_img is None:
                continue

            # 存储结果
            image_results = {}
            for algo, res in results.items():
                image_results[algo] = {
                    'psnr': res['psnr'],
                    'ssim': res['ssim'],
                    'time': times[algo]
                }
            all_results[(image_name, noise_type, noise_param)] = image_results

            # 可视化
            plot_results(clean_img, noisy_img, results, image_name, noise_type, noise_param)
            plot_convergence(results, image_name, noise_type, noise_param)

    # 保存和显示结果
    df = save_results_to_csv(all_results)
    print_summary_table(df)

    # 保存详细结果到Excel
    detailed_path = 'results2/detailed_results.xlsx'
    with pd.ExcelWriter(detailed_path) as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)

        # 按噪声类型和参数分组
        for (noise_type, noise_param), group in df.groupby(['NoiseType', 'NoiseParam']):
            sheet_name = f"{noise_type}_{noise_param}"[:31]
            group.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved detailed results to {detailed_path}")


if __name__ == "__main__":
    main()