import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
import pywt
import time
import os
import bm3d
from scipy.fftpack import dct, idct
import glob
import pandas as pd
from tqdm import tqdm
import scipy.fft

# 设置随机种子以确保结果可重现
np.random.seed(42)


def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, amount=0.05, s_vs_p=0.5):
    """
    添加椒盐噪声
    参数:
        image: 输入图像
        amount: 噪声比例 (0-1)
        s_vs_p: 盐噪声比例 (0-1)
    """
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
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def ista_l1(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5, wavelet='db4', level=3):
    """ISTA算法（L1小波正则化）"""
    noisy = noisy_img.astype(np.float32)
    x = noisy.copy()
    last_x = x.copy()
    psnrs = []

    coeffs = pywt.wavedec2(noisy, wavelet, level=level)

    for i in range(max_iter):
        coeffs = pywt.wavedec2(x, wavelet, level=level)
        coeffs_thresh = [coeffs[0]]  # 低频分量
        for j in range(1, len(coeffs)):
            cH, cV, cD = coeffs[j]
            cH_t = soft_threshold(cH, lambda_)
            cV_t = soft_threshold(cV, lambda_)
            cD_t = soft_threshold(cD, lambda_)
            coeffs_thresh.append((cH_t, cV_t, cD_t))

        x = pywt.waverec2(coeffs_thresh, wavelet)
        x = np.clip(x, 0, 255)

        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, x, data_range=255)
            psnrs.append(psnr_val)

        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            break
        last_x = x.copy()

    return x.astype(np.uint8), psnrs


def fista_l1(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5, wavelet='db4', level=3):
    """FISTA算法（L1小波正则化）"""
    noisy = noisy_img.astype(np.float32)
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    last_x = x.copy()
    psnrs = []

    for i in range(max_iter):
        last_x = x.copy()
        coeffs = pywt.wavedec2(y, wavelet, level=level)
        coeffs_thresh = [coeffs[0]]
        for j in range(1, len(coeffs)):
            coeffs_thresh.append(tuple(soft_threshold(c, lambda_) for c in coeffs[j]))

        x = pywt.waverec2(coeffs_thresh, wavelet)
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t - 1) / t_next) * (x - last_x)
        t = t_next

        x = np.clip(x, 0, 255)
        y = np.clip(y, 0, 255)

        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, x, data_range=255)
            psnrs.append(psnr_val)

        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            break

    return x.astype(np.uint8), psnrs


def fista_tv(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5):
    """FISTA算法（TV正则化）"""
    noisy = noisy_img.astype(np.float32) / 255.0
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    last_x = x.copy()
    psnrs = []

    for i in range(max_iter):
        last_x = x.copy()
        x = denoise_tv_chambolle(y, weight=lambda_, max_num_iter=10)
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t - 1) / t_next) * (x - last_x)
        t = t_next

        denoised_uint8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, denoised_uint8, data_range=255)
            psnrs.append(psnr_val)

        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            break

    return (np.clip(x, 0, 1) * 255).astype(np.uint8), psnrs


def admm_tv(noisy_img, clean_img, lambda_, rho=1.0, max_iter=100, tol=1e-5):
    """ADMM算法（TV正则化）"""
    # 初始化变量
    x = noisy_img.copy().astype(np.float32)
    z_x = np.zeros_like(noisy_img)
    z_y = np.zeros_like(noisy_img)
    u_x = np.zeros_like(noisy_img)
    u_y = np.zeros_like(noisy_img)
    last_x = x.copy()
    psnrs = []

    # 定义梯度算子（前向差分）
    def grad(img):
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, :-1] = img[:, 1:] - img[:, :-1]  # x方向梯度
        gy[:-1, :] = img[1:, :] - img[:-1, :]  # y方向梯度
        return gx, gy

    def div(gx, gy):
        d = np.zeros_like(gx)
        # 更精确的散度计算
        d[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
        d[:, 0] = gx[:, 0]
        d[:, -1] = -gx[:, -2]

        gy_diff = np.zeros_like(gy)
        gy_diff[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
        gy_diff[0, :] = gy[0, :]
        gy_diff[-1, :] = -gy[-2, :]

        d = d + gy_diff
        return d

    # 使用FFT求解线性系统
    h, w = noisy_img.shape
    # 创建拉普拉斯算子的频域表示
    kernel = np.zeros((h, w))
    kernel[0, 0] = 4
    if w > 1:
        kernel[0, 1] = -1
        kernel[0, -1] = -1
    if h > 1:
        kernel[1, 0] = -1
        kernel[-1, 0] = -1

    denom_fft = scipy.fft.fft2(kernel)

    for i in range(max_iter):
        # 计算当前梯度
        dx, dy = grad(x)

        # z-更新：软阈值
        z_x_new = np.sign(dx + u_x) * np.maximum(np.abs(dx + u_x) - lambda_ / rho, 0)
        z_y_new = np.sign(dy + u_y) * np.maximum(np.abs(dy + u_y) - lambda_ / rho, 0)
        z_x, z_y = z_x_new, z_y_new

        # u-更新：对偶变量
        u_x = u_x + dx - z_x
        u_y = u_y + dy - z_y

        # x-更新：求解线性系统 (I - ρΔ)x = b
        b = noisy_img + rho * div(z_x - u_x, z_y - u_y)
        b_fft = scipy.fft.fft2(b)
        x_fft = b_fft / (1 + rho * denom_fft)
        x_new = np.real(scipy.fft.ifft2(x_fft))
        x = np.clip(x_new, 0, 255)

        # 记录PSNR
        psnrs.append(psnr(clean_img, x.astype(np.uint8), data_range=255))

        # 检查收敛性
        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            print(f"ADMM converged at iteration {i + 1}")
        break

        last_x = x.copy()

    return x.astype(np.uint8), psnrs


def bm3d_denoise(noisy_img, clean_img, sigma):
    """BM3D去噪算法"""
    start_time = time.time()
    denoised = bm3d.bm3d(noisy_img, sigma)
    elapsed_time = time.time() - start_time
    psnr_val = psnr(clean_img, denoised, data_range=255)
    ssim_val = ssim(clean_img, denoised, data_range=255)
    return denoised, psnr_val, ssim_val, elapsed_time


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
    bm3d_denoised, bm3d_psnr, bm3d_ssim, bm3d_time = bm3d_denoise(noisy_img, clean_img,
                                                                  noise_param if noise_type == 'gaussian' else 25)
    results['BM3D'] = {'img': bm3d_denoised, 'psnr': bm3d_psnr, 'ssim': bm3d_ssim}
    times['BM3D'] = bm3d_time

    # ISTA
    start_time = time.time()
    ista_denoised, ista_psnrs = ista_l1(noisy_img, clean_img, lambda_=10, max_iter=max_iter)
    results['ISTA'] = {'img': ista_denoised, 'psnr': psnr(clean_img, ista_denoised, data_range=255),
                       'ssim': ssim(clean_img, ista_denoised, data_range=255), 'psnrs': ista_psnrs}
    times['ISTA'] = time.time() - start_time

    # FISTA
    start_time = time.time()
    fista_denoised, fista_psnrs = fista_l1(noisy_img, clean_img, lambda_=10, max_iter=max_iter)
    results['FISTA'] = {'img': fista_denoised, 'psnr': psnr(clean_img, fista_denoised, data_range=255),
                        'ssim': ssim(clean_img, fista_denoised, data_range=255), 'psnrs': fista_psnrs}
    times['FISTA'] = time.time() - start_time

    # FISTA_TV
    start_time = time.time()
    fista_tv_denoised, fista_tv_psnrs = fista_tv(noisy_img, clean_img, lambda_=0.1, max_iter=max_iter)
    results['FISTA_TV'] = {'img': fista_tv_denoised, 'psnr': psnr(clean_img, fista_tv_denoised, data_range=255),
                           'ssim': ssim(clean_img, fista_tv_denoised, data_range=255), 'psnrs': fista_tv_psnrs}
    times['FISTA_TV'] = time.time() - start_time

    # ADMM
    start_time = time.time()
    admm_denoised, admm_psnrs = admm_tv(noisy_img, clean_img, lambda_=0.1, rho=1.0, max_iter=max_iter)
    results['ADMM'] = {'img': admm_denoised, 'psnr': psnr(clean_img, admm_denoised, data_range=255),
                       'ssim': ssim(clean_img, admm_denoised, data_range=255), 'psnrs': admm_psnrs}
    times['ADMM'] = time.time() - start_time

    return clean_img, noisy_img, results, times


def plot_results(clean_img, noisy_img, results, image_name, noise_type, noise_param, save_dir='results'):
    """可视化结果并保存"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Noisy Image\n({noise_type}, param={noise_param})')
    plt.axis('off')

    algorithms = ['BM3D', 'ISTA', 'FISTA', 'FISTA_TV', 'ADMM']
    for i, algo in enumerate(algorithms):
        plt.subplot(2, 4, i + 3)
        plt.imshow(results[algo]['img'], cmap='gray')
        plt.title(f"{algo}\nPSNR: {results[algo]['psnr']:.2f} dB\nSSIM: {results[algo]['ssim']:.4f}")
        plt.axis('off')

    plt.suptitle(f'Image: {image_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'denoising_{image_name}.png')
    plt.savefig(save_path)
    plt.close()


def plot_convergence(results, image_name, noise_type, noise_param, save_dir='results'):
    """绘制收敛曲线并保存"""
    plt.figure(figsize=(10, 6))

    algorithms = ['ISTA', 'FISTA', 'FISTA_TV', 'ADMM']
    for algo in algorithms:
        if algo in results:
            psnrs = results[algo]['psnrs']
            iterations = np.arange(10, 10 * (len(psnrs) + 1), 10)
            plt.plot(iterations, psnrs, label=algo, marker='o')

    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title(f'Convergence: {image_name} ({noise_type}, param={noise_param})')
    plt.legend()
    plt.grid(True)

    noise_dir = os.path.join(save_dir, noise_type, str(noise_param))
    os.makedirs(noise_dir, exist_ok=True)
    save_path = os.path.join(noise_dir, f'convergence_{image_name}.png')
    plt.savefig(save_path)
    plt.close()


def save_results_to_csv(all_results, save_path='results/summary.csv'):
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

    print("\n" + "=" * 100)
    print("Average Performance Across All Images")
    print("=" * 100)
    print(
        f"{'NoiseType':<15}{'NoiseParam':<15}{'Algorithm':<15}{'Avg PSNR (dB)':<15}{'Avg SSIM':<15}{'Avg Time (s)':<15}")
    print("-" * 100)

    for _, row in avg_results.iterrows():
        print(f"{row['NoiseType']:<15}{row['NoiseParam']:<15}{row['Algorithm']:<15}"
              f"{row['PSNR']:<15.2f}{row['SSIM']:<15.4f}{row['Time']:<15.2f}")

    print("=" * 100)


# 主程序
if __name__ == "__main__":
    dataset_path = '/project/test/Set14'
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
                max_iter=100,
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
    detailed_path = 'results/detailed_results.xlsx'
    with pd.ExcelWriter(detailed_path) as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)

        # 按噪声类型和参数分组
        for (noise_type, noise_param), group in df.groupby(['NoiseType', 'NoiseParam']):
            sheet_name = f"{noise_type}_{noise_param}"[:31]
            group.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved detailed results to {detailed_path}")