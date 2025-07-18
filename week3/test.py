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
import scipy.fft
import glob
import pandas as pd


def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def soft_threshold(x, threshold):
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def ista_l1(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5, wavelet='db4', level=3):
    """
    ISTA算法（L1小波正则化）

    参数:
        noisy_img: 噪声图像 (uint8)
        clean_img: 干净图像 (uint8)
        lambda_: 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        wavelet: 使用的小波基
        level: 小波分解层数

    返回:
        denoised: 去噪后的图像
        psnrs: 每次迭代的PSNR值列表
    """
    # 将图像转换为float类型
    noisy = noisy_img.astype(np.float32)
    # 初始化
    x = noisy.copy()
    last_x = x.copy()
    psnrs = []

    # 执行小波变换
    coeffs = pywt.wavedec2(noisy, wavelet, level=level)

    # ISTA迭代
    for i in range(max_iter):
        # 小波分解
        coeffs = pywt.wavedec2(x, wavelet, level=level)

        # 对小波系数应用软阈值
        coeffs_thresh = [coeffs[0]]  # 低频分量
        for j in range(1, len(coeffs)):
            cH, cV, cD = coeffs[j]
            cH_t = soft_threshold(cH, lambda_)
            cV_t = soft_threshold(cV, lambda_)
            cD_t = soft_threshold(cD, lambda_)
            coeffs_thresh.append((cH_t, cV_t, cD_t))

        # 小波重构
        x = pywt.waverec2(coeffs_thresh, wavelet)

        # 确保图像在0-255范围内
        x = np.clip(x, 0, 255)

        # 计算PSNR
        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, x, data_range=255)
            psnrs.append(psnr_val)
            print(f"Iteration {i + 1}/{max_iter}, PSNR: {psnr_val:.2f}")

        # 检查收敛性
        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            print(f"ISTA converged at iteration {i + 1}")
            break

        last_x = x.copy()

    return x.astype(np.uint8), psnrs


def fista_l1(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5, wavelet='db4', level=3):
    """
    FISTA算法（L1小波正则化）

    参数:
        noisy_img: 噪声图像 (uint8)
        clean_img: 干净图像 (uint8)
        lambda_: 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        wavelet: 使用的小波基
        level: 小波分解层数

    返回:
        denoised: 去噪后的图像
        psnrs: 每次迭代的PSNR值列表
    """
    # 将图像转换为float类型
    noisy = noisy_img.astype(np.float32)
    # 初始化
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    last_x = x.copy()
    psnrs = []

    # FISTA迭代
    for i in range(max_iter):
        # 保存上一步的x
        last_x = x.copy()

        # 小波分解
        coeffs = pywt.wavedec2(y, wavelet, level=level)

        # 对小波系数应用软阈值
        coeffs_thresh = [coeffs[0]]
        for j in range(1, len(coeffs)):
            coeffs_thresh.append(tuple(soft_threshold(c, lambda_) for c in coeffs[j]))

        # 小波重构
        x = pywt.waverec2(coeffs_thresh, wavelet)

        # 更新t和y
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t - 1) / t_next) * (x - last_x)
        t = t_next

        # 确保图像在0-255范围内
        x = np.clip(x, 0, 255)
        y = np.clip(y, 0, 255)

        # 计算PSNR
        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, x, data_range=255)
            psnrs.append(psnr_val)
            print(f"Iteration {i + 1}/{max_iter}, PSNR: {psnr_val:.2f}")

        # 检查收敛性
        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            print(f"FISTA converged at iteration {i + 1}")
            break

    return x.astype(np.uint8), psnrs


def fista_tv(noisy_img, clean_img, lambda_, max_iter=100, tol=1e-5):
    """
    FISTA算法（TV正则化）

    参数:
        noisy_img: 噪声图像 (uint8)
        clean_img: 干净图像 (uint8)
        lambda_: 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度

    返回:
        denoised: 去噪后的图像
        psnrs: 每次迭代的PSNR值列表
    """
    # 将图像转换为float类型并归一化到[0,1]
    noisy = noisy_img.astype(np.float32) / 255.0
    # 初始化
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    last_x = x.copy()
    psnrs = []

    # FISTA迭代
    for i in range(max_iter):
        # 保存上一步的x
        last_x = x.copy()

        # 使用Chambolle TV去噪作为proximal操作
        x = denoise_tv_chambolle(y, weight=lambda_, max_num_iter=10)

        # 更新t和y
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t - 1) / t_next) * (x - last_x)
        t = t_next

        # 计算PSNR（需要转换回0-255范围）
        denoised_uint8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        if i % 10 == 0 or i == max_iter - 1:
            psnr_val = psnr(clean_img, denoised_uint8, data_range=255)
            psnrs.append(psnr_val)
            print(f"Iteration {i + 1}/{max_iter}, PSNR: {psnr_val:.2f}")

        # 检查收敛性
        if np.linalg.norm(x - last_x) / np.linalg.norm(last_x) < tol:
            print(f"FISTA-TV converged at iteration {i + 1}")
            break

    # 将图像转换回0-255范围
    denoised = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return denoised, psnrs


def admm_tv(noisy_img, clean_img, lambda_, rho=1.0, max_iter=100, tol=1e-5):
    """
    ADMM算法（TV正则化）

    参数:
        noisy_img: 噪声图像 (uint8)
        lambda_: 正则化参数
        rho: 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度

    返回:
        denoised: 去噪后的图像
        psnrs: 每次迭代的PSNR值列表
    """

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

    # 定义散度算子（负梯度的共轭）
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


def process_image(image_path, sigma=25, max_iter=100, resize=True):
    """处理单个图像"""
    # 读取图像
    clean_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if clean_img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None

    # 调整图像大小以加快处理速度
    if resize:
        clean_img = cv2.resize(clean_img, (256, 256))

    # 添加噪声
    noisy_img = add_gaussian_noise(clean_img, sigma)

    # 存储结果
    results = {}
    times = {}

    # 1. BM3D
    print("\nRunning BM3D...")
    start_time = time.time()
    bm3d_denoised, bm3d_psnr, bm3d_ssim, bm3d_time = bm3d_denoise(noisy_img, clean_img, sigma)
    results['BM3D'] = {'img': bm3d_denoised, 'psnr': bm3d_psnr, 'ssim': bm3d_ssim}
    times['BM3D'] = bm3d_time

    # 2. ISTA (L1小波正则化)
    print("\nRunning ISTA (L1 wavelet)...")
    start_time = time.time()
    ista_denoised, ista_psnrs = ista_l1(noisy_img, clean_img, lambda_=10, max_iter=max_iter)
    ista_time = time.time() - start_time
    results['ISTA'] = {'img': ista_denoised, 'psnr': psnr(clean_img, ista_denoised, data_range=255),
                       'ssim': ssim(clean_img, ista_denoised, data_range=255), 'psnrs': ista_psnrs}
    times['ISTA'] = ista_time

    # 3. FISTA (L1小波正则化)
    print("\nRunning FISTA (L1 wavelet)...")
    start_time = time.time()
    fista_denoised, fista_psnrs = fista_l1(noisy_img, clean_img, lambda_=10, max_iter=max_iter)
    fista_time = time.time() - start_time
    results['FISTA'] = {'img': fista_denoised, 'psnr': psnr(clean_img, fista_denoised, data_range=255),
                        'ssim': ssim(clean_img, fista_denoised, data_range=255), 'psnrs': fista_psnrs}
    times['FISTA'] = fista_time

    # 4. FISTA (TV正则化)
    print("\nRunning FISTA (TV)...")
    start_time = time.time()
    fista_tv_denoised, fista_tv_psnrs = fista_tv(noisy_img, clean_img, lambda_=1.0, max_iter=max_iter)
    fista_tv_time = time.time() - start_time
    results['FISTA_TV'] = {'img': fista_tv_denoised, 'psnr': psnr(clean_img, fista_tv_denoised, data_range=255),
                           'ssim': ssim(clean_img, fista_tv_denoised, data_range=255), 'psnrs': fista_tv_psnrs}
    times['FISTA_TV'] = fista_tv_time

    # 5. ADMM (TV正则化)
    print("\nRunning ADMM (TV)...")
    start_time = time.time()
    admm_denoised, admm_psnrs = admm_tv(noisy_img, clean_img, lambda_=0.1, rho=1.0, max_iter=max_iter)
    admm_time = time.time() - start_time
    results['ADMM'] = {'img': admm_denoised, 'psnr': psnr(clean_img, admm_denoised, data_range=255),
                       'ssim': ssim(clean_img, admm_denoised, data_range=255), 'psnrs': admm_psnrs}
    times['ADMM'] = admm_time

    return clean_img, noisy_img, results, times


def plot_results(clean_img, noisy_img, results, image_name, save_dir='results'):
    """可视化结果并保存"""
    plt.figure(figsize=(15, 10))

    # 显示原始图像和噪声图像
    plt.subplot(1, 7, 1)
    plt.imshow(clean_img, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(1, 7, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Noisy Image (σ=25)')
    plt.axis('off')

    # 显示各种算法的去噪结果
    algorithms = ['BM3D', 'ISTA', 'FISTA', 'FISTA_TV', 'ADMM']
    for i, algo in enumerate(algorithms):
        plt.subplot(1, 7, i + 3)
        plt.imshow(results[algo]['img'], cmap='gray')
        plt.title(f"{algo}\nPSNR: {results[algo]['psnr']:.2f} dB\nSSIM: {results[algo]['ssim']:.4f}")
        plt.axis('off')

    plt.suptitle(f'Image: {image_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'denoising_{image_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_convergence(results, image_name, save_dir='results'):
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
    plt.title(f'Convergence Behavior: {image_name}')
    plt.legend()
    plt.grid(True)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'convergence_{image_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to {save_path}")


def save_results_to_csv(all_results, save_path='results/summary.csv'):
    """保存所有结果到CSV文件"""
    # 创建数据列表
    data = []
    for image_name, results in all_results.items():
        for algo, metrics in results.items():
            data.append({
                'Image': image_name,
                'Algorithm': algo,
                'PSNR': metrics['psnr'],
                'SSIM': metrics['ssim'],
                'Time': metrics['time']
            })

    # 转换为DataFrame并保存
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved summary results to {save_path}")
    return df


def print_summary_table(df):
    """打印汇总表格"""
    # 计算平均值
    avg_results = df.groupby('Algorithm').agg({
        'PSNR': 'mean',
        'SSIM': 'mean',
        'Time': 'mean'
    }).reset_index()

    print("\n" + "=" * 80)
    print("Average Performance Across All Images")
    print("=" * 80)
    print(f"{'Algorithm':<15}{'Avg PSNR (dB)':<15}{'Avg SSIM':<15}{'Avg Time (s)':<15}")
    print("-" * 80)

    for _, row in avg_results.iterrows():
        print(f"{row['Algorithm']:<15}{row['PSNR']:<15.2f}{row['SSIM']:<15.4f}{row['Time']:<15.2f}")

    print("=" * 80)


# 主程序
def main():
    # 设置随机种子以确保结果可重现
    np.random.seed(42)

    # 设置数据集路径
    dataset_path = '/project/week3/Set14'  # 修改为你的Set14数据集路径

    # 获取所有图像文件
    image_paths = glob.glob(os.path.join(dataset_path, '*.png')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.bmp'))

    if not image_paths:
        print(f"No images found in {dataset_path}. Using sample image.")
        # 如果没有找到图像，创建一个示例图像
        clean_img = np.ones((256, 256), dtype=np.uint8) * 128
        cv2.putText(clean_img, 'Sample Image', (50, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imwrite('sample.png', clean_img)
        image_paths = ['sample.png']

    # 存储所有结果
    all_results = {}

    # 处理每张图像
    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        print(f"\n{'=' * 80}")
        print(f"Processing image: {image_name}")
        print(f"{'=' * 80}")

        # 处理当前图像
        clean_img, noisy_img, results, times = process_image(
            img_path,
            sigma=25,
            max_iter=100,
            resize=True  # 为加速处理调整大小
        )

        if clean_img is None:
            continue

        # 为当前图像存储结果
        image_results = {}
        for algo, res in results.items():
            image_results[algo] = {
                'psnr': res['psnr'],
                'ssim': res['ssim'],
                'time': times[algo]
            }
        all_results[image_name] = image_results

        # 可视化当前图像结果
        plot_results(clean_img, noisy_img, results, image_name)

        # 绘制收敛曲线
        plot_convergence(results, image_name)

        # 打印当前图像结果
        print(f"\nResults for {image_name}:")
        print(f"{'Algorithm':<15}{'PSNR (dB)':<15}{'SSIM':<15}{'Time (s)':<15}")
        print("-" * 80)
        for algo, res in image_results.items():
            print(f"{algo:<15}{res['psnr']:<15.2f}{res['ssim']:<15.4f}{res['time']:<15.2f}")
        print("=" * 80)

    # 保存所有结果到CSV
    df = save_results_to_csv(all_results)

    # 打印汇总表格
    print_summary_table(df)

    # 保存详细结果到CSV
    detailed_path = 'results/detailed_results.csv'  # 修改文件扩展名为.csv
    df.to_csv(detailed_path, index=False)  # 使用to_csv方法保存

    # 为每张图像创建单独的工作表
    for image_name, results in all_results.items():
        image_data = []
        for algo, metrics in results.items():
            image_data.append({
                'Algorithm': algo,
                'PSNR': metrics['psnr'],
                'SSIM': metrics['ssim'],
                'Time': metrics['time']
            })
        pd.DataFrame(image_data).to_csv(detailed_path, index=False)

    print(f"Saved detailed results to {detailed_path}")


if __name__ == "__main__":
    main()