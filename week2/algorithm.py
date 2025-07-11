import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import time
import os
import glob
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import bm3d
from tqdm import tqdm  # 进度条


def load_image(image_path):
    """
    加载图像并转换为灰度图
    :param image_path: 图像文件路径
    :return: 灰度图像 (numpy数组)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像文件未找到: {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def add_gaussian_noise(image, sigma):
    """
    添加高斯噪声
    :param image: 原始图像 (灰度, uint8)
    :param sigma: 噪声标准差
    :return: 带噪声图像 (uint8)
    """
    # 生成高斯噪声
    noise = np.random.normal(0, sigma, image.shape)
    # 添加噪声并确保像素值在0-255范围内
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy_image.astype(np.uint8)


def add_salt_pepper_noise(image, prob):
    """
    添加椒盐噪声
    :param image: 原始图像 (灰度, uint8)
    :param prob: 噪声像素比例 (0-1)
    :return: 带噪声图像 (uint8)
    """
    noisy_image = np.copy(image)
    # 生成随机掩码
    random_mask = np.random.rand(*image.shape)

    # 添加椒噪声 (黑色像素)
    pepper_mask = random_mask < prob / 2
    noisy_image[pepper_mask] = 0

    # 添加盐噪声 (白色像素)
    salt_mask = random_mask > 1 - prob / 2
    noisy_image[salt_mask] = 255

    return noisy_image


def soft_threshold(data, threshold):
    """
    软阈值函数 (核心ISTA操作)
    :param data: 输入数据
    :param threshold: 阈值
    :return: 阈值处理后的数据
    """
    # 计算绝对值并应用阈值
    abs_data = np.abs(data)
    thresholded = np.sign(data) * np.maximum(abs_data - threshold, 0)
    return thresholded


def ista_denoise(noisy_img, lambda_, max_iter=100, step_size=1.0, wavelet='db4', level=3, verbose=False):
    """
    ISTA图像去噪实现（完全修复尺寸问题版本）

    参数:
        noisy_img: 噪声图像 (uint8格式，H x W)
        lambda_: 正则化系数
        max_iter: 最大迭代次数
        step_size: 学习率
        wavelet: 使用的小波基
        level: 小波分解层数
        verbose: 是否打印迭代信息

    返回:
        denoised: 去噪图像 (uint8，保持原始尺寸)
        psnr_history: PSNR历史记录
    """
    # 1. 记录原始尺寸并检查
    original_shape = noisy_img.shape
    if len(original_shape) != 2:
        raise ValueError("输入必须是二维灰度图像")

    # 2. 计算最小填充量（使每边都能被2^level整除）
    def calculate_padding(size, level):
        remainder = size % (2 ** level)
        return 0 if remainder == 0 else (2 ** level - remainder)

    pad_h = calculate_padding(original_shape[0], level)
    pad_w = calculate_padding(original_shape[1], level)

    # 3. 预处理（归一化+填充）
    img_float = noisy_img.astype(np.float32) / 255.0
    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(img_float, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        x_padded = img_float.copy()

    psnr_history = []

    # 4. ISTA迭代
    for i in range(max_iter):
        # 梯度计算
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img_float, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            img_padded = img_float
        gradient = x_padded - img_padded

        # 梯度更新
        x_temp = x_padded - step_size * gradient

        # 小波变换
        coeffs = pywt.wavedec2(x_temp, wavelet, level=level)

        # 软阈值处理
        coeffs_thresh = [coeffs[0]]  # 低频分量
        for j in range(1, len(coeffs)):
            cH, cV, cD = coeffs[j]
            cH_t = soft_threshold(cH, lambda_ * step_size)
            cV_t = soft_threshold(cV, lambda_ * step_size)
            cD_t = soft_threshold(cD, lambda_ * step_size)
            coeffs_thresh.append((cH_t, cV_t, cD_t))

        # 小波重建
        x_recon = pywt.waverec2(coeffs_thresh, wavelet)

        # 裁剪回原始尺寸
        x_recon = x_recon[:original_shape[0], :original_shape[1]]
        x_recon = np.clip(x_recon, 0, 1)

        # 计算PSNR（确保尺寸匹配）
        denoised_uint8 = (x_recon * 255).astype(np.uint8)
        assert noisy_img.shape == denoised_uint8.shape, "尺寸不匹配"
        current_psnr = psnr(noisy_img, denoised_uint8, data_range=255)
        psnr_history.append(current_psnr)

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            print(f"Iteration {i + 1}/{max_iter}, PSNR: {current_psnr:.2f} dB")

        # 更新变量
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(x_recon, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            x_padded = x_recon

    # 最终结果
    denoised = (x_recon * 255).astype(np.uint8)
    return denoised, psnr_history


def bm3d_denoise(noisy_img, noise_type, noise_param):
    """
    BM3D图像去噪包装函数
    :param noisy_img: 噪声图像 (uint8)
    :param noise_type: 噪声类型 ('gaussian' 或 'salt_pepper')
    :param noise_param:
            - 高斯噪声时表示标准差σ
            - 椒盐噪声时表示噪声密度(0-1)
    :return: 去噪后的图像 (uint8)
    """
    # 高斯噪声直接使用BM3D
    if noise_type == 'gaussian':
        return bm3d.bm3d(noisy_img, noise_param)
    else:
        # 对于椒盐噪声，先应用3x3中值滤波去除脉冲噪声，再用BM3D处理残留噪声
        window_size = 3 if noise_param < 0.2 else 5
        preprocessed = cv2.medianBlur(noisy_img, window_size)  # 中值滤波预处理
        residual_sigma = min(noise_param * 100, 30)  # 估计残留噪声水平（经验公式） 上限设为30
        return bm3d.bm3d(preprocessed, residual_sigma)  # BM3D处理残留噪声

def evaluate_denoising(clean_img, denoised_img):
    """
    评估去噪效果
    :param clean_img: 原始干净图像
    :param denoised_img: 去噪后的图像
    :return: PSNR和SSIM值
    """
    psnr_value = psnr(clean_img, denoised_img, data_range=255)
    ssim_value = ssim(clean_img, denoised_img, data_range=255)
    return psnr_value, ssim_value


def plot_results(clean_img, noisy_img, ista_denoised, bm3d_denoised, noise_type, intensity, image_name, save_path=None):
    """
    可视化结果
    :param clean_img: 原始干净图像
    :param noisy_img: 噪声图像
    :param ista_denoised: ISTA去噪结果
    :param bm3d_denoised: BM3D去噪结果
    :param noise_type: 噪声类型
    :param intensity: 噪声强度
    :param image_name: 图像名称
    :param save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(clean_img, cmap='gray')
    plt.title(f'Original: {image_name}')
    plt.axis('off')

    # 噪声图像
    plt.subplot(2, 2, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(
        f'{noise_type} Noise (σ={intensity})' if noise_type == 'Gaussian' else f'{noise_type} Noise (p={intensity})')
    plt.axis('off')

    # ISTA去噪结果
    plt.subplot(2, 2, 3)
    plt.imshow(ista_denoised, cmap='gray')
    plt.title('ISTA Denoised')
    plt.axis('off')

    # BM3D去噪结果
    if bm3d_denoised is not None:
        plt.subplot(2, 2, 4)
        plt.imshow(bm3d_denoised, cmap='gray')
        plt.title('BM3D Denoised')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence(psnr_history, title, save_path=None):
    """
    绘制收敛曲线
    :param psnr_history: PSNR历史记录
    :param title: 图表标题
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(psnr_history, marker='o', markersize=3, linestyle='-', linewidth=1.5, color='b')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def process_image(image_path, results_dir, noise_configs, max_iter=100, step_size=0.8):
    """
    处理单个图像的去噪实验
    :param image_path: 图像路径
    :param results_dir: 结果保存目录
    :param noise_configs: 噪声配置列表
    :param max_iter: ISTA最大迭代次数
    :param step_size: ISTA步长
    :return: 该图像的结果列表
    """
    # 获取图像名称
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 加载图像
    try:
        clean_img = load_image(image_path)
    except Exception as e:
        print(f"加载图像 {image_path} 失败: {e}")
        return []

    # 为当前图像创建目录
    img_results_dir = os.path.join(results_dir, "per_image_results", image_name)
    os.makedirs(img_results_dir, exist_ok=True)

    # 存储当前图像的结果
    image_results = []

    # 处理所有噪声配置
    for config in tqdm(noise_configs, desc=f"处理 {image_name}", leave=False):
        noise_type = config['type']
        intensity = config['intensity']

        # 1. 添加噪声
        if noise_type == 'Gaussian':
            noisy_img = add_gaussian_noise(clean_img, intensity)
            lambda_ = 0.1  # 高斯噪声的正则化参数
        else:  # Salt-Pepper
            noisy_img = add_salt_pepper_noise(clean_img, intensity)
            lambda_ = 0.2  # 椒盐噪声的正则化参数

        # 2. ISTA去噪
        start_time = time.time()
        ista_denoised, psnr_history = ista_denoise(
            noisy_img,
            lambda_=lambda_,
            max_iter=max_iter,
            step_size=step_size,
            wavelet='db4'
        )
        ista_time = time.time() - start_time

        # 3. 评估ISTA
        ista_psnr, ista_ssim = evaluate_denoising(clean_img, ista_denoised)

        # 4. BM3D去噪
        start_time = time.time()
        bm3d_denoised = bm3d_denoise(
            noisy_img=noisy_img,
            noise_type=noise_type,
            noise_param=intensity
        )
        bm3d_time = time.time() - start_time
        bm3d_psnr, bm3d_ssim = evaluate_denoising(clean_img, bm3d_denoised)

        # 5. 保存结果
        result = {
            'image': image_name,
            'noise_type': noise_type,
            'intensity': intensity,
            'ista_psnr': ista_psnr,
            'ista_ssim': ista_ssim,
            'ista_time': ista_time,
            'bm3d_psnr': bm3d_psnr,
            'bm3d_ssim': bm3d_ssim,
            'bm3d_time': bm3d_time
        }
        image_results.append(result)

        # 6. 可视化结果
        if intensity in [15, 0.05] if noise_type == 'gaussian' else [0.05, 0.1]:
            vis_path = os.path.join(img_results_dir, f"{noise_type}_{intensity}.png")
            plot_results(clean_img, noisy_img, ista_denoised, bm3d_denoised,
                      noise_type, intensity, image_name, save_path=vis_path)

        # 7. 保存收敛曲线
        conv_path = os.path.join(results_dir, "convergence", f"{image_name}_{noise_type}_{intensity}_convergence.png")
        plot_convergence(psnr_history,
                         f'ISTA Convergence ({image_name}, {noise_type} Noise, Intensity: {intensity})',
                         save_path=conv_path)

    return image_results


def analyze_step_size_impact(images, results_dir, noise_configs=None):
    """
    分析步长对ISTA收敛性的影响
    :param images: 图像列表 (numpy数组)
    :param results_dir: 结果保存目录
    :param sigma: 高斯噪声标准差
    """

    # 默认噪声配置
    if noise_configs is None:
        noise_configs = [
            {'type': 'Gaussian', 'intensity': 25},  # 高斯噪声σ=25
            {'type': 'Salt-Pepper', 'intensity': 0.1}  # 椒盐噪声10%
        ]

    print("\n分析步长对收敛性的影响...")
    step_sizes = [0.1, 0.5, 1.0, 1.5]

    # 对每种噪声类型分别分析
    for config in noise_configs:
        noise_type = config['type']
        intensity = config['intensity']
        # 存储每种步长的平均PSNR历史
        avg_psnr_histories = {step: [] for step in step_sizes}

    # 为每个步长创建存储列表
    for step in step_sizes:
        for _ in range(100):  # 预分配空间
            avg_psnr_histories[step].append(0)

    # 处理每张图像
    for img_idx, clean_img in enumerate(images):
        # 添加噪声
        if noise_type == 'Gaussian':
            noisy_img = add_gaussian_noise(clean_img, intensity)
        else:  # Salt-Pepper
            noisy_img = add_salt_pepper_noise(clean_img, intensity)

        # 测试不同步长
        for step in step_sizes:
            _, psnr_history = ista_denoise(
                noisy_img,
                lambda_=0.1,
                max_iter=100,
                step_size=step,
                wavelet='db4'
            )

            # 累积PSNR值用于平均
            for i, psnr_val in enumerate(psnr_history):
                if i < len(avg_psnr_histories[step]):
                    avg_psnr_histories[step][i] += psnr_val

    # 计算平均PSNR
    num_images = len(images)
    for step in step_sizes:
        for i in range(len(avg_psnr_histories[step])):
            avg_psnr_histories[step][i] /= num_images

    # 绘制结果
    plt.figure(figsize=(10, 6))
    for step in step_sizes:
        plt.plot(avg_psnr_histories[step], label=f'Step Size={step}')

    # 根据噪声类型设置标题
    if noise_type == 'Gaussian':
            title = f'ISTA Convergence under Gaussian Noise (σ={intensity})'
            filename = f"step_size_impact_gaussian_{intensity}.png"
    else:
            title = f'ISTA Convergence under Salt-Pepper Noise (p={intensity})'
            filename = f"step_size_impact_sp_{intensity}.png"

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Average PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存结果
    save_path = os.path.join(results_dir, "step_size_impact.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"步长影响分析结果保存至: {save_path}")


def generate_summary_table(all_results, results_dir):
    """
    生成汇总表格并保存为CSV和LaTeX格式
    :param all_results: 所有结果数据
    :param results_dir: 结果保存目录
    """
    # 创建DataFrame
    df = pd.DataFrame(all_results)

    # 保存为CSV
    csv_path = os.path.join(results_dir, "summary_results.csv")
    df.to_csv(csv_path, index=False)

    # 创建汇总表格
    summary_data = []

    # 按噪声类型和强度汇总
    noise_configs = [
        {'type': 'Gaussian', 'intensity': 15},
        {'type': 'Gaussian', 'intensity': 25},
        {'type': 'Gaussian', 'intensity': 50},
        {'type': 'Salt-Pepper', 'intensity': 0.05},
        {'type': 'Salt-Pepper', 'intensity': 0.10},
        {'type': 'Salt-Pepper', 'intensity': 0.20}
    ]

    for config in noise_configs:
        noise_type = config['type']
        intensity = config['intensity']

        # 过滤当前配置的结果
        subset = df[(df['noise_type'] == noise_type) & (df['intensity'] == intensity)]

        # 计算平均值
        metrics = {
            'Noise Type': noise_type,
            'Intensity': intensity,
            'ISTA_PSNR': subset['ista_psnr'].mean(),
            'ISTA_SSIM': subset['ista_ssim'].mean(),
            'ISTA_Time': subset['ista_time'].mean(),
            'BM3D_PSNR': subset['bm3d_psnr'].mean(),
            'BM3D_SSIM': subset['bm3d_ssim'].mean(),
            'BM3D_Time': subset['bm3d_time'].mean()
        }

        # 计算性能提升百分比（BM3D相对于ISTA）
        if not np.isnan(metrics['BM3D_PSNR']):
            metrics['PSNR_Improve'] = (metrics['BM3D_PSNR'] - metrics['ISTA_PSNR']) / metrics['ISTA_PSNR'] * 100
            metrics['SSIM_Improve'] = (metrics['BM3D_SSIM'] - metrics['ISTA_SSIM']) / metrics['ISTA_SSIM'] * 100
        else:
            metrics['PSNR_Improve'] = np.nan
            metrics['SSIM_Improve'] = np.nan

        summary_data.append(metrics)

    # 创建汇总DataFrame
    summary_df = pd.DataFrame(summary_data)

    # 保存汇总表格
    summary_csv_path = os.path.join(results_dir, "summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # 打印汇总表格
    print("\n" + "=" * 120)
    print("去噪性能汇总结果 (Set14数据集平均值)")
    print("=" * 120)

    # 表头
    print(f"{'Noise Type':<15} | {'Intensity':<12} | {'Algorithm':<10} | {'PSNR (dB)':<10} | "
          f"{'SSIM':<8} | {'Time (s)':<10} | {'PSNR Improve %':<15} | {'SSIM Improve %':<15}")
    print("-" * 120)

    for _, row in summary_df.iterrows():
        # ISTA行
        print(f"{row['Noise Type']:<15} | {str(row['Intensity']):<12} | {'ISTA':<10} | "
              f"{row['ISTA_PSNR']:<10.2f} | {row['ISTA_SSIM']:<8.4f} | {row['ISTA_Time']:<10.4f} | "
              f"{'N/A':<15} | {'N/A':<15}")

        # BM3D行（如果有数据）
        if not np.isnan(row['BM3D_PSNR']):
            print(f"{' ':<15} | {' ':<12} | {'BM3D':<10} | "
                  f"{row['BM3D_PSNR']:<10.2f} | {row['BM3D_SSIM']:<8.4f} | {row['BM3D_Time']:<10.4f} | "
                  f"{row['PSNR_Improve']:<15.1f} | {row['SSIM_Improve']:<15.1f}")
            print("-" * 120)

    # 保存为LaTeX表格
    latex_path = os.path.join(results_dir, "summary_table.tex")
    with open(latex_path, 'w') as f:
        # 添加表格说明和多列格式
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Denoising Performance Comparison on Set14 Dataset}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lcccrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Noise Type & Intensity & Algorithm & PSNR (dB) & SSIM & Time (s) & PSNR Imp. (\%) & SSIM Imp. (\%) \\\\\n")
        f.write("\\midrule\n")

        for _, row in summary_df.iterrows():
            # ISTA行
            f.write(f"{row['Noise Type']} & {row['Intensity']} & ISTA & "
                    f"{row['ISTA_PSNR']:.2f} & {row['ISTA_SSIM']:.4f} & {row['ISTA_Time']:.4f} & -- & -- \\\\\n")
            # BM3D行
            if not np.isnan(row['BM3D_PSNR']):
                f.write(f" & & BM3D & {row['BM3D_PSNR']:.2f} & {row['BM3D_SSIM']:.4f} & {row['BM3D_Time']:.4f} & "
                        f"{row['PSNR_Improve']:.1f}\% & {row['SSIM_Improve']:.1f}\% \\\\\n")
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

        # 生成按噪声类型分组的对比表格
    noise_comparison = summary_df.groupby('Noise Type').agg({
        'ISTA_PSNR': 'mean',
        'BM3D_PSNR': 'mean',
        'PSNR_Improve': 'mean'
    })

    print("\n按噪声类型分组的PSNR对比:")
    print(noise_comparison.round(2))

    print(f"\n详细结果保存至: {csv_path}")
    print(f"汇总表格保存至: {summary_csv_path}")
    print(f"LaTeX表格保存至: {latex_path}")


def main():
    # 设置随机种子以确保结果可重现
    np.random.seed(42)

    # 创建结果目录
    results_dir = "denoising_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "per_image_results"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "convergence"), exist_ok=True)

    # 加载Set14数据集
    dataset_path = "/project/week2/Set14"
    image_paths = glob.glob(os.path.join(dataset_path, "*.png"))

    if not image_paths:
        print(f"在 {dataset_path} 中未找到图像文件!")
        return

    print(f"找到 {len(image_paths)} 张图像")

    # 噪声配置
    noise_configs = [
        {'type': 'Gaussian', 'intensity': 15},
        {'type': 'Gaussian', 'intensity': 25},
        {'type': 'Gaussian', 'intensity': 50},
        {'type': 'Salt-Pepper', 'intensity': 0.05},
        {'type': 'Salt-Pepper', 'intensity': 0.10},
        {'type': 'Salt-Pepper', 'intensity': 0.20}
    ]

    # 存储所有结果
    all_results = []
    clean_images = []  # 用于步长分析

    # 处理每张图像
    for img_path in tqdm(image_paths, desc="处理Set14图像"):
        # 加载干净图像
        try:
            clean_img = load_image(img_path)
            clean_images.append(clean_img)
        except Exception as e:
            print(f"跳过图像 {img_path}: {e}")
            continue

        # 处理当前图像
        img_results = process_image(
            img_path,
            results_dir,
            noise_configs,
            max_iter=100,
            step_size=0.8
        )

        all_results.extend(img_results)

    # 生成汇总表格
    generate_summary_table(all_results, results_dir)

    # 分析步长对收敛性的影响
    analyze_step_size_impact(clean_images, results_dir, noise_configs)

    print("\n实验完成! 所有结果保存在以下目录:")
    print(f"1. 每张图像的去噪结果: {os.path.join(results_dir, 'per_image_results')}")
    print(f"2. 收敛曲线: {os.path.join(results_dir, 'convergence')}")
    print(f"3. 汇总表格: {os.path.join(results_dir, 'summary_table.csv')}")
    print(f"4. 步长影响分析: {os.path.join(results_dir, 'step_size_impact.png')}")


if __name__ == "__main__":
    main()