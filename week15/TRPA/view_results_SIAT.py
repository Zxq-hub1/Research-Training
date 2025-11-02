import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import imageio
import csv

# ======= 路径配置 =======
data_root = './data/SIAT/test'                         # 原始测试数据 (.npy)
result_root = './results/SIAT/radial_10/spiral_030'   # 重建结果 (.mat)
save_png = './results/SIAT/spiral_results_030/spiral_030_png'             # 保存单张重建图
save_compare = './results/SIAT/spiral_results_030/spiral_030_compare'     # 保存对比图 (原图 vs 重建)
save_csv = './results/SIAT/spiral_results_030/spiral_030_metrics.csv'     # 保存 PSNR/SSIM 结果
os.makedirs(save_png, exist_ok=True)
os.makedirs(save_compare, exist_ok=True)

# ======= 获取文件列表 =======
files = sorted([f for f in os.listdir(result_root) if f.endswith('.mat')])
print(f'找到 {len(files)} 个重建结果。\n')

# ======= 初始化结果表 =======
results = [["filename", "PSNR", "SSIM"]]
psnr_list, ssim_list = [], []

# ======= 主循环 =======
for i, file_ in enumerate(files):
    file_name = file_.replace('.mat', '')
    mat_path = os.path.join(result_root, file_)
    npy_path = os.path.join(data_root, file_name + '.npy')

    # 加载重建结果
    rec_data = loadmat(mat_path)
    rec_im = np.abs(rec_data['rec_im'])
    rec_im = rec_im / np.max(rec_im)

    # 加载原始图像
    if not os.path.exists(npy_path):
        print(f"找不到原图 {npy_path}，跳过。")
        continue
    orig_im = np.load(npy_path)
    orig_im = np.abs(orig_im)
    orig_im = orig_im / np.max(orig_im)

    # ======= 计算 PSNR & SSIM =======
    psnr_val = psnr(orig_im, rec_im, data_range=1.0)
    ssim_val = ssim(orig_im, rec_im, data_range=1.0)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    results.append([file_name, f"{psnr_val:.4f}", f"{ssim_val:.4f}"])

    # ======= 显示并保存对比图 (两列) =======
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_im, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rec_im, cmap='gray')
    plt.title(f'Reconstructed\nPSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}')
    plt.axis('off')

    plt.suptitle(file_name, fontsize=11)
    plt.tight_layout()

    # 保存对比图
    compare_path = os.path.join(save_compare, file_name + '_compare.png')
    plt.savefig(compare_path, bbox_inches='tight', dpi=150)
    plt.close()

    # 保存单张重建图
    out_png = os.path.join(save_png, file_name + '.png')
    imageio.imwrite(out_png, (rec_im * 255).astype(np.uint8))

    print(f" [{i+1}/{len(files)}] {file_name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")
    print(f" 单张重建图保存到: {out_png}")
    print(f" 对比图保存到: {compare_path}\n")

# ======= 保存指标到 CSV =======
with open(save_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(results)
print(f"\n 所有结果已保存到: {save_csv}")

# ======= 打印平均与标准差 =======
if len(psnr_list) > 0:
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)

    print("\n================= 指标统计汇总 =================")
    print(f"平均 PSNR: {psnr_mean:.4f}  (±{psnr_std:.4f})")
    print(f"平均 SSIM: {ssim_mean:.4f}  (±{ssim_std:.4f})")
    print("====================================================\n")
else:
    print("没有有效的结果可统计。")
