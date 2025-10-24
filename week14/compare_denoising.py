import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti

# 创建 results 目录
os.makedirs('results', exist_ok=True)

# 加载数据
original_data, _ = load_nifti('./data/HARDI150.nii.gz')
denoised_data = nib.load('experiments/hardi150_denoise_251020_130810/my_denoised_output/hardi150_denoised.nii.gz').get_fdata()

# 归一化
original_normalized = (original_data - original_data.min()) / (original_data.max() - original_data.min())
denoised_normalized = (denoised_data - denoised_data.min()) / (denoised_data.max() - denoised_data.min())

# 参数
slice_idx = 40
gradient_idx = 0
diff = np.abs(original_normalized[:, :, slice_idx, gradient_idx] - denoised_normalized[:, :, slice_idx, gradient_idx])

# 画图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# 原始
im0 = axes[0].imshow(original_normalized[:, :, slice_idx, gradient_idx], cmap='gray')
axes[0].set_title('Original Data', fontsize=14, fontweight='bold')
axes[0].text(0.5, -0.1, 'Original with Noise', transform=axes[0].transAxes, ha='center', fontsize=11, style='italic')

# 去噪
im1 = axes[1].imshow(denoised_normalized[:, :, slice_idx, gradient_idx], cmap='gray')
axes[1].set_title('Denoised Data', fontsize=14, fontweight='bold')
axes[1].text(0.5, -0.1, 'DDM² Denoised', transform=axes[1].transAxes, ha='center', fontsize=11, style='italic')

# 差异
im2 = axes[2].imshow(diff, cmap='hot')
axes[2].set_title('Difference Map', fontsize=14, fontweight='bold')
axes[2].text(0.5, -0.1, 'Removed Noise', transform=axes[2].transAxes, ha='center', fontsize=11, style='italic')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label('差异强度', rotation=270, labelpad=15)

plt.tight_layout()

# 保存
output_path = 'results/compare_denoising.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"图像已保存至：{output_path}")

plt.show()