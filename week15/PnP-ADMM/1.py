import torch
import torch.nn.functional as F
from model import Unet
from pnp import pnp_admm
from utils import conv2d_from_kernel, compute_psnr, ImagenetDataset
from PIL import Image
import os
import matplotlib.pyplot as plt


# ============================================================
# 设置设备
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')


# ============================================================
# 结果保存路径
# ============================================================
save_dir = 'results2'
os.makedirs(save_dir, exist_ok=True)


# ============================================================
# 自定义 myplot 并保存图片
# ============================================================
def save_myplot(degraded, reconstruction, target, save_path, title='Result'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(degraded.permute(0, 2, 3, 1).squeeze().cpu())
    plt.title('Degraded')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstruction.permute(0, 2, 3, 1).squeeze().cpu())
    plt.title('Reconstruction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(target.permute(0, 2, 3, 1).squeeze().cpu())
    plt.title('Ground Truth')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved result figure: {save_path}")


# ============================================================
# 加载模型
# ============================================================
model = Unet(in_chans=3, out_chans=3, chans=64).to(device)
state_dict = torch.load('denoiser.pth', map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================
# 加载测试图像
# ============================================================
test_image = Image.open('figs/ppt3.png').convert("RGB")
transform = ImagenetDataset([]).test_transform
test_image = transform(test_image)
channels, h, w = test_image.shape
test_image = test_image.unsqueeze(0).to(device)  # [1,3,H,W]


# ============================================================
# 任务 1: Motion Deblur（运动模糊）
# ============================================================
kernel_size = 21
kernel_motion_blur = torch.ones((1, kernel_size))
forward, forward_adjoint = conv2d_from_kernel(kernel_motion_blur, channels, device)

y = forward(test_image)

with torch.no_grad():
    x = pnp_admm(y, forward, forward_adjoint, model, num_iter=50)
    x = x.clamp(0, 1)

psnr_val = compute_psnr(x, test_image)
print(f'[Motion Deblur] PSNR [dB]: {psnr_val:.2f}')

save_myplot(F.pad(y, (kernel_size // 2, kernel_size // 2)), x, test_image,
             os.path.join(save_dir, 'motion_deblur.png'),
             title=f'Motion Deblur (PSNR={psnr_val:.2f}dB)')


# ============================================================
# 任务 2: Inpainting（随机遮挡修复）
# ============================================================
mask = (torch.rand(1, 1, h, w, device=device) < 0.2).float()

def forward_inpaint(x):
    return x * mask

forward = forward_inpaint
forward_adjoint = forward_inpaint
y = forward(test_image)

with torch.no_grad():
    x = pnp_admm(y, forward, forward_adjoint, model, num_iter=100)
    x = x.clamp(0, 1)

psnr_val = compute_psnr(x, test_image)
print(f'[Inpainting] PSNR [dB]: {psnr_val:.2f}')

save_myplot(y, x, test_image,
             os.path.join(save_dir, 'inpainting.png'),
             title=f'Inpainting (PSNR={psnr_val:.2f}dB)')


# ============================================================
# 任务 3: Super-Resolution（超分辨率）
# ============================================================
kernel_size = 4
kernel_downsampling = torch.ones((kernel_size, kernel_size))
forward, forward_adjoint = conv2d_from_kernel(kernel_downsampling, channels, device, stride=kernel_size)

y = forward(test_image)

with torch.no_grad():
    x = pnp_admm(y, forward, forward_adjoint, model,
                 num_iter=100, max_cgiter=30, cg_tol=1e-4)
    x = x.clamp(0, 1)

psnr_val = compute_psnr(x, test_image)
print(f'[Super-resolution] PSNR [dB]: {psnr_val:.2f}')

save_myplot(y, x, test_image,
             os.path.join(save_dir, 'super_resolution.png'),
             title=f'Super-Resolution (PSNR={psnr_val:.2f}dB)')
