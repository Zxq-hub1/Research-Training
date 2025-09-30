from __future__ import print_function
import matplotlib.pyplot as plt
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *


import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *

import cv2
import os


# 2. 全局变量：记录最优（或最终）结果
best_psnr_gt = 0.0
best_out_np  = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.

# deJPEG
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/denoising/ppt3.png'


# 1. 构造保存目录（与 fname 完全同名文件夹）
base_name = os.path.splitext(os.path.basename(fname))[0]
save_dir  = os.path.join(os.path.dirname(fname), base_name)
os.makedirs(save_dir, exist_ok=True)

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)

    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np

    if PLOT:
        plot_image_grid([img_np], 4, 5);

elif fname == 'data/denoising/F16_GT.png' or fname == 'data/denoising/ppt3.png':
    # 加合成噪声
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False

INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3
    figsize = 5

    net = skip(
        input_depth, 3,
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

elif fname == 'data/denoising/F16_GT.png'or fname == 'data/denoising/ppt3.png':
    num_iter = 3000
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0


def closure():
    global i, out_avg, psrn_noisy_last, last_net, net_input
    global best_psnr_gt, best_out_np, best_out_avg_np  # 新增

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    global best_psnr_gt, best_out_np  # 新增
    if psrn_gt > best_psnr_gt:  # 新增
        best_psnr_gt = psrn_gt  # 新增
        best_out_np = torch_to_np(out)  # 新增

    # 记录最优（或最后一次）结果
    if psrn_gt > best_psnr_gt:
        best_psnr_gt = psrn_gt
        best_out_np = torch_to_np(out)
        best_out_avg_np = torch_to_np(out_avg)

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
    i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss

# 训练结束后画图+保存
def save_result_with_psnr():
    out_clipped = np.clip(best_out_np, 0, 1).transpose(1,2,0)   # CHW->HWC
    # 转成 0-255 BGR 供 cv2 使用
    out_bgr = (out_clipped*255)[:,:,::-1].copy()
    h, w = out_bgr.shape[:2]

    text = f'PSNR_gt: {best_psnr_gt:.2f} dB'
    cv2.putText(out_bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    save_path = os.path.join(save_dir, base_name+'_result.png')
    cv2.imwrite(save_path, out_bgr)
    print('结果已保存至:', save_path)


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

save_result_with_psnr()          # 新增：保存带 PSNR 标注的结果图

out_np = torch_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);

