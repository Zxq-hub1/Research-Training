import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt


def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net).cuda()
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     net = net.cuda()
    return net


def save_model(net, optimizer, epoch, save_dir, scheduler=None):
    '''save model'''

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    if scheduler is None:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    else:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()
            # 'lr_scheduler': scheduler.state_dict()
        },
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))


def ssim(img1, img2, window_size=11, size_average=True):
    # Constants for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Normalize images to [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    # Mean of both images
    mu1 = F.avg_pool2d(img1, window_size)
    mu2 = F.avg_pool2d(img2, window_size)

    # Covariance and variance
    mu1_mu2 = mu1 * mu2
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    mse = F.mse_loss(img1, img2)
    psnr_value = 10 * torch.log10(1.0 / mse)
    return psnr_value


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_noisePair(image, noise_type='gaussian', mode=None):
    # Convert tensor to numpy array
    image_np = image.numpy()

    # Determine noise parameters based on mode
    if mode == 'train':
        if noise_type == 'gaussian':
            std = np.random.randint(0, 51)  # Random standard deviation between 0 and 50
        elif noise_type == 'poisson':
            lam = np.random.randint(0, 51)  # Random λ between 0 and 50
    else:
        std = 25  # Default standard deviation for Gaussian noise
        lam = 30  # Default λ for Poisson noise

    # Add noise based on noise_type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, std, size=image_np.shape)
        source = image_np + noise

        noise = np.random.normal(0, std, size=image_np.shape)
        target = image_np + noise
    elif noise_type == 'poisson':
        noise = np.random.poisson(lam, size=image_np.shape)
        source = image_np + noise

        noise = np.random.poisson(lam, size=image_np.shape)
        target = image_np + noise

    # Clip noisy image values to ensure they are within [0, 255] range
    source = np.clip(source, 0, 255)

    # Convert back to tensor
    source = torch.tensor(source, dtype=torch.float32)

    target = np.clip(target, 0, 255)

    target = torch.tensor(target, dtype=torch.float32)

    return source, target


def tensor_to_rgb(image_tensor):
    # 转换为CHW维度
    image_tensor = image_tensor.cpu()
    image_chw = image_tensor.squeeze(0)
    # 将张量值范围调整到 [0, 1]
    image_chw = image_chw.clamp(0, 255).to(torch.uint8)
    # 转换为RGB格式
    image_rgb = transforms.ToPILImage()(image_chw)
    return image_rgb


# 保存RGB图像
def save_rgb_image(image_tensor, file_path):
    # 转换为RGB图像
    image_rgb = tensor_to_rgb(image_tensor)
    # 保存图像
    image_rgb.save(file_path)


def show_rgb_image(image_tensor1, image_tensor2,image_tensor3,title1='Noise1', title2='Noise2',title3='Result'):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 将张量转换为RGB图像
    img_rgb1 = tensor_to_rgb(image_tensor1)
    img_rgb2 = tensor_to_rgb(image_tensor2)
    img_rgb3=tensor_to_rgb(image_tensor3)

    # 将RGB图像转换为numpy数组
    image_np1 = np.array(img_rgb1)
    image_np2 = np.array(img_rgb2)
    image_np3=np.array(img_rgb3)

    # 显示图像
    plt.figure(figsize=(10, 5))  # 设置图像大小
    plt.subplot(1, 3, 1)  # 第一个子图
    plt.imshow(image_np1)
    plt.title(title1)
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(1, 3, 2)  # 第二个子图
    plt.imshow(image_np2)
    plt.title(title2)
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(1, 3, 3)  # 第二个子图
    plt.imshow(image_np3)
    plt.title(title3)
    plt.axis('off')  # 关闭坐标轴

    return fig

