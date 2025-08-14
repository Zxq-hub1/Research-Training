# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# no need to run this code separately


import glob
# import os
import cv2
import numpy as np

# from multiprocessing import Pool


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    assert img is not None, f"无法读取图像: {file_name}"
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                # patches.append(x)
                # data aug
                assert x.ndim == 2
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


# def datagenerator(data_dir='data/Train400', verbose=False):
#     file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
#     # initrialize
#     data = []
#     # generate patches
#     for i in range(len(file_list)):
#         patch = gen_patches(file_list[i])
#         # data.append(patch)
#         data.extend(patch)
#         if verbose:
#             print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
#     data = np.array(data, dtype='uint8')
#     data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
#     discard_n = len(data) - len(data) // batch_size * batch_size;
#     data = np.delete(data, range(discard_n), axis=0)
#     print('^_^-training data finished-^_^')
#     return data


def datagenerator(data_dir='data/Train400', verbose=False):
    file_list = glob.glob(data_dir + '/*.png')
    data = []

    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        data.extend(patches)
        if verbose:
            print(f'{i + 1}/{len(file_list)} done')

    # 转换为 numpy 数组并添加通道维度
    data = np.array(data, dtype='uint8')
    print("原始数据形状:", data.shape)  # 调试

    # 确保是 4D (num_patches, h, w, channels)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)

    print("处理后形状:", data.shape)  # 调试

    # 调整 batch 大小
    discard_n = len(data) - len(data) // batch_size * batch_size
    if discard_n > 0:
        data = np.delete(data, range(discard_n), axis=0)

    print('最终数据形状:', data.shape)
    return data


if __name__ == '__main__':
    data = datagenerator(data_dir='data/Train400')

