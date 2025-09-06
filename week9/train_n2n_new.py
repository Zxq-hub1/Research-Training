#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise2Noise 彩色图像去噪 —— DIV2K 训练脚本
Author : you
"""
import os
import time
import random
import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torch.optim.lr_scheduler import CosineAnnealingLR

# --------------------------- 网络 ----------------------------------
class SimpleNoise2Noise(nn.Module):
    """简易顺序 Encoder-Decoder，无 skip，彩色 3→3"""
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        self.encoder = nn.Sequential(
            # 128
            nn.Conv2d(in_c, 64, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 64
            nn.Conv2d(64, 128, 3, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 32
            nn.Conv2d(128, 256, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(64, out_c, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.decoder(self.encoder(x)))


# --------------------------- 数据 ----------------------------------
class DIV2KDataset(Dataset):
    def __init__(self, root, patch_size=128, noise_level=25, mode='train'):
        assert mode in ('train', 'val')
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.mode = mode
        self.im_list = sorted(glob.glob(os.path.join(root, '**', '*.*'), recursive=True))
        if len(self.im_list) == 0:
            raise FileNotFoundError(f'No images found in {root}')
        print(f'[{mode}] found {len(self.im_list)} images')

    def __len__(self):
        return len(self.im_list)

    @staticmethod
    def _add_noise(img, sigma):
        """img: 0~1 float32  ->  0~1 float32"""
        noise = np.random.normal(0, sigma/255., img.shape).astype(np.float32)
        return np.clip(img + noise, 0., 1.)

    def _crop(self, img):
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
            h = w = self.patch_size
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        return img[y:y+self.patch_size, x:x+self.patch_size]

    def _augment(self, img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        return img

    def __getitem__(self, idx):
        img = cv2.imread(self.im_list[idx])
        if img is None:
            img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            img = self._augment(img)
            img = self._crop(img)
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size))

        img = img.astype(np.float32) / 255.
        # Noise2Noise 需要两次独立噪声
        n1 = self._add_noise(img, self.noise_level)
        n2 = self._add_noise(img, self.noise_level)
        return torch.from_numpy(n1).permute(2,0,1), torch.from_numpy(n2).permute(2,0,1)


# --------------------------- 训练 ----------------------------------
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    train_set = DIV2KDataset(args.train_dir, args.patch_size, args.noise_level, 'train')
    val_set   = DIV2KDataset(args.val_dir,   args.patch_size, args.noise_level, 'val')
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 模型
    net = SimpleNoise2Noise(3, 3).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        print(f'resume from {args.resume}')

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 1e9
    for ep in range(1, args.epochs+1):
        # ---- train ----
        net.train()
        running_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {ep}/{args.epochs}')
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            rec = net(src)
            loss = nn.MSELoss()(rec, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()*src.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = running_loss/len(train_set)

        # ---- val ----
        net.eval()
        running_loss = 0.
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                rec = net(src)
                loss = nn.MSELoss()(rec, tgt)
                running_loss += loss.item()*src.size(0)
        val_loss = running_loss/len(val_set)
        sched.step()

        print(f'Epoch {ep:03d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f} | lr={opt.param_groups[0]["lr"]:.2e}')

        # ---- save ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch':ep, 'model_state_dict':net.state_dict(), 'val_loss':val_loss},
                       os.path.join(args.out_dir, 'best.pth'))
            print('  *best*')
        if ep % 10 == 0:
            torch.save(net.state_dict(), os.path.join(args.out_dir, f'epoch{ep}.pth'))

        # ---- sample ----
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                src, tgt = next(iter(val_loader))
                src, tgt = src[:4].to(device), tgt[:4]
                rec = net(src)
                grid = torch.cat([src.cpu(), rec.cpu(), tgt], dim=0)
                vutils.save_image(grid, os.path.join(args.out_dir, f'sample_ep{ep}.png'), nrow=4)

    print('Training finished. Best val MSE =', best_val)


# --------------------------- 入口 ----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/DIV2K_train')
    parser.add_argument('--val_dir',   default='data/DIV2K_valid')
    parser.add_argument('--out_dir',   default='models/N2N_DIV2K')
    parser.add_argument('--patch_size',type=int, default=128)
    parser.add_argument('--noise_level',type=int,default=25, help='gaussian sigma')
    parser.add_argument('--bs',        type=int, default=16)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--resume',    type=str, default='', help='path to checkpoint')
    args = parser.parse_args()

    train(args)