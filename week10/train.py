import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CustomDataset
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
import time
import torch.optim as optim
import argparse
from models import UNet
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mse=torch.nn.MSELoss()

def train(train_loader, model, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    loss_img = AverageMeter()
    model.train()
    end = time.time()

    step = 0

    for data in train_loader:

        source=data["source"]
        source=source.cuda()

        target=data["target"]
        target=target.cuda()

        pred=model(source)

        loss=mse(pred,target)
        # loss = 0.01 * loss1 + loss3

        loss_img.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('train_loss', {'loss_img': loss_img.avg}, epoch + 1)
    scheduler.step()

    print('Train Epoch: {}\t train_loss: {:.6f}\t'.format(epoch + 1, loss_img.avg))

def valid(valid_loader, model, writer, epoch):

    batch_time = AverageMeter()
    loss_img = AverageMeter()
    model.eval()
    end = time.time()

    step = 0

    for data in valid_loader:

        source=data["source"]
        source=source.cuda()

        target=data["target"]
        target=target.cuda()

        with torch.no_grad():

            pred = model(source)

            loss=mse(pred,target)

        loss_img.update(loss.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('valid_loss', {'loss_img': loss_img.avg}, epoch + 1)

    print('Valid Epoch: {}\t valid_loss: {:.6f}\t'.format(epoch + 1, loss_img.avg))

if __name__ == "__mFain__":

    noise_type='gaussian'

    train_dir='data/DIV2K_train'
    valid_dir='data/DIV2K_valid'
    
    method = 'Noise2Noise'
    result_path = './runs/' + method +noise_type+ '/logs/'
    save_dir = './runs/' + method +noise_type+ '/checkpoints/'

    batch_size = 4

    # Get dataset
    train_dataset = CustomDataset(data_dir=train_dir,noise_type=noise_type,mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = CustomDataset(data_dir=valid_dir,noise_type=noise_type,mode=None)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.6)

    model=model.cuda()

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, writer, epoch)

        save_model(model, optimizer, epoch + 1, save_dir)