import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from model.LSTM_attention import *
from model.RIFE import Model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from model.VimeoSeptuplet import *

device = torch.device("cuda")

log_path = 'train_log'
intrain_path = 'intrain_log'

from model.pretrained_RIFE_loader import IFNet_update
from model.pretrained_RIFE_loader import convert_load




def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_tea_pred_list = []
    loss_mse_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    addImageFlag =1

    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device, non_blocking=True)
        # Changed for Septuplet 0 1 2 3 4 5 6
        b_data_gpu, _,h_data_gpu,w_data_gpu = data_gpu.shape
        display_all = data_gpu.view(b_data_gpu,7,3,h_data_gpu,w_data_gpu) 

        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        # learning_rate = get_learning_rate(step) * args.world_size / 4
        # pred, info = model.update(data_gpu, learning_rate, training=True) # pass timestep if you are training RIFEm        
        gt = []
        for i in range(3):
            # img0 = data_gpu[:, 6*i:6*i+3]
            gt.append(data_gpu[:, 6*i+3:6*i+6])
            # img1 = data_gpu[:, 6*i+6:6*i+9]
        gt = torch.stack(gt,dim=1) # B*N*C*H*W
        with torch.no_grad():
            pred, teapred, info = model.update(data_gpu, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['Sum_loss_context'].cpu().numpy())
        loss_mse_list.append(info['Sum_loss_mse'].cpu().numpy())
        loss_tea_pred_list.append(info['Sum_loss_tea_pred'].cpu().numpy())
        for j in range(gt.shape[0]):
            setpsnr = torch.tensor(0.0)
            for k in range(3):
                epsilon = 1e-8  # Small value to avoid log10(0)
                mse = torch.mean((gt[j][k] -validCheck pred[j][k]) ** 2).cpu().data  # Better to use **2 for clarity
                single_psnr = -10 * math.log10(mse + epsilon)
                # single_psnr = -10 * math.log10(torch.mean((gt[j][k] - pred[j][k]) * (gt[j][k] - pred[j][k])).cpu().data)
                setpsnr = setpsnr + single_psnr
            setpsnr = setpsnr / 3
            psnr_list.append(setpsnr)
            # print("--\n")
            # psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            # psnr_list_teacher.append(psnr)
        # gt = (gt[][:,-3:].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # pred = (pred[:,-6:-3].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
            # for j in range(10):
                # imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                # writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                # writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
        # eval_time_interval = time.time() - time_stamp
    #print(".\n")
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.flush()
    # writer_val.add_scalar('psnr_tea


         