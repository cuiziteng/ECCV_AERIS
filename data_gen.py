import torch
import torch.nn as nn
from torch.nn.functional import interpolate

import kornia
from kornia.filters import gaussian_blur2d, motion_blur, filter2d
import sys

from kernel import isotropic_Gaussian, anisotropic_Gaussian

import os
import random
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default=r'/data/unagi0/cui_data/coco/val2017',
                    help='the place you store ISIC dataset')
parser.add_argument('--target', type=str, default=r'/data/unagi0/cui_data/coco/val2017_kernel2',
                    help='the place to store the output images')

# parser.add_argument('--with_low', type=bool, default=True)
opt = parser.parse_args()
print(opt)

data_root_coco = r'/home/czt/mmdetection/data/coco/val2017'
device = 'cuda:3'


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


mkdir(opt.target)
# if opt.with_low:
#     mkdir(opt.target+'_low')


def random_noise(img):
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    noise_level_img = random.uniform(5, 50) / 255.0
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_hr.shape).to(torch.device(device))
    img_hr += noise

    return img_hr.squeeze(0).permute(1, 2, 0)

def fix_noise(img, n=10.0):
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    noise_level_img = n / 255.0
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_hr.shape).to(torch.device(device))
    img_hr += noise

    return img_hr.squeeze(0).permute(1, 2, 0)

def random_blur(img):
    # device = img.device
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
    kernel_width_iso = random.uniform(0.1, 2.4)
    resolution = 2
    angle = random.uniform(0, np.pi)
    kernel_width_un1 = random.uniform(0.5, 6)
    kernel_width_un2 = random.uniform(0.5, kernel_width_un1)
    
    deg_type = random.choice(['iso', 'aniso'])

    if deg_type == 'iso':  # adopt isotropic Gaussian blur
        k = isotropic_Gaussian(kernel_size, kernel_width_iso)
        k_ts = torch.from_numpy(k).unsqueeze(0).to(torch.device(device))
        img_blur = filter2d(img_hr, k_ts)

    else:  # adopt anisotropic Gaussian blur
        k = anisotropic_Gaussian(kernel_size, angle, kernel_width_un1, kernel_width_un2)
        k_ts = torch.from_numpy(k).unsqueeze(0).to(torch.device(device))
        img_blur = filter2d(img_hr, k_ts)


    return img_blur.squeeze(0).permute(1, 2, 0)

def fix_isoblur(img):
    # device = img.device
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    # kernel_size = 17
    # kernel_width_iso = 2.0

    k1 = isotropic_Gaussian(17, 2.4)
    k_ts = torch.from_numpy(k1).unsqueeze(0).to(torch.device(device))
    img_blur = filter2d(img_hr, k_ts)

    return img_blur.squeeze(0).permute(1, 2, 0)

def fix_anisoblur(img):
    # device = img.device
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    # kernel_size = 17
    # kernel_width_iso = 2.0
    k2 = anisotropic_Gaussian(17, np.pi/3, 5, 2.3)
    k3 = anisotropic_Gaussian(21, 2*np.pi/3, 6, 2.5)
    k_ts = torch.from_numpy(k2).unsqueeze(0).to(torch.device(device))
    img_blur = filter2d(img_hr, k_ts)

    return img_blur.squeeze(0).permute(1, 2, 0)

def down_2(img):
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    resolution = 2
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
    img_lr = interpolate(img_hr, scale_factor=1 / resolution, mode=scale_mode)
    return img_lr.squeeze(0).permute(1, 2, 0)

def down_4(img):
    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    resolution = 4
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
    img_lr = interpolate(img_hr, scale_factor=1 / resolution, mode=scale_mode)
    return img_lr.squeeze(0).permute(1, 2, 0)

def random_deg(img):
    '''
    input:
    img (Tensor): Input images of shape (H,W,C).
    resolution (tuple): Resolution number, range from 1~4
    keep_shape (bool): Choose to return same size or lower size

    return:
    keep_shape = True: img_low (Tensor): Output degraded images of shape (H,W,C).
    keep_shape = False: img_low (Tensor): Output degraded images of shape (H/ratio,W/ratio,C).
    '''

    device = img.device
    # print(img.shape)
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    # print(img_hr.shape)
    img_hr = img_hr.unsqueeze(0)
    kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
    kernel_width_iso = random.uniform(0.1, 2.4)
    resolution = 2
    angle = random.uniform(0, np.pi)
    kernel_width_un1 = random.uniform(0.5, 6)
    kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

    # random choose from three degradation types
    deg_type = random.choice(['none', 'iso', 'aniso', 'both'])
    # random choose from three interpolate types
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])

    # Choose to use or not use blur
    if deg_type == 'none':  # adopt none blur
        img_blur = img_hr

    elif deg_type == 'iso':  # adopt isotropic Gaussian blur
        k = isotropic_Gaussian(kernel_size, kernel_width_iso)
        k_ts = torch.from_numpy(k).unsqueeze(0).to(torch.device(device))
        img_blur = filter2d(img_hr, k_ts)

    elif deg_type == 'aniso':  # adopt anisotropic Gaussian blur
        k = anisotropic_Gaussian(kernel_size, angle, kernel_width_un1, kernel_width_un2)
        k_ts = torch.from_numpy(k).unsqueeze(0).to(torch.device(device))
        img_blur = filter2d(img_hr, k_ts)

    # Down Sampling, random choose from 1~4
    img_dr = interpolate(img_blur, scale_factor=1 / resolution, mode=scale_mode)

    # add noise
    noise_level_img = random.uniform(0, 25) / 255.0
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_dr.shape).to(torch.device(device))
    img_dr += noise
    # Up Sampling
    #img_lr = interpolate(img_dr, (img_blur.shape[2], img_blur.shape[3]), mode='bicubic')

    return img_dr.squeeze(0).permute(1, 2, 0)




old_root = opt.source
new_root = opt.target

for i, img_name in tqdm(enumerate(os.listdir(old_root))):
    # print(i)
    # print(img_name)
    img_full_name = os.path.join(old_root, img_name)
    img_save_name = os.path.join(new_root, img_name)
    
    img = cv2.imread(img_full_name) / 255.0
    # down sample rate random choose from 1~4
    # s = random.randint(10, 40) / 10

    # turn numpy to tensor
    img_ts = torch.from_numpy(img).to(torch.device(device))
    # print(img_ts.shape)
    img_deg_ts = fix_anisoblur(img_ts)
    # print(img_deg_ts.shape)
    img_deg = img_deg_ts.cpu().numpy()
    # print(img_deg.shape)
    cv2.imwrite(img_save_name, img_deg * 255.0)
    

