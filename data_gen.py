import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from kornia.filters import filter2d

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
parser.add_argument('--coco_path', type=str, default=r'/data/unagi0/cui_data/coco',
                    help='the place you store COCO validation dataset')
parser.add_argument('--data_type', type=str, default=r'val',
                    help='choose train or val')
opt = parser.parse_args()

image_path = os.path.join(opt.coco_path, '%s2017'%opt.data_type)
label_path = os.path.join(opt.coco_path, 'annotations', 'instances_%s2017.json'%opt.data_type)
device = 'cuda:0'


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Random Gaussian Noise, noise level: sigma ~ (5, 50)
def random_noise(img):
    device = img.device
    img_hr = img.permute(2, 0, 1)  
    img_hr = img_hr.unsqueeze(0)    # (H,W,C) to (1,C,H,W)
    noise_level_img = random.uniform(5, 50) / 255.0
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_hr.shape).to(torch.device(device))
    img_hr += noise

    return img_hr.squeeze(0).permute(1, 2, 0)

# Fix Gaussian Noise, noise level sigma is fixed
def fix_noise(img, n=10.0):
    device = img.device
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    img_hr = img_hr.unsqueeze(0)
    noise_level_img = n / 255.0
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_hr.shape).to(torch.device(device))
    img_hr += noise

    return img_hr.squeeze(0).permute(1, 2, 0)

def random_blur(img):
    # device = img.device
    device = img.device
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
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

# Fix isotropic Gaussian blur, you could adjust kernel size and width
def fix_isoblur(img, kernel_size=17, kernel_width_iso=3.0):
    
    device = img.device
    img_hr = img.permute(2, 0, 1)
    img_hr = img_hr.unsqueeze(0)

    k1 = isotropic_Gaussian(kernel_size, kernel_width_iso)
    k_ts = torch.from_numpy(k1).unsqueeze(0).to(torch.device(device))
    img_blur = filter2d(img_hr, k_ts)

    return img_blur.squeeze(0).permute(1, 2, 0)

# Fix anisotropic Gaussian blur, you could adjust kernel size and width and angle
def fix_anisoblur(img, kernel_size=17, kernel_width_1=5, kernel_width_2 = 2.3, angle = np.pi/3):
    
    device = img.device
    img_hr = img.permute(2, 0, 1)  
    img_hr = img_hr.unsqueeze(0)
    k2 = anisotropic_Gaussian(kernel_size, angle, kernel_width_1, kernel_width_2)
    k_ts = torch.from_numpy(k2).unsqueeze(0).to(torch.device(device))
    img_blur = filter2d(img_hr, k_ts)

    return img_blur.squeeze(0).permute(1, 2, 0)

def down_sample(img, resolution=2):
    img_hr = img.permute(2, 0, 1)  # change (H,W,C) to (1,C,H,W)
    img_hr = img_hr.unsqueeze(0)
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
    img_lr = interpolate(img_hr, scale_factor=1 / resolution, mode=scale_mode)
    return img_lr.squeeze(0).permute(1, 2, 0)

def down_json(json_file=label_path, resolution=2):
    data = json.load(open(label_path, 'r'))
    
    image_dicts = data['images']
    anno_dicts = data['annotations']
    
    for i_image in range(len(image_dicts)):
        image_dicts[i_image]['width'] = int(image_dicts[i_image]['width']//resolution)
        image_dicts[i_image]['height'] = int(image_dicts[i_image]['height']//resolution)
    
    for i_anno in range(len(anno_dicts)):
        x, y, w, h = anno_dicts[i_anno]['bbox'][0], anno_dicts[i_anno]['bbox'][1], anno_dicts[i_anno]['bbox'][2], anno_dicts[i_anno]['bbox'][3]
        anno_dicts[i_anno]['bbox'] = [int(x//resolution), int(y//resolution), int(w//resolution), int(h//resolution)]
    
    data_out = json.dumps(data)
    json_store = json_file.replace('.json', 'down_%s.json'%str(resolution))
    f = open(json_store, 'w')
    f.write(data_out)
    f.close()
    
    
    

# Random Generation a degradation dataset
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

    return img_dr.squeeze(0).permute(1, 2, 0)


def main(img_path, json_path, deg_type='random_blur', resolution=2):

    assert deg_type in ['random_blur', 'fix_isoblur', 'fix_anisoblur',
                        'random_noise', 'fix_noise',
                        'resolution']
    if deg_type != 'resolution':
        img_out_path = img_path + deg_type
    else:
        img_out_path = img_path + deg_type + 'down_%s'%str(resolution)

    # Radom Blurry
    if deg_type == 'random_blur':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = random_blur(img_ts)
            img_deg = img_deg_ts.cpu().numpy()
            print(os.path.join(img_out_path, img_name))
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)

    # Fix Isotropic Gaussian Blurry
    elif deg_type == 'fix_isoblur':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = fix_isoblur(img_ts, kernel_size=17, kernel_width_iso=3.0)  # change hyper-parameter here
            img_deg = img_deg_ts.cpu().numpy()
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)

    # Fix Anisotropic Gaussian Blurry
    elif deg_type == 'fix_anisoblur':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = fix_anisoblur(img_ts, kernel_size=17, kernel_width_1=5, kernel_width_2 = 2.3, angle = np.pi/3)  # change hyper-parameter here
            img_deg = img_deg_ts.cpu().numpy()
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)

    # Random Noise
    elif deg_type == 'random_noise':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = random_noise(img_ts)  
            img_deg = img_deg_ts.cpu().numpy()
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)
    
    # Fix Noise
    elif deg_type == 'fix_noise':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = fix_noise(img_ts, n=10.0)  # change hyper-parameter here
            img_deg = img_deg_ts.cpu().numpy()
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)
    
    # Down-Sampling, need change the label json file 
    elif deg_type == 'resolution':
        for i, img_name in tqdm(enumerate(os.listdir(img_path))):
            img_full_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_full_name)/255.0
            img_ts = torch.from_numpy(img).to(torch.device(device))
            img_deg_ts = down_sample(img_ts, resolution)  # change hyper-parameter here
            img_deg = img_deg_ts.cpu().numpy()
            cv2.imwrite(os.path.join(img_out_path, img_name), img_deg * 255.0)
            down_json(json_path, resolution)    # Change Json File




if __name__ == "__main__":
    '''
    Choose Deg Type from:
    ['random_blur', 'fix_isoblur', 'fix_anisoblur',
    'random_noise', 'fix_noise', 'resolution']
    '''
    main(image_path, label_path, deg_type='random_blur')
