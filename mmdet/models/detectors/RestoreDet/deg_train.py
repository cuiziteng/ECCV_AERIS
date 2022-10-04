import random
import torch
from torch.nn.functional import interpolate
import numpy as np
from .kernel import isotropic_Gaussian, anisotropic_Gaussian
from kornia.filters import filter2D

# Low-resolution Transformation
def deg_resolution(img, img_meta, l_size, gt_bbox):
    img_hr = img.unsqueeze(0)  # (1, C, H, W)
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
    # Down Sampling, rate random choose from 1~4
    img_dr = interpolate(img_hr, size=(l_size, l_size), mode=scale_mode)
    s = 512/l_size

    gt_bbox_down = gt_bbox/s
    new_h, new_w = int(img_meta['pad_shape'][0]/s), int(img_meta['pad_shape'][1]/s)
    img_meta['pad_shape'] = (new_h, new_w, 3)
    return img_dr.squeeze(0), gt_bbox_down


# Mixed Degradation Transformation
def random_deg(img, img_meta, l_size, gt_bbox, keep_shape=False):
    '''
    input:
    img (Tensor): Input images of shape (C, H, W).
    img_meta (dict): A image info dict contain some information like name ,shape ...
    keep_shape (bool): Choose to return same size or lower size
    s (float): down-sampling scale factor ,from 1~4
    gt_bbox (Tensor): original bbox label of image

    return:
    keep_shape = True: img_low (Tensor): Output degraded images of shape (C, H, W).
    keep_shape = False: img_low (Tensor): Output degraded images of shape (C, H/ratio, W/ratio).
    '''
    device = img.device

    # here use unsqueeze, this is because torch's bicubic can only implenment on 4D tenspr
    img_hr = img.unsqueeze(0)  # (1, C, H, W)
    kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
    kernel_width_iso = random.uniform(0.1, 2.4)
    angle = random.uniform(0, np.pi)
    kernel_width_un1 = random.uniform(0.5, 6)
    kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

    # random choose from three degradation types
    deg_type = random.choice(['none', 'iso', 'aniso'])
    # random choose from three interpolate types
    scale_mode = random.choice(['nearest', 'bilinear', 'bicubic'])

    # Choose to use or not use blur
    if deg_type == 'none':  # adopt none blur
        img_blur = img_hr

    elif deg_type == 'iso':  # adopt isotropic Gaussian blur
        k = isotropic_Gaussian(kernel_size, kernel_width_iso)
        k_ts = torch.from_numpy(k).to(torch.device(device)).unsqueeze(0)
        img_blur = filter2D(img_hr, k_ts)

    elif deg_type == 'aniso':  # adopt anisotropic Gaussian blur
        k = anisotropic_Gaussian(kernel_size, angle, kernel_width_un1, kernel_width_un2)
        k_ts = torch.from_numpy(k).to(torch.device(device)).unsqueeze(0)
        img_blur = filter2D(img_hr, k_ts)


    if keep_shape:
        img_dr = img_blur
        noise_level_img = random.uniform(0, 50) / 255.0

    # Down Sampling, rate random choose from 1~4
    else:
        img_dr = interpolate(img_blur, size=(l_size, l_size), mode=scale_mode)
        s = 512/l_size
        # add noise
        noise_level_img = random.uniform(0, 25) / 255.0
    
    noise = torch.normal(mean=0.0, std=noise_level_img, size=img_dr.shape).to(torch.device(device))
    img_dr += noise

    if not keep_shape:
        gt_bbox_down = gt_bbox/s
        new_h, new_w = int(img_meta['pad_shape'][0]/s), int(img_meta['pad_shape'][1]/s)
        img_meta['pad_shape'] = (new_h, new_w, 3)
    
        return img_dr.squeeze(0), gt_bbox_down

    else:
        return img_dr.squeeze(0), gt_bbox
        