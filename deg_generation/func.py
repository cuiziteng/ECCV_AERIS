import numpy as np
import scipy.stats as ss
from scipy import ndimage
import matplotlib.pyplot as plt

import random
import cv2

# Blurry Condition
def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

def isotropic_Gaussian(ksize=15, l=6):
    """ generate an isotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        l    : [0.1,50], scaling of eigenvalues
    Returns:
        k     : kernel
    """

    V = np.array([[1, 0], [0, -1]])
    D = np.array([[l, 0], [0, l]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


# Add Gaussian Noise
def add_noise(img, sigma=25.0):

    noise = np.random.normal(0, sigma/255.0, size=(img.shape[0],img.shape[1], 1))

    return np.clip(img + noise, 0.0, 1.0)

# Image Low Resolution
def low_resolution(img, type='bicubic', ratio=2):
    w_down, h_down = int(img.shape[0]/ratio), int(img.shape[1]/ratio)

    if type == 'bicubic':
        img_down = cv2.resize(img, (h_down, w_down), interpolation=cv2.INTER_CUBIC)

    if type == 'bilinear':
        img_down = cv2.resize(img, (h_down, w_down), interpolation=cv2.INTER_LINEAR) 

    if type == 'nearest':
        img_down = cv2.resize(img, (h_down, w_down), interpolation=cv2.INTER_NEAREST) 

    return np.clip(img_down , 0.0, 1.0)

# Gaussian Blurry
def gaussian_blurry(img, type='iso'):
    
    kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))    # size
    kernel_width_iso = random.uniform(0.1, 2.4)
    angle = random.uniform(0, np.pi)
    kernel_width_un1 = random.uniform(0.5, 6)
    kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

    if type == 'none':
        return img

    elif type == 'iso':
        k1 = isotropic_Gaussian(17, 2.4)    
        img = ndimage.filters.convolve(img, np.expand_dims(k1, axis=2), mode='wrap')  # 'nearest' | 'mirror'
        return img
    
    elif type == 'aniso':
        k2 = anisotropic_Gaussian(17, np.pi/3, 5, 2.3)
        img = ndimage.filters.convolve(img, np.expand_dims(k2, axis=2), mode='wrap')  # 'nearest' | 'mirror'
        return img