import numpy as np
import scipy
import scipy.stats as ss
import scipy.signal as sg
import matplotlib.pyplot as plt
from kornia.filters import gaussian_blur2d, motion_blur, filter2D
import torch
import torch.nn.functional as F
import random
import cv2


def imshow(x, title=None, cbar=False, figsize=None, filename='czt_utils/kernel'):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.savefig(filename + '.jpg')


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


def isotropic_Gaussian(ksize=17, l=2.0):
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
    # print('0000', v)
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    # print('1111', V)
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    # H = H
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components

# if __name__ == "__main__":
#     #design kernel
#     kernel_size = int(random.choice([7, 9, 11, 13, 15, 17, 19, 21]))    # size
#     kernel_width_iso = random.uniform(0.1, 2.4)
#     angle = random.uniform(0, np.pi)
#     kernel_width_un1 = random.uniform(0.5, 6)
#     kernel_width_un2 = random.uniform(0.5, kernel_width_un1)

#     k1 = isotropic_Gaussian(kernel_size, kernel_width_iso)
#     print(k1.shape)
#     k2 = anisotropic_Gaussian(kernel_size, angle, kernel_width_un1, kernel_width_un2)
#     print(k2.shape)

# k_new_numpy = k_new.cpu().numpy()
# print(k_new.shape)
# print(k_new_numpy)
# imshow(k_new_numpy)
# k_new = k_new.view(1,-1)


# # load image
# img_path = r'/home/czt/mmdetection/czt_utils/009963.jpg'
# img = cv2.imread(img_path)/255.0
# img_ts = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
# img_out_ts = filter2D(img_ts,k_tensor)
# img_out = img_out_ts.squeeze(0).permute(1,2,0).cpu().numpy()
# cv2.imwrite(img_path.replace('.jpg', 'out.jpg'), img_out*255.0)
