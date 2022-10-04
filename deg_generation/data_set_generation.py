import os
import numpy as np
import scipy.stats as ss
from scipy import ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
import random
from joblib import Parallel, delayed
import cv2
from func import *
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="/data/unagi0/cui_data/coco/val2017")   # MS-COCO 2017 evaluation dataset image
parser.add_argument('--anno_path', type=str, default="/data/unagi0/cui_data/coco/annotations/instances_train2017.json")   # MS-COCO 2017 evaluation dataset label
parser.add_argument('--num_cores', default=6, type=int, help='Number of CPU Cores')
config = parser.parse_args()
print(config)

NUM_CORES = config.num_cores

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def LR_process(ratio, type='multi'):
    '''
    ratio: down-sampling ratio
    type: down-sampling type, choose from multi/nearest/bilinear/bicubic
    '''

    new_path = config.img_path + '_resoultion_' + str(type) + str(ratio)
    mkdir(new_path)

    for file in tqdm(os.listdir(config.img_path)):
        #print(file)
        file_name = os.path.join(config.img_path, file)
        #print(file_name)
        img = plt.imread(file_name)/255.0
        if type == 'multi':
            type = random.choice(['nearest', 'bilinear', 'bicubic'])
        
        img_down = low_resolution(img, type, ratio=ratio)
        #print(os.path.join(new_path, file))
        plt.imsave(os.path.join(new_path, file), img_down)


Parallel(n_jobs=NUM_CORES)(delayed(LR_process(ratio=4)))

# LR_process(ratio=4)
# LR_process(ratio=2)








