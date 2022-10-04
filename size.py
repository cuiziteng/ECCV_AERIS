import os

import cv2
import numpy as np

for file in os.listdir(r'/data/unagi0/cui_data/coco/val2017'):
    img = cv2.imread('/data/unagi0/cui_data/coco/val2017/'+file)
    print(img.shape)