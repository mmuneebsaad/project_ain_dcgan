# -*- coding: utf-8 -*-
"""MS-SSIM_xray.ipynb
# Importing Libraries...

import cv2
import glob
import numpy as np
from sewar.full_ref import msssim
# from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import matplotlib.animation as animation
from IPython.display import HTML

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image
import zipfile
import matplotlib.pyplot as plt
import numpy as np
# print(os.listdir("../input"))
import sys, os, glob, time, imageio 
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img 
#%%

real_images= 'real.zip'

with zipfile.ZipFile(real_images, "r") as zip_data:
    content_list = zip_data.namelist()
    images1 = []
    for name_file in content_list:
        img_bytes = zip_data.open(name_file)          # 1
        img_data = Image.open(img_bytes)              # 2
        image_as_array = np.array(img_data, np.uint8) # 3
        res = cv2.resize(image_as_array, dsize=(128, 128))
        images1.append(res)

myscore = []

for i in range (0, len(images1), 2):
    if i+1 < len(images1):
        f1, f2 = images1[i], images1[i+1]
    else:
        f1, f2 = images1[i], None
    # ms-ssim score
    m = msssim(f1, f2)
    myscore.append(m)
#%%
myscore = np.real(myscore)
final_value = np.mean(myscore)
print(final_value)
final_score = np.savetxt("real.csv", myscore, delimiter = ",")