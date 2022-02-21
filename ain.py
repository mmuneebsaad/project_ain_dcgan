# -*- coding: utf-8 -*-

# Importing Libraries...

import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import glob
import os

#.................

# Path for original images
orig_images = 'orig_images/*.jpg'

# contrast threshold
ct = 10

# window size
ws = (4,4)

# new image size
image_size = (128,128)

images = [cv2.imread(file) for file in glob.glob(orig_images)]

# Transforming all images to gray scale values  
for i in range(len(images)):
  if len(images[i].shape)>2:
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

# Resizing images 
resize_images = []

for i in images:
    resized = cv2.resize(i, dsize = image_size)
    resize_images.append(np.uint8(i))

# Adaptive input image normalization using histogram equalization

AIN = cv2.createCLAHE(clipLimit=ct, tileGridSize=ws)

processed_images = []

for i in resize_images:
    i = AIN.apply(i)
    processed_images.append(i)

#...........................

# Writing preprocessed images to a new folder

os.makedirs("preprocessed_images", exist_ok=False)

for i in range(0, len(processed_images)):
    save_images = cv2.imwrite('preprocessed_images/{}.jpg'.format(i), processed_images[i])