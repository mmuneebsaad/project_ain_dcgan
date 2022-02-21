# -*- coding: utf-8 -*-

from __future__ import print_function
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
import matplotlib.pyplot as plt
# print(os.listdir("../input"))
import sys, os, glob, time, imageio 
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import zipfile
# print(os.listdir("../input"))
import sys, os, glob, time, imageio 
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[768]
model = InceptionV3([block_idx])
model=model.cuda()

def calculate_activation_statistics(images,model, dims=768,
                    cuda=False):
    model.eval()
    
    act=np.empty((len(images), dims))
    # torch.from_numpy(act)
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value

# Batch size during training
batch_size = 20
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 128
ngpu=1

# Source files

# real images
real_images = 'real.zip'
# generated images
gen_images = 'gen.zip'

#generate directories for real and fake images
os.makedirs("FID_real/folder", exist_ok=True)
os.makedirs("FID_fake/folder", exist_ok=True)

tar_real = 'FID_real/folder'
tar_gen = 'FID_fake/folder'

# extract all images into generated folders
with zipfile.ZipFile(real_images, 'r') as zip_ref:
    zip_ref.extractall(tar_real)

with zipfile.ZipFile(gen_images, 'r') as zip_ref:
    zip_ref.extractall(tar_gen)
# # Root directory for dataset
path_real = "FID_real"
path_fake = "FID_fake"

# Number of workers for dataloader
workers = 0

data_real = dset.ImageFolder(root=path_real,
                            transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0), (1)),
                            ]))
# Create the dataloader for real images
dataloader_real = torch.utils.data.DataLoader(data_real, batch_size=batch_size,
                                          shuffle=True, drop_last=True, num_workers=workers)

data_gen = dset.ImageFolder(root=path_fake,
                            transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0), (1)),
                            ]))

# Create the dataloader for generated images
dataloader_gen = torch.utils.data.DataLoader(data_gen, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)
#Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Creating image batches for FID computation
images1 = []
images2 = []

def main():
    for i, data in enumerate(dataloader_real, 0):
        # Format batch
        img1 = data[0].to(device)
        images1.append(img1)
    real_images = torch.stack(images1)
    return images1, real_images 
if __name__ == '__main__':
    im1, real = main()

def main():
    for j, data in enumerate(dataloader_gen, 0):
        # Format batch
        img2 = data[0].to(device)
        images2.append(img2)
    gen_images = torch.stack(images2)
    return images2, gen_images 
if __name__ == '__main__':
    im2, gen = main()

# FID computation 
FID = []

for k in range(0, len(real)):
    fretchet_dist=calculate_fretchet(real[k], gen[k], model)
    FID.append(fretchet_dist)

# Saving FID all values to a csv file
np.savetxt("FID.csv", 
           FID,
           delimiter =", ", 
           fmt ='% s')