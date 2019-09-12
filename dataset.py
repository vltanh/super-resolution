import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

class DIV2KDataset(data.Dataset):
    def __init__(self, img_path, downscale_method, scale_factor=8):
        self.hr_path = [os.path.join(img_path, x)
                         for x in os.listdir(img_path)]
        self.downscale_method = downscale_method
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.hr_path)

    def get_lr_path(self, img_path):
        lr_path = img_path.replace('DIV2K_HR', 'DIV2K_LR-{}'.format(self.downscale_method))
        lr_path = lr_path.replace('.png', '{}.png'.format(self.downscale_method))
        return lr_path

    def __getitem__(self, i):
        # Load high resolution image
        hr = Image.open(self.hr_path[i])
        hr_size = hr.size
        # Convert to YCbCr and get the Y channel
        hr_ycbcr = hr.convert('YCbCr')
        hr_y, _, _ = hr_ycbcr.split()
        hr_y_tensor = transforms.ToTensor()(hr_y)

        # Load low resolution image
        lr_path = self.get_lr_path(self.hr_path[i])
        lr = Image.open(lr_path)
        # Convert to YCbCr and get the Y channel
        lr_ycbcr = lr.convert('YCbCr')
        lr_y, _, _ = lr_ycbcr.split()
        # Upscaling low resolution image
        w = self.scale_factor * lr_y.size[0]
        h = self.scale_factor * lr_y.size[1]
        lr_y_bicubic = transforms.Resize((h, w), interpolation=Image.BICUBIC)(lr_y)
        lr_y_bicubic_tensor = transforms.ToTensor()(lr_y_bicubic)
        hr_y_tensor = transforms.ToTensor()(transforms.Resize((h, w), interpolation=Image.BICUBIC)(hr_y))
        
        # Get residual image
        residual = hr_y_tensor - lr_y_bicubic_tensor
        lr_y_tensor = transforms.ToTensor()(lr_y)

        # Get original image tensor
        lr_ycbcr_tensor = transforms.ToTensor()(lr_ycbcr)
        hr_ycbcr_tensor = transforms.ToTensor()(hr_ycbcr)

        return lr_y_tensor, residual