import os
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

class DIV2KDataset(data.Dataset):
    def __init__(self, img_path, downscale_method):
        self.hr_path = [os.path.join(img_path, x)
                         for x in os.listdir(img_path)]
        self.downscale_method = downscale_method

    def __len__(self):
        return len(self.hr_path)

    def get_lr_path(self, img_path):
        lr_path = img_path.replace('DIV2K_HR', 'DIV2K_LR-{}'.format(self.downscale_method))
        lr_path = lr_path.replace('.png', '{}.png'.format(self.downscale_method))
        return lr_path

    def __getitem__(self, i):
        hr = Image.open(self.hr_path[i]).convert('RGB')

        lr_path = self.get_lr_path(self.hr_path[i])
        lr = Image.open(lr_path).convert('RGB')

        hr, lr = self.transform(hr, lr)

        return hr, lr

    def transform(self, hr, lr):
        hr = transforms.ToTensor()(hr)
        lr = transforms.ToTensor()(lr)
        return hr, lr
