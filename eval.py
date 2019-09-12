from dataset import DIV2KDataset
from model import ESPCN

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time

current_time = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
print('Training start at', current_time)

root_path = 'data/DIV2K/DIV2K_HR'

folders = {'train': 'train', 'val': 'val'}
datasets = {split: DIV2KDataset(os.path.join(root_path, folder), 'x8')
            for split, folder in folders.items()}
datasizes = {split: len(datasets[split])
              for split, folder in folders.items()}

batch_size = 4

dataloaders = {split: data.DataLoader(dataset, batch_size=1, shuffle=split == 'train', num_workers=0)
               for split, dataset in datasets.items()}

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nepochs = 50
log_step = 100

# Define network
net = ESPCN(8)
net.to(dev)
print(net)

# Define loss
criterion = nn.MSELoss()

# Define optim
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

pretrained = torch.load('models/unet_bn_20190912_040318.pth', map_location='cuda:0')
net.load_state_dict(pretrained['model_state_dict'])
optimizer.load_state_dict(pretrained['optimizer_state_dict'])

net.eval()
with torch.no_grad():
    val_loss = 0.0
    start = time.time()
    for i, (inps, lbls) in enumerate(dataloaders['val']):
        inps = inps.to(dev)
        lbls = lbls.to(dev)

        outs = net(inps)

        loss = criterion(outs, lbls)
        val_loss += loss.item()

        for j, (inp, lbl, out) in enumerate(zip(inps, lbls, outs)):
            inp_ = transforms.ToPILImage()(inp.cpu()).resize((out.shape[2], out.shape[1]), Image.BICUBIC)
            
            y_ = transforms.ToTensor()(inp_)
            y_pred = y_ + out.cpu()
            y_pred = transforms.ToPILImage()(y_pred.clamp(0, 1))
            y_pred.save('output/{:03d}_pred.png'.format(i+j))
            inp_.save('output/{:03d}.png'.format(i + j))


print('Prediction takes {} (s)'.format((time.time() - start) / datasizes['val']))
print('Validation loss:', val_loss / datasizes['val'])
scheduler.step(val_loss)