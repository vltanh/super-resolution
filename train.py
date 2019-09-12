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

pretrained = torch.load('models/unet_bn_20190912_030527.pth', map_location='cuda:0')
net.load_state_dict(pretrained['model_state_dict'])
optimizer.load_state_dict(pretrained['optimizer_state_dict'])

best_val_loss = 10000.0
for epoch in tqdm(range(nepochs)):
    print('Epoch {:>3d}'.format(epoch))
    running_loss = 0.0

    net.train()
    for i, (inps, lbls) in enumerate(dataloaders['train']):
        inps = inps.to(dev)
        lbls = lbls.to(dev)

        optimizer.zero_grad()

        outs = net(inps)

        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % log_step == 0:
            print('Iter {:>5d}, loss: {:.5f}'.format(
                i + 1, running_loss / log_step))
            running_loss = 0.0

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

            # if i > 1: continue
            # for j, (inp, lbl, out) in enumerate(zip(inps, lbls, outs)):
            #     y_ = transforms.ToPILImage()(inp.cpu()).resize((out.shape[2], out.shape[1]), Image.BICUBIC)
            #     y_ = transforms.ToTensor()(y_)
            #     y_pred = y_ + out.cpu()

            #     plt.subplot(len(inps), 3, j + 1)
            #     plt.imshow(y_.squeeze().clamp(min=0, max=1).cpu().numpy(), cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])

            #     plt.subplot(len(outs), 3, j + 1 + len(outs))
            #     plt.imshow(out.squeeze().clamp(min=0, max=1).cpu().numpy(), cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])

            #     plt.subplot(len(outs), 3, j + 1 + 2*len(outs))
            #     plt.imshow(y_pred.squeeze().clamp(min=0, max=1).cpu().numpy(), cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])

            #     plt.show()

    print('Prediction takes {} (s)'.format((time.time() - start) / datasizes['val']))
    print('Validation loss:', val_loss / datasizes['val'])
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'models/unet_bn_{}.pth'.format(current_time))
        best_val_loss = val_loss

# for i, (ys, ress) in enumerate(dataloaders['train']):
    