from dataset import DIV2KDataset

import torch
import torch.utils.data as data

import os
import matplotlib.pyplot as plt

root_path = 'data/DIV2K/DIV2K_HR'

folders = {'train': 'train', 'val': 'val'}
datasets = {split: DIV2KDataset(os.path.join(root_path, folder), 'x8')
            for split, folder in folders.items()}
datasizes = {split: len(datasets[split])
              for split, folder in folders.items()}

batch_size = 4

dataloaders = {split: data.DataLoader(dataset, batch_size=1, shuffle=split == 'train', num_workers=12)
               for split, dataset in datasets.items()}

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for i, (hrs, lrs) in enumerate(dataloaders['train']):
    for j, (hr, lr) in enumerate(zip(hrs, lrs)):
        plt.subplot(len(hrs), 2, j + 1)
        plt.imshow(hr.numpy().transpose(1, 2, 0))
        plt.title('High resolution #{:03d}'.format(j))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(lrs), 2, j + 1 + len(lrs))
        plt.imshow(lr.numpy().transpose(1, 2, 0))
        plt.title('Low resolution #{:03d}'.format(j))
        plt.xticks([])
        plt.yticks([])

        plt.show()