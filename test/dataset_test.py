import os
import sys
Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)

import torch
import torch.utils.data as torch_data

from gpd import GraspImageDataset


if __name__ == '__main__':
    dataset = GraspImageDataset(root='images', train=True)
    dataloader = torch_data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=12)
    for i, data in enumerate(dataloader):
        repr, label = data
        print(repr.shape)
        print(label.shape)