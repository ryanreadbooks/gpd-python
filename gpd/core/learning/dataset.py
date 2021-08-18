import os
import pathlib
from typing import Dict, List
import numpy as np
import glob
import h5py
import torch
from torch._C import dtype
import torch.utils.data as torch_data


class GraspImageDataset(torch_data.Dataset):
    def __init__(self, root : str, train : bool = True, channel : int= 3) -> None:
        super(GraspImageDataset).__init__()
        assert channel in [3, 12], f'channel must be 3 or 12, but got {channel}, not supported'
        self.root_dir = root
        self.train = train
        self.channel = channel
        all_h5filenames = glob.glob(str(pathlib.Path(root) / '*.h5'))
        if len(all_h5filenames) == 0:
            raise ValueError('empty root dir')
        self.train_type = 'train' if self.train else 'test'
        self.channel_type = 'normal' if self.channel == 3 else 'image'
        self.total_amount = 0
        self.obj: List = []
        self.h5_files: List[h5py.File] = []
        for i, h5file in enumerate(all_h5filenames):
            f = h5py.File(h5file, 'r')
            self.h5_files.append(f)
            group: h5py.Group = f[self.train_type][self.channel_type]
            self.total_amount += len(group)
            self.obj.append(len(group))

        self.n_objs = len(self.h5_files)

    def __getitem__(self, index: int):
        # find the corresponding index of the h5 file
        h5_file_idx, real_idx = self._find_real_index(index)
        group: h5py.Group = self.h5_files[h5_file_idx][self.train_type][self.channel_type]
        dataset_name = f'{real_idx}-{self.channel_type}'
        h5_dataset: h5py.Dataset = group[dataset_name] 
        label = h5_dataset.attrs['label']
        img_data = np.asarray(h5_dataset)  # shape = (60, 60, 3 or 12)
        img_data = img_data.transpose([2, 0, 1])

        return torch.from_numpy(img_data).to(torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return self.total_amount
    
    # def __del__(self):
    #     for file in self.h5_files:
    #         file.close()

    def _find_real_index(self, index):
        cached_index = index
        h5file_idx = 0
        real_idx = 0
        last_idx = 0
        for i, n in enumerate(self.obj):
            cached_index = cached_index - n
            if cached_index < 0:
                h5file_idx = i
                real_idx = last_idx
                break
            last_idx = cached_index
        
        return h5file_idx, real_idx