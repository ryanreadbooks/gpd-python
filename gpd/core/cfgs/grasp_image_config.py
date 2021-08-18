import yaml
import numpy as np


class GraspImageConfig:
    def __init__(self, cfg_path: str):
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
            self.image_size = cfg['image_size']
            self.num_voxels = self.image_size
            self.total_num_channels = cfg['total_num_channels']
            self.max_voxel_points = cfg['max_voxel_points']
            self.margin = cfg['margin']
            self.need_both = cfg['need_both']
