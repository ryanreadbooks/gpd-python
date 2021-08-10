import yaml
import numpy as np


class GraspImageConfig:
    def __init__(self, cfg_path: str):
        self.image_size = 60
        self.num_voxels = self.image_size
        self.total_num_channels = 12
        self.max_voxel_points = 50
        self.margin = 1
