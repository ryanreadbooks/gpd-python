from typing import List

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as spy_rot

from ..utils import timer
from .local_frame_calculator import LocalFrameCalculator, LocalFrame, PointCloud, KDTree
from .cfgs import HandSearcherConfig
from .entity import *


class HandSearcher:
    """
    The class represents a grasp candidate search process, mainly for searching the grasp candidates
    """

    ROTATION_AXIS_NORMAL = 0
    ROTATION_AXIS_BINORNAL = 1
    ROTATION_AXIS_CURVATURE_AXIS = 2

    def __init__(self, config: HandSearcherConfig):
        self.config = config

    def generate_grasps(self, points):

        rot_start = self.config.range_rotation / 180 * np.pi
        rot_end = -rot_start
        rotations_step = np.linspace(rot_start, rot_end, self.config.n_rotations)
        potential_grasps = list()
        # make step
        for angle in rotations_step:
            rot_mat = 0



