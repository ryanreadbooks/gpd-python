from typing import List

import numpy as np
import open3d as o3d

from .entity import Hand
from .cfgs import *
from ..utils import timer


class AntipodalGraspLabeling:
    """
    helper class to label the grasp to be antipodal grasp or not
    """

    NO_GRASP = 0
    HALF_GRASP = 1
    FULL_GRASP = 2

    def __init__(self, extremal_thresh: float, n_viable: float, friction_coeff):
        self.extremal_thresh = extremal_thresh
        self.n_viable = n_viable
        self.friction_coeff = friction_coeff  # angle(degree) of the friction cone

    def label_grasp(self, grasp: Hand, points: np.ndarray, normals: np.ndarray) -> bool:
        """

        :param grasp:
        :param points: points with respect to hand coordinate frame, shape = (n, 3)
        :param normals: normals with respect to hand coordinate frame, shape = (n, 3)
        :return:
        """
        result = self.NO_GRASP

        pts_in_region = points[grasp.contained_pts_idx]
        normals_region = normals[grasp.contained_pts_idx]

        LEFT_LATERAL_DIRECTION = np.array([0, 1, 0])
        RIGHT_LATERAL_DIRECTION = -LEFT_LATERAL_DIRECTION

        cos_friction_coeff: float = np.cos(self.friction_coeff * np.pi / 180.0)

        max_y = np.max(pts_in_region[:, 1]) - self.extremal_thresh
        min_y = np.min(pts_in_region[:, 1]) + self.extremal_thresh

        # shape = (n,)
        left_normals = LEFT_LATERAL_DIRECTION @ normals_region.T
        right_normals = RIGHT_LATERAL_DIRECTION @ normals_region.T
        left_pre_valid: np.ndarray = left_normals > cos_friction_coeff
        right_pre_valid: np.ndarray = right_normals > cos_friction_coeff
        # shape = (n,)
        left_hit_extremal = pts_in_region[:, 1] > max_y
        right_hit_extremal = pts_in_region[:, 1] < min_y

        left_viable_indices = np.logical_and(left_pre_valid, left_hit_extremal)
        right_viable_indices = np.logical_and(right_pre_valid, right_hit_extremal)

        # store the indices of the viable points
        has_left_viable_pts = np.any(left_viable_indices)
        has_right_viable_pts = np.any(right_viable_indices)
        if has_left_viable_pts or has_right_viable_pts:
            result = self.HALF_GRASP

        if has_left_viable_pts and has_right_viable_pts:
            left_viable_pts = pts_in_region[np.where(left_viable_indices)[0]]
            right_viable_pts = pts_in_region[np.where(right_viable_indices)[0]]

            # further check viable
            # find max x and min x in all viable points
            front_viable_x: float = min(np.max(left_viable_pts[:, 0]), np.max(right_viable_pts[:, 0]))
            bottom_viable_x: float = max(np.min(left_viable_pts[:, 0]), np.min(right_viable_pts[:, 0]))
            # fine max z and min z in all viable points
            top_viable_z: float = min(np.max(left_viable_pts[:, 2]), np.max(right_viable_pts[:, 2]))
            down_viable_z: float = max(np.min(left_viable_pts[:, 2]), np.min(right_viable_pts[:, 2]))

            left_viables = np.where((left_viable_pts[:, 0] >= bottom_viable_x) &
                                    (left_viable_pts[:, 0] <= front_viable_x) &
                                    (left_viable_pts[:, 2] >= down_viable_z) &
                                    (left_viable_pts[:, 2] <= top_viable_z))[0]
            right_viable = np.where((right_viable_pts[:, 0] >= bottom_viable_x) &
                                    (right_viable_pts[:, 0] <= front_viable_x) &
                                    (right_viable_pts[:, 2] >= down_viable_z) &
                                    (right_viable_pts[:, 2] <= top_viable_z))[0]

            if len(left_viables) >= self.n_viable and len(right_viable) >= self.n_viable:
                result = self.FULL_GRASP

        return result
