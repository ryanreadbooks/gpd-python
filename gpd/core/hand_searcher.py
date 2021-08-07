from typing import List

import numpy as np
import open3d as o3d

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
        self.handsets: List[HandSet] = []

    @timer
    def search_hands(self, samples: np.ndarray, cloud: PointCloud, tree: KDTree = None):
        """
        search the robot hands
        :param samples: the sample points for generating hand grasps. np.ndarray shape = (n, 3)
        :param cloud: the input point cloud for the search process
        :param tree: the kdtree corresponding to the input point cloud
        :return: list of grasp candidates
        """
        assert samples.shape[1] == 3, f'the shape of samples must be (n, 3), but got {samples.shape}'
        if tree is None:
            tree = o3d.geometry.KDTreeFlann(cloud)

        # 1. estimate local reference frames
        print('[HandSearcher.search_hands] Estimating local reference frames ...')
        frame_calculator = LocalFrameCalculator(radius=self.config.radius, cloud=cloud,
                                                normals_radius=self.config.normal_radius, tree=tree)
        frames: List[LocalFrame] = frame_calculator.calculate_local_frames(samples)

        # 2. evaluate possible hand placements
        print('[HandSearcher.search_hands] Finding hand poses ...')
        self._search_hand(cloud, frames, tree)

    def _search_hand(self, cloud: PointCloud, frames: List[LocalFrame], tree: KDTree):
        """
        search the robot hand given a list of local reference frame on the object point cloud
        :param cloud: the input point cloud
        :param frames: the given local reference frames
        :param tree: the kdtree corresponding to the input point cloud
        :return:
        """
        if tree is None:
            raise RuntimeError('the kdtree must not be None')
        # (n_orientations, )
        angles: np.ndarray = np.linspace(-np.pi / 2, np.pi / 2, self.config.n_orientations + 1)[:-1]
        # we iterate every local reference frame
        # every handset is associated with an LRF
        for i, frame in enumerate(frames):
            hand_set = HandSet(self.config.hand_geometry, angles, self.config.rot_axes, self.config.need_deepen_hand)
            self.handsets.append(hand_set)
            # todo we can optimize this, because there has already been a search radius vector in calculating LRF
            [k, idx, _] = tree.search_radius_vector_3d(frame.sample, self.config.radius)
            if k > 0:
                # we only need the neighborhood points
                # and search the hand in the neighborhood
                hand_set.self_assess(cloud.select_by_index(idx), frame)
