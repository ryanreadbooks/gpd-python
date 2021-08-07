from typing import List

import numpy as np
import open3d as o3d


class PointSampler:
    """
    Sample points from the given point cloud
    """
    Uniformly = 0
    Normally = 1

    def __init__(self, n: int, method):
        """

        :param n: number of points to sample
        :param method: how to sample points from pointcloud
        """
        self.n = n
        self.method = method
        assert method == PointSampler.Uniformly or method == PointSampler.Normally, \
            "method must be PointSampler.Uniformly of PointSampler.Normally"

    def sample(self, cloud: o3d.geometry.PointCloud) -> (o3d.geometry.PointCloud, List):
        if self.method == self.Normally:
            return self._sample_normally(cloud)
        else:
            return self._sample_uniformly(cloud)

    def _sample_uniformly(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        num_points = np.asarray(cloud.points).shape[0]
        assert self.n <= num_points, "can not sample {} points in cloud with {} total points".format(self.n, num_points)
        rand_indices: np.ndarray = np.random.choice(num_points, self.n, replace=False).tolist()

        return cloud.select_by_index(rand_indices), rand_indices

    def _sample_normally(self, cloud: o3d.geometry.PointCloud):
        print(self.n)
        raise NotImplementedError
