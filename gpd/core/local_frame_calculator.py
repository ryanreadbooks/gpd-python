from typing import List

import numpy as np
import open3d as o3d

from .entity import LocalFrame

PointCloud = o3d.geometry.PointCloud
KDTree = o3d.geometry.KDTreeFlann


class LocalFrameCalculator:
    """
    Local frame calculator
    """

    def __init__(self, radius: float, cloud: PointCloud, normals_radius: float, tree: KDTree = None):
        """

        :param radius:
        :param cloud: the input pointcloud
        :param normals_radius: if the input cloud does not contain normals, calculate normals with this radius
        :param tree: the kdtree corresponding to the input point cloud
        """
        self.radius = radius
        self.cloud = cloud
        if not cloud.has_normals():
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=30))
        if tree is None:
            self.tree = o3d.geometry.KDTreeFlann(cloud)
        else:
            self.tree = tree
        cloud.orient_normals_consistent_tangent_plane(30)
        # o3d.visualization.draw_geometries([cloud.paint_uniform_color(np.array([255, 0, 0]))], point_show_normal=True)

    def calculate_local_frames(self, samples: np.ndarray) -> List[LocalFrame]:
        """
        calculate the local reference frame for the samples
        :param samples: the sampled points whose local reference frame needs calculating, shape=(n, 3)
        :return:
        """
        assert samples.ndim == 2, "dimension of samples must be 2, with shape = (n, 3)"
        local_frames: List[LocalFrame] = []
        for sample in samples:
            local_frames.append(self.calculate_local_frame(sample))

        return local_frames

    def calculate_local_frame(self, sampled: np.ndarray) -> LocalFrame:
        """
        calculate the local reference frame for the given sampled point in the cloud
        :param sampled: the sampled point whose local reference frame needs calculating
        :return:
        """
        assert sampled.ndim != 3, "dimension of sampled not correct, it should be of shape (3, ) or (3, 1) or (1, 3)"
        sampled = sampled.reshape(3, 1)

        # k to be the number of neighbors; idx is the index of them in the original cloud
        [k, idx, _] = self.tree.search_radius_vector_3d(sampled, self.radius)
        if k == 0:
            raise RuntimeError("can not find neighbors for the input cloud with search radius = {}".format(self.radius))

        # extract the normals of the neighbors
        # shape=(num_neighbors, 3)
        neighbors_normals: np.ndarray = np.asarray(self.cloud.normals)[idx]
        local_frame = LocalFrame(sampled)
        local_frame.calculate_axis(neighbors_normals)

        return local_frame
