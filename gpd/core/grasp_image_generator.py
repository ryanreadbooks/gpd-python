import numpy as np
import open3d as o3d

from .cfgs import GraspImageConfig
from .entity import Hand

Voxel = o3d.geometry.Voxel
VoxelGrid = o3d.geometry.VoxelGrid
PointCloud = o3d.geometry.PointCloud


class GraspImageGenerator:
    """
    generate the grasp image, only supports 3-channels and 12-channels grasp representation
    """

    ProjectionPlane_XY = np.array([0, 1, 2])
    ProjectionPlane_XZ = np.array([0, 2, 1])
    ProjectionPlane_YZ = np.array([1, 2, 0])

    def __init__(self, cfg: GraspImageConfig, gripper_width: float):
        self.cfg = cfg
        self.gripper_width = gripper_width

    def generate_grasp_repr(self, points: np.ndarray, normals: np.ndarray, order: np.ndarray):
        """
        generate grasp images representation. Inspired by https://github.com/lianghongzhuo/PointNetGPD/blob/master/PointNetGPD/model/dataset.py.
        :param points: the input points in hand frame that are inside the opening region of the hand gripper. shape = (n, 3)
        :param normals: normals of corresponding points, with respect to hand frame. shape = (n, 3)
        :param order: projection direction
        :return:
        """
        n_voxel_axis = self.cfg.image_size  # also it's the number of voxel along one axis
        occupied_pic = np.zeros([n_voxel_axis, n_voxel_axis, 1])
        normal_pic = np.zeros([n_voxel_axis, n_voxel_axis, 3])

        max_x = points[:, order[0]].max()
        min_x = points[:, order[0]].min()
        max_y = points[:, order[1]].max()
        min_y = points[:, order[1]].min()
        min_z = points[:, order[2]].min()

        if max((max_x - min_x), (max_y - min_y)) == 0:
            print('WARNING : the num of input points seems only have one')
            return occupied_pic, normal_pic

        # the size of each voxel
        res = self.gripper_width / (n_voxel_axis - self.cfg.margin)

        voxel_points_square_norm = []
        x_coord_r = ((points[:, order[0]]) / res + n_voxel_axis / 2)
        y_coord_r = ((points[:, order[1]]) / res + n_voxel_axis / 2)
        z_coord_r = ((points[:, order[2]]) / res + n_voxel_axis / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)

        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        # eliminate the repeating voxel indices
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.cfg.max_voxel_points, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, points, normals):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            # only max self.cfg.max_voxel_points points could be put into one voxel
            if number < self.cfg.max_voxel_points:
                # voxel
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1
        # feature_buffer shape = (K, 50, 6)
        voxel_points_square_normal = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupied_pic, normal_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        normal_pic[x_coord_square, y_coord_square, :] = voxel_points_square_normal
        occupied_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupied_max = occupied_pic.max()
        assert (occupied_max > 0)
        occupied_pic = occupied_pic / occupied_max
        return occupied_pic, normal_pic

    def generate_grasp_image(self, grasp: Hand, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Generate the corresponding grasp image for the given 'grasp'
        :param grasp: the given grasp
        :param points: the input points
        :param normals: corresponding normals
        :return: image representation
        """
        points_shape = points.shape
        assert points_shape[1] == 3, 'points must be shape = (n, 3)'
        normals_shape = normals.shape
        assert normals_shape[1] == 3, 'normals must be shape = (n, 3)'

        # transform the points and normals to hand frame
        points_t = (grasp.rotation.T @ (points - grasp.bottom_center).T).T
        normals_t = (grasp.rotation @ normals.T).T
        # we only take points that are inside opening region
        pts = points_t[grasp.contained_pts_idx]
        normal = normals_t[grasp.contained_pts_idx]
        if self.cfg.need_both:
            occupied_pic, normal_pic = self.generate_grasp_repr(pts, normals_t, self.ProjectionPlane_XY)
            occupied_pic_xy, normal_pic_xy = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_XY)
            occupied_pic_yz, normal_pic_yz = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_YZ)
            occupied_pic_xz, normal_pic_xz = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_XZ)
            output2 = np.dstack([occupied_pic_xy, normal_pic_xy, occupied_pic_yz, normal_pic_yz, occupied_pic_xz, normal_pic_xz])
            return normal_pic, output2
        if self.cfg.total_num_channels == 3:
            occupied_pic, normal_pic = self.generate_grasp_repr(pts, normals_t, self.ProjectionPlane_XY)
            output = normal_pic
        elif self.cfg.total_num_channels == 12:
            occupied_pic_xy, normal_pic_xy = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_XY)
            occupied_pic_yz, normal_pic_yz = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_YZ)
            occupied_pic_xz, normal_pic_xz = self.generate_grasp_repr(pts, normal, self.ProjectionPlane_XZ)
            output = np.dstack([occupied_pic_xy, normal_pic_xy, occupied_pic_yz, normal_pic_yz, occupied_pic_xz, normal_pic_xz])
        else:
            raise ValueError(f'Not supported total number of channels = {self.cfg.total_num_channels}')

        return output
