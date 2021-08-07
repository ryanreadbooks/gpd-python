from typing import List
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .finger import Finger
from ..entity import HandGeometry, LocalFrame

ROT_AXES = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
PointCloud = o3d.geometry.PointCloud


class BoundingBox:
    """
    The 2D bounding box of hand closing region with respect to hand frame
    """

    def __init__(self, center: float, top: float, bottom: float):
        self.center = center
        self.top = top
        self.bottom = bottom


class Hand:
    """
    The representation of the robot hand(grasp), containing the basic information of a grasp,
    such as position and rotation.
    This class represents a grasp candidate by the position and orientation of
    the robot hand at the grasp before the fingers are closed.
    """

    def __init__(self, point: np.ndarray, frame: np.ndarray, width: float,
                 l_finger: Finger, r_finger: Finger):
        """
        init a hand
        :param point: the center of the point neighborhood associated with the grasp
        :param frame: the orientation of the grasp as a rotation matrix
        :param width: the opening width of the robot hand, the distance between two fingers (inner side) when fully open
        :param l_finger: the left finger of the hand
        :param r_finger: the right finger of the hand
        """
        assert frame.shape == (3, 3), "frame as rotation matrix for the hand, shape must be (3,3), but got {}".format(frame.shape)
        self.point = point
        self.grasp_width = width
        self.left_finger = l_finger
        self.right_finger = r_finger
        self.rotation = frame
        self.position = None
        self.valid = False

    def get_approach(self):
        return self.rotation[:, 0]

    def get_binormal(self):
        return self.rotation[:, 1]

    def get_axis(self):
        return self.rotation[:, 2]


# todo implement this class
class HandSet:
    """
    This class is responsible for holding a set of hands
    """

    def __init__(self, h_geometry: HandGeometry, angles: np.ndarray, rot_axes, ndh: bool, n_placement:int):
        """

        :param h_geometry:
        :param angles:
        :param rot_axes:
        :param ndh:
        :param n_placement:
        """
        self.hands: List[Hand] = None
        self.hand_geometry = h_geometry
        self.angles = angles
        self.rot_axes: List = rot_axes
        self.need_deepen_hand = ndh
        self.n_placements = n_placement
        self.sample = None
        self.frame = None

    def self_assess(self, cloud_neighbor: PointCloud, local_frame: LocalFrame):
        """
        perform self assessment for hand search
        :param cloud_neighbor: point cloud from neighborhood
        :param local_frame: local reference frame
        :return:
        """
        self.sample = local_frame.sample
        # concatenate frame axes to form frame matrix
        self.frame = np.array([local_frame.normal_axis, local_frame.binormal_axis, local_frame.curvature_axis]).T
        # we do the real rotation for every rotation axis
        r: R = R.as_rotvec(np.pi * np.array([0., 1., 0.]))
        ROT_BINORMAL = r.as_matrix()
        for axis in self.rot_axes:
            for angle in self.angles:
                rot_mat = R.as_rotvec(angle * ROT_AXES[axis]).as_matrix()
                # the hand rotation
                frame_rot = self.frame @ ROT_BINORMAL @ rot_mat  # (3, 3)
                # rotate points into the hand orientation
                # transform(rotation only) the cloud to the hand reference frame
                homo_transform = np.zeros((4, 4))
                homo_transform[:3, :3] = frame_rot.T

                cloud_neighbor = PointCloud(
                    o3d.utility.Vector3dVector(
                        np.asarray(cloud_neighbor.points) - self.sample
                    )
                )
                cloud_neighbor: PointCloud = cloud_neighbor.transform(homo_transform)

                # crop the cloud out by hand height
                hand_height = self.hand_geometry.hand_height
                cloud_neighbor_points = np.asarray(cloud_neighbor.points)
                indices: np.ndarray = np.where(np.logical_and(cloud_neighbor_points[:, 2] > -hand_height, cloud_neighbor_points[:, 2] < hand_height))[0]
                # this is the points that are occluded by the base of the robot hand
                cloud_in_region: PointCloud = cloud_neighbor.select_by_index(o3d.utility.IntVector(indices.tolist()))

                # evaluate finger placements for this orientation.
                finger = Finger(self.hand_geometry.finger_width, self.hand_geometry.outer_diameter,
                                self.hand_geometry.hand_depth, self.n_placements)
