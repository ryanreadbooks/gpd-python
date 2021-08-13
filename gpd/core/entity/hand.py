from typing import List
import numpy as np
import open3d as o3d

from .local_frame import LocalFrame
from .hand_geometry import HandGeometry

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

    OpenRegion = 0x01
    LeftFinger = 0x02
    RightFInger = 0x04
    HandBottom = 0x08

    def __init__(self, sample: np.ndarray, frame: LocalFrame, bottom_center: np.ndarray, geometry: HandGeometry):
        """
        init a robot hand
        :param sample: the point neighborhood associated with the grasp
        :param frame: the orientation of the grasp as a rotation matrix associated with the sample point
        :param bottom_center: the center coordinates of hand bottom(base). location of the hand, np.ndarray, shape = (3,)
        :param geometry: the hand geometry
        """
        shape = frame.as_matrix().shape
        assert shape == (3, 3), "frame as rotation matrix for the hand, shape must be (3,3), but got {}".format(shape)
        c_shape = bottom_center.shape
        assert c_shape == (3,) or c_shape == (3, 1) or c_shape == (1, 3), 'bottom center represents 3d coordinate, ' \
                                                                          'only 3-element vector are allowed'
        self.sample = sample
        self.finger_width = geometry.finger_width
        self.hand_height = geometry.hand_height
        self.hand_outer_diameter = geometry.hand_outer_diameter
        self.hand_depth = geometry.hand_depth
        self.rotation = frame.as_matrix()
        self.frame = frame
        self.approach_axis = self.rotation[:, 0]
        self.binormal_axis = self.rotation[:, 1]
        self.curvature_axis = self.rotation[:, 2]
        # we need to determine the coordinates of each point of the hand based on the hand geometry
        # bottom center is just like translation/position of the hand
        self.bottom_center = bottom_center
        # hand_points_local is for collision check
        self.hand_points_local = self.cal_hand_points_loc(np.array([0, 0, 0]), None, None)
        # hand_points_vis is for visualization, the coordinate is respect to world reference frame
        self.hand_points_vis = self.cal_hand_points_loc(bottom_center, self.approach_axis, self.binormal_axis)
        # store the indices of points of the object when the grasp surrounding an object
        self.contained_pts_idx = None

    def check_square_collision(self, points: np.ndarray, way) -> (bool, np.ndarray):
        """
        check whether there are collisions between given 'points' and the specified hand part
        :param points: the graspable object points, shape = (n, 3) or (3, n)
        :param way: which part of the robot hand should be considered for collision check,
                    OpenRegion, LeftFinger, RightFInger, HandBottom are allowed
        :return: true - no collision; false - has collision; and the index of inside points are also returned
        """
        p_shape = points.shape
        assert p_shape[1] == 3 or p_shape[0] == 3, f'shape of points = (n, 3) or (3, n) are allowed, but got {p_shape}'
        if p_shape[0] == 3:
            points = points.reshape(-1, 3)  # make shape = (n, 3)
        # is this necessary? => yes, this is necessary
        # n_points = points.shape[0]
        # transform the point to the hand frame
        points_g = (self.rotation.T @ (points - self.bottom_center).T).T

        # define the cube space below, hand local coordinates
        if way == self.OpenRegion:
            # check if points are in the region between fingers, no collision
            s1, s2, s4, s8 = self.hand_points_local[1], self.hand_points_local[2], self.hand_points_local[4], self.hand_points_local[8]
        elif way == self.RightFInger:
            # check if points are collided with left finger
            s1, s2, s4, s8 = self.hand_points_local[9], self.hand_points_local[1], self.hand_points_local[10], self.hand_points_local[12]
        elif way == self.LeftFinger:
            # check if points are collided with right finger
            s1, s2, s4, s8 = self.hand_points_local[2], self.hand_points_local[13], self.hand_points_local[3], self.hand_points_local[7]
        elif way == self.HandBottom:
            # check if points are collided with the hand bottom
            s1, s2, s4, s8 = self.hand_points_local[11], self.hand_points_local[15], self.hand_points_local[12], self.hand_points_local[20]
        else:
            raise ValueError('parameter way can only be OpenRegion, LeftFinger, RightFInger, HandBottom')

        # check if points (x, y, z) in the defined cube
        ymin, ymax = s1[1], s2[1]
        a1 = ymin < points_g[:, 1]  # (n, )
        a2 = ymax > points_g[:, 1]  # (n, )

        zmin, zmax = s4[2], s1[2]
        a3 = zmin < points_g[:, 2]  # (n, )
        a4 = zmax > points_g[:, 2]  # (n, )

        xmin, xmax = s8[0], s4[0]
        a5 = xmin < points_g[:, 0]  # (n, )
        a6 = xmax > points_g[:, 0]  # (n, )

        # if all True, means all points are inside cube
        a = np.vstack([a1, a2, a3, a4, a5, a6])  # (6, n)
        has_points_in_cube = False
        points_in_cube_idx = np.where(np.sum(a, axis=0) == len(a))[0]  # len(a) = 6
        if len(points_in_cube_idx) != 0:
            has_points_in_cube = True
            if way == self.OpenRegion:
                self.contained_pts_idx = points_in_cube_idx

        return has_points_in_cube, points_in_cube_idx

    def check_collided(self, points: np.ndarray) -> bool:
        """
        check whether the given points collided with any part of the robot hand
        :param points: the graspable object points, shape = (n, 3) or (3, n)
        :return: true -> collided; false -> not collided
        """
        if self.check_square_collision(points, self.HandBottom)[0]:
            return True
        if self.check_square_collision(points, self.LeftFinger)[0]:
            return True
        if self.check_square_collision(points, self.RightFInger)[0]:
            return True
        return False

    def update_hand_points_position(self, bottom_center: np.ndarray):
        """
        update the position of the points on hand based the new 'bottom_center', which means updating self.hand_points
        :param bottom_center: the new bottom center of the hand, which means a new position is given
        :return:
        """
        self.bottom_center = bottom_center
        self.hand_points_vis = self.cal_hand_points_loc(bottom_center, self.approach_axis, self.binormal_axis)

    def cal_hand_points_loc(self, bottom_center: np.ndarray,
                            approach_axis=None, binormal_axis=None) -> np.ndarray:
        """
        calculate the coordinates of 20 points on the robot hand
        :param bottom_center: the center coordinate of the hand bottom
        :param approach_axis: the approach axis of the hand frame
        :param binormal_axis: the binormal axis of the hand frame
        :return: the points on hand, shape = (21, 3)
        """

        if approach_axis is None or binormal_axis is None:
            curvature_axis = np.array([0., 0., 1.])
            binormal_axis = np.array([0., 1., 0.])
            approach_axis = np.array([1., 0., 0.])
        else:
            curvature_axis = self.curvature_axis
            binormal_axis = self.binormal_axis
            approach_axis = self.approach_axis

        hh = self.hand_height
        fw = self.finger_width
        hod = self.hand_outer_diameter
        hd = self.hand_depth
        open_width = hod - 2 * fw
        p5_p6_mid = curvature_axis * hh * 0.5 + bottom_center
        p7_p8_mid = -curvature_axis * hh * 0.5 + bottom_center

        p5 = -binormal_axis * open_width * 0.5 + p5_p6_mid
        p6 = binormal_axis * open_width * 0.5 + p5_p6_mid

        p7 = binormal_axis * open_width * 0.5 + p7_p8_mid
        p8 = -binormal_axis * open_width * 0.5 + p7_p8_mid

        p1 = approach_axis * hd + p5
        p2 = approach_axis * hd + p6
        p3 = approach_axis * hd + p7
        p4 = approach_axis * hd + p8

        p9 = -binormal_axis * fw + p1
        p10 = -binormal_axis * fw + p4
        p11 = -binormal_axis * fw + p5
        p12 = -binormal_axis * fw + p8
        p13 = binormal_axis * fw + p2
        p14 = binormal_axis * fw + p3
        p15 = binormal_axis * fw + p6
        p16 = binormal_axis * fw + p7

        p17 = -approach_axis * hh + p11
        p18 = -approach_axis * hh + p15
        p19 = -approach_axis * hh + p16
        p20 = -approach_axis * hh + p12

        return np.vstack([bottom_center, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])

    def approach_object(self, dist: float):
        """
        approach the hand to the object along the approach axis by the distance of 'dist'
        :param dist: the distance to go
        :return:
        """
        # update the bottom center, make it approach the object
        self.bottom_center = self.approach_axis * dist + self.bottom_center
        self.frame.sample = self.bottom_center
        # update points on hand at the same time
        self.hand_points_vis = self.cal_hand_points_loc(self.bottom_center, self.approach_axis, self.binormal_axis)

    def find_contacts(self, points, normals):
        # transform the point to the hand frame
        points_g = (self.rotation.T @ (points[self.contained_pts_idx] - self.bottom_center).T).T
        normals_g = (self.rotation @ normals.T).T
        right_idx = np.argmin(points_g[:, 1], axis=0)
        left_idx = np.argmax(points_g[:, 1], axis=0)
        right_point = points[right_idx]
        left_point = points[left_idx]
        right_normal = normals[right_idx]
        left_normal = normals[left_idx]

        return left_point, right_point, left_normal, right_normal, left_idx, right_idx

    def check_force_closure(self, points, normals, friction_coef) -> bool:
        """

        :param points: shape = (n, 3)
        :param normals: shape = (n, 3)
        :param friction_coef: the friction coefficient
        :return:
        """
        p1, p2, n1, n2, _, _ = self.find_contacts(points, normals)

        n1, n2 = -n1, -n2  # inward facing normals

        if (p1 == p2).all():  # same point
            return False

        for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
            diff = other_contact - contact
            normal_proj = normal.dot(diff) / np.linalg.norm(normal)

            if normal_proj < 0:
                return False  # wrong side
            alpha = np.arccos(normal_proj / np.linalg.norm(diff))
            if alpha > np.arctan(friction_coef):
                return False  # outside of friction cone
        return True

    def __str__(self):
        return f'{self.approach_axis}, {self.binormal_axis}, {self.curvature_axis}, {self.bottom_center}'
