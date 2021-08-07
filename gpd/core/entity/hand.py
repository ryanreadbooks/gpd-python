from typing import List
import numpy as np
import open3d as o3d

from .finger import Finger
from .local_frame import LocalFrame

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

    def __init__(self, sample: np.ndarray, frame: np.ndarray, bottom_center: np.ndarray, finger_width: float, hand_height: float,
                 hand_outer_diameter: float, hand_depth: float):
        """
        init a robot hand
        :param sample: the point neighborhood associated with the grasp
        :param frame: the orientation of the grasp as a rotation matrix associated with the sample point
        :param bottom_center: the center coordinates of hand bottom(base)
        :param finger_width: the width of the finger, assuming that two fingers are identical
        :param hand_height: the height of the robot hand
        :param hand_outer_diameter: the total width of the whole robot hand
        :param hand_depth: the finger length
        """
        assert frame.shape == (3, 3), "frame as rotation matrix for the hand, shape must be (3,3), but got {}".format(frame.shape)
        self.sample = sample
        self.finger_width = finger_width
        self.hand_height = hand_height
        self.hand_outer_diameter = hand_outer_diameter
        self.hand_depth = hand_depth
        self.rotation = frame
        self.position = None
        self.approach_axis = frame[:, 0]
        self.binormal_axis = frame[:, 1]
        self.curvature_axis = frame[:, 2]
        # we need to determine the coordinates of each point of the hand based on the hand geometry
        self.hand_points = self._cal_hand_points_loc(bottom_center)

    def check_square_collision(self, bottom_center: np.ndarray, points: np.ndarray, way) -> (bool, np.ndarray):
        """
        check whether there are collisions between given 'points' and the specified hand part
        :param bottom_center: location of the hand. np.ndarray, shape = (3,)
        :param points: the graspable object points, shape = (n, 3) or (3, n)
        :param way: which part of the robot hand should be considered for collision check,
                    OpenRegion, LeftFinger, RightFInger, HandBottom are allowed
        :return: true - no collision; false - has collision; and the index of inside points are also returned
        """
        c_shape = bottom_center.shape
        p_shape = points.shape
        assert c_shape == (3,) or c_shape == (3, 1) or c_shape == (1, 3), 'bottom center represents 3d coordinate, ' \
                                                                          'only 3-element vector are allowed'
        assert p_shape[1] == 3 or p_shape[0] == 3, f'shape of points = (n, 3) or (3, n) are allowed, but got {p_shape}'
        if p_shape[0] == 3:
            points = points.reshape(-1, 3)
        # n_points = points.shape[0]
        points = points - bottom_center.reshape(1, 3)
        # rotate the point to the hand frame
        points_g = (self.rotation @ points.T).T  # (n, 3)

        # define the cube space below
        if way == self.OpenRegion:
            # check if points are in the region between fingers, no collision
            s1, s2, s4, s8 = self.hand_points[1], self.hand_points[2], self.hand_points[4], self.hand_points[8]
        elif way == self.LeftFinger:
            # check if points are collided with left finger
            s1, s2, s4, s8 = self.hand_points[9], self.hand_points[1], self.hand_points[10], self.hand_points[12]
        elif way == self.RightFInger:
            # check if points are collided with right finger
            s1, s2, s4, s8 = self.hand_points[2], self.hand_points[13], self.hand_points[3], self.hand_points[7]
        elif way == self.HandBottom:
            # check if points are collided with the hand bottom
            s1, s2, s4, s8 = self.hand_points[11], self.hand_points[15], self.hand_points[12], self.hand_points[20]
        else:
            raise ValueError('parameter way can only be OpenRegion, LeftFinger, RightFInger, HandBottom')

        # check if points (x, y, z) in the defined cube
        ymin, ymax = s1[1], s2[1]
        a1 = ymin < points_g[:, 1]  # (n, )
        a2 = ymax > points_g[:, 1]  # (n, )

        zmin, zmax = s1[2], s4[2]
        a3 = zmin > points_g[:, 2]  # (n, )
        a4 = zmax < points_g[:, 2]  # (n, )

        xmin, xmax = s4[0], s8[0]
        a5 = xmin > points_g[:, 0]  # (n, )
        a6 = xmax < points_g[:, 0]  # (n, )

        # if all True, means all points are inside cube
        a = np.vstack([a1, a2, a3, a4, a5, a6])  # (6, n)
        points_in_cube_idx = np.where(np.sum(a, axis=0) == len(a))[0]  # len(a) = 6
        has_points_in_cube = False
        if len(points_in_cube_idx) != 0:
            has_points_in_cube = True

        return has_points_in_cube, points_in_cube_idx

    def check_collided(self, bottom_center: np.ndarray, points: np.ndarray) -> bool:
        """
        check whether the given points collided with any part of the robot hand
        :param bottom_center: location of the hand. np.ndarray, shape = (3,)
        :param points: the graspable object points, shape = (n, 3) or (3, n)
        :return: true -> collided; false -> not collided
        """
        if self.check_square_collision(bottom_center, points, self.HandBottom)[0]:
            return True
        if self.check_square_collision(bottom_center, points, self.LeftFinger)[0]:
            return True
        if self.check_square_collision(bottom_center, points, self.RightFInger)[0]:
            return True
        return False

    def _cal_hand_points_loc(self, bottom_center: np.ndarray) -> np.ndarray:
        """
        calculate the coordinates of 20 points on the robot hand
        :param bottom_center: the center coordinate of the hand bottom
        :return:
        """
        hh = self.hand_height
        fw = self.finger_width
        hod = self.hand_outer_diameter
        hd = self.hand_depth
        open_width = hod - 2 * fw
        p5_p6_mid = self.curvature_axis * hh * 0.5 + bottom_center
        p7_p8_mid = -self.curvature_axis * hh * 0.5 + bottom_center

        p5 = -self.binormal_axis * open_width * 0.5 + p5_p6_mid
        p6 = -p5

        p7 = self.binormal_axis * open_width * 0.5 + p7_p8_mid
        p8 = -p7

        p1 = self.approach_axis * hd + p5
        p2 = self.approach_axis * hd + p6
        p3 = self.approach_axis * hd + p7
        p4 = self.approach_axis * hd + p8

        p9 = -self.binormal_axis * fw + p1
        p10 = -self.binormal_axis * fw + p4
        p11 = -self.binormal_axis * fw + p5
        p12 = -self.binormal_axis * fw + p8
        p13 = self.binormal_axis * fw + p2
        p14 = self.binormal_axis * fw + p3
        p15 = self.binormal_axis * fw + p6
        p16 = self.binormal_axis * fw + p7

        p17 = -self.approach_axis * hh + p11
        p18 = -self.approach_axis * hh + p15
        p19 = -self.approach_axis * hh + p16
        p20 = -self.approach_axis * hh + p12

        return np.vstack([bottom_center, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
