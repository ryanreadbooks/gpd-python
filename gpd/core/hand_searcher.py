from typing import List

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

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

    def generate_grasps(self, points: np.ndarray, frames: List[LocalFrame]) -> List[Hand]:
        """
        generate grasps based on the given points
        :param points: points, shape = (n, 3)
        :param frames: the list of local reference frame calculated from points sampled from 'points'
        :return:
        """

        ndy = self.config.n_finger_placements
        fw = self.config.hand_geometry.finger_width
        dy_steps = np.linspace(-fw * ndy, (ndy + 1) * fw, ndy)

        # in radians
        rot_start = -self.config.range_rotation / 180 * np.pi
        rot_end = -rot_start
        rotations_steps = np.linspace(rot_start, rot_end, self.config.n_rotations)
        print(f'[HandSearcher.generate_grasps] total rotations is {self.config.n_rotations}')
        # stores the potential grasps
        potential_grasps: List[Hand] = list()
        # stores the final grasps
        final_grasps: List[Hand] = list()

        # every frame should do the homogeneous transformation to be a new frame and then check for collision
        for frame in frames:
            # make rotation step
            for angle in rotations_steps:
                rot_mat = Rotation.from_rotvec(angle * frame.curvature_axis).as_matrix()

                dy_potential_grasps: List[Hand] = list()
                # dy step is finger width
                for dy in dy_steps:
                    # perform rotation and translation
                    frame_rotated: LocalFrame = frame.rotate(rot_mat)
                    # move the frame sample by the distance of 'dy' along binormal axis(y direction)
                    sample_translated = frame.sample + frame_rotated.binormal_axis * dy
                    # go back a 'bite' after rotation and translation
                    bottom_center: np.ndarray = -self.config.hand_geometry.init_bite * frame_rotated.normal_axis + sample_translated

                    # now we can check collision
                    grasp = Hand(sample_translated, frame_rotated, bottom_center, self.config.hand_geometry)
                    in_open_region, _ = grasp.check_square_collision(points, Hand.OpenRegion)
                    collided_with_bottom, _ = grasp.check_square_collision(points, Hand.HandBottom)
                    # if points in region between fingers and do not collided with the bottom, we do further checking
                    if in_open_region and not collided_with_bottom:
                        collided_with_left = grasp.check_square_collision(points, Hand.LeftFinger)
                        collided_with_right = grasp.check_square_collision(points, Hand.RightFInger)
                        if not collided_with_left and not collided_with_right:
                            # add the grasp to dy_potential_grasps pool which contains grasps that satisfy condition along y axis(binormal axis)
                            dy_potential_grasps.append(grasp)

                if len(dy_potential_grasps) != 0:
                    # we only take the middle grasps from dy direction
                    middle_grasp: Hand = dy_potential_grasps[len(dy_potential_grasps) // 2 - 1]
                    # then check if the hand is collided with the table by checking if this grasp is a grasp from down to top direction (greater than 30 degrees)
                    # the finger tip position
                    fingertip_pos = middle_grasp.bottom_center + middle_grasp.approach_axis * middle_grasp.hand_depth
                    # fingertip_pos[2] => the z coordinate of fingertip
                    # middle_grasp.bottom_center[2] => the z coordinate of bottom center
                    # -middle_grasp.hand_depth * 0.5 means we grasp objects with the angle larger than 30 degrees
                    # if grasp angle from down to top is greater than 30 degrees, than we accept this grasp as a potential?
                    if fingertip_pos[2] < middle_grasp.bottom_center[2] - middle_grasp.hand_depth * np.sin(30. / 180. * np.pi):
                        potential_grasps.append(middle_grasp)

                # now we can make the grasp approach the object
                approach_max_dist = self.config.hand_geometry.hand_depth  # the objective distance to approach
                n_approaches = approach_max_dist // self.config.approach_step

                # for every potential grasps
                for p_grasp in potential_grasps:
                    for i in range(n_approaches):
                        # approach the hand along the approach axis by the distance of approach_step and check collision everytime
                        approach_dist = i * self.config.approach_step
                        p_grasp.approach_object(approach_dist)
                        # check collision
                        is_collided = p_grasp.check_collided(points)
                        if not is_collided:
                            final_grasps.append(p_grasp)

            # print(f'[HandSearcher.generate_grasps] done evaluating one frame')

        return final_grasps
