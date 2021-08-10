import copy
import time
from typing import List

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from ..utils import timer
from .local_frame_calculator import LocalFrameCalculator, LocalFrame, PointCloud, KDTree
from .cfgs import HandSearcherConfig
from .entity import *
from .visualization import DynamicVisualizer
from ..utils import timer


class HandSearcher:
    """
    The class represents a grasp candidate search process, mainly for searching the grasp candidates
    """

    ROTATION_AXIS_NORMAL = 0
    ROTATION_AXIS_BINORNAL = 1
    ROTATION_AXIS_CURVATURE_AXIS = 2

    def __init__(self, config: HandSearcherConfig):
        self.config = config

    @timer
    def generate_grasps(self, points: np.ndarray, frames: List[LocalFrame]) -> List[Hand]:
        """
        generate grasps based on the given points
        :param points: points, shape = (n, 3)
        :param frames: the list of local reference frame calculated from points sampled from 'points'
        :return:
        """

        ndy = self.config.n_finger_placements
        fw = self.config.hand_geometry.finger_width
        dy_steps = np.arange(-fw * ndy, (ndy + 1) * fw, fw)
        # in radians
        rot_start = -self.config.range_rotation / 180 * np.pi
        rot_end = -rot_start
        rotations_steps = np.linspace(rot_start, rot_end, self.config.n_rotations)
        # stores the potential grasps
        potential_grasps: List[Hand] = list()
        # stores the final grasps
        final_grasps: List[Hand] = list()

        # every frame should do the homogeneous transformation to be a new frame and then check for collision
        for frame in frames:
            # make rotation step
            for angle in rotations_steps:
                inv_xy_rot_mat = Rotation.from_rotvec(np.pi * frame.curvature_axis).as_matrix()
                rot_mat = Rotation.from_rotvec(angle * frame.curvature_axis).as_matrix()
                dy_potential_grasps: List[Hand] = list()
                # dy step is finger width
                # perform rotation
                frame_rotated: LocalFrame = frame.rotate(inv_xy_rot_mat).rotate(rot_mat)

                for dy in dy_steps:
                    # move the frame sample by the distance of 'dy' along binormal axis(y direction)
                    sample_translated = frame.sample + frame_rotated.binormal_axis * dy
                    # go back a 'bite' after rotation and translation
                    bottom_center: np.ndarray = -self.config.hand_geometry.init_bite * frame_rotated.normal_axis + sample_translated

                    transformed_frame = copy.deepcopy(frame_rotated)
                    transformed_frame.sample = bottom_center

                    # create a grasp candidate based on the transformed frame
                    grasp: Hand = Hand(frame.sample, transformed_frame, bottom_center, self.config.hand_geometry)

                    # now we can check collision
                    has_points_in_open_region, _ = grasp.check_square_collision(points, Hand.OpenRegion)
                    collided_with_bottom = True
                    collided_with_right = True
                    collided_with_left = True
                    if has_points_in_open_region:
                        collided_with_bottom, _ = grasp.check_square_collision(points, Hand.HandBottom)
                        # if points in region between fingers and do not collided with the bottom, we do further checking
                        if not collided_with_bottom:
                            collided_with_left, _ = grasp.check_square_collision(points, Hand.LeftFinger)
                            if not collided_with_left:
                                collided_with_right, _ = grasp.check_square_collision(points, Hand.RightFInger)
                                if not collided_with_right:
                                    # add the grasp to dy_potential_grasps pool which contains grasps that satisfy condition along y axis(binormal axis)
                                    dy_potential_grasps.append(grasp)
                    if not has_points_in_open_region and collided_with_bottom and collided_with_right and collided_with_left:
                        continue

                if len(dy_potential_grasps) != 0:
                    # we only take the middle grasps from dy direction
                    middle_grasp: Hand = dy_potential_grasps[len(dy_potential_grasps) // 2]
                    potential_grasps.append(middle_grasp)
                    # final_grasps.append(middle_grasp)
                    # then check if the hand is collided with the table by checking if this grasp is a grasp from down to top direction (greater than 30 degrees)
                    # the finger tip position
                    fingertip_pos = middle_grasp.bottom_center + middle_grasp.approach_axis * middle_grasp.hand_depth
                    # fingertip_pos[2] => the z coordinate of fingertip
                    # middle_grasp.bottom_center[2] => the z coordinate of bottom center
                    # -middle_grasp.hand_depth * 0.5 means we grasp objects with the angle larger than 30 degrees
                    # if grasp angle from down to top is greater than 30 degrees, than we accept this grasp as a potential?
                    # fixme this step check if hand collided with table, we need another method for doing it!!! this doesn't work
                    if fingertip_pos[2] < middle_grasp.bottom_center[2] - middle_grasp.hand_depth * np.sin(30. / 180. * np.pi):
                        potential_grasps.append(middle_grasp)

            # now we can make the grasps approach the object
            approach_max_dist = self.config.hand_geometry.hand_depth  # the objective distance to approach
            n_approaches = int(approach_max_dist // self.config.approach_step)
            for p_grasp in potential_grasps:
                collided_during_approaching = False
                for i in range(n_approaches):
                    # approach the hand along the approach axis by the distance of approach_step and check collision everytime
                    # approach_dist = i * self.config.approach_step
                    p_grasp.approach_object(dist=self.config.approach_step)
                    # check collision
                    is_collided = p_grasp.check_collided(points)
                    # if no collision happens, we continue to "push" the hand to the object
                    # vis.add_hand('approaching_hand', p_grasp)
                    if is_collided:
                        collided_during_approaching = True
                        # we “push” the hand forward along the negative x axis
                        # until one of the fingers or the hand base makes contact with the point cloud.
                        # this one has collision, we take two step back
                        p_grasp.approach_object(dist=-self.config.approach_step * 2)
                        # final check collision
                        pts_in_open_region, _ = p_grasp.check_square_collision(points, Hand.OpenRegion)
                        collided = p_grasp.check_collided(points)
                        if pts_in_open_region and not collided:
                            final_grasps.append(p_grasp)
                        break
                if not collided_during_approaching:
                    final_grasps.append(p_grasp)

        return final_grasps
