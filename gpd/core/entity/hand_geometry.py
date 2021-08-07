import numpy as np


class HandGeometry:
    """
    This class stores parameters which define the geometry of the robot hand.
    This geometry is used to calculate the grasp candidates.
    """

    def __init__(self, finger_width: float, hand_outer_diameter: float, hand_depth: float, hand_height: float, init_bite: float):
        """
        init
        :param finger_width: the width of the finger, assuming that two fingers are identical
        :param hand_height: the height of the robot hand
        :param hand_outer_diameter: the total width of the whole robot hand
        :param hand_depth: the finger length
        :param init_bite: a backup distance for grasping
        """
        self.finger_width = finger_width
        self.hand_outer_diameter = hand_outer_diameter
        self.hand_depth = hand_depth
        self.hand_height = hand_height
        self.init_bite = init_bite

    def __str__(self):
        return f'finger_width = {self.finger_width}, ' \
               f'outer_diameter = {self.hand_outer_diameter} ' \
               f'hand_depth = {self.hand_depth} ' \
               f'hand_height = {self.hand_height} '
