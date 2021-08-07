import numpy as np


class HandGeometry:
    """
    This class stores parameters which define the geometry of the robot hand.
    This geometry is used to calculate the grasp candidates.
    """

    def __init__(self, finger_width: float, outer_diameter: float, hand_depth: float,
                 hand_height: float, init_bite: float):
        self.finger_width = finger_width
        self.outer_diameter = outer_diameter
        self.hand_depth = hand_depth
        self.hand_height = hand_height
        self.init_bite = init_bite

    def __str__(self):
        return f'finger_width = {self.finger_width}, ' \
               f'outer_diameter = {self.outer_diameter} ' \
               f'hand_depth = {self.hand_depth} ' \
               f'hand_height = {self.hand_height} ' \
               f'init_bite = {self.init_bite}'
