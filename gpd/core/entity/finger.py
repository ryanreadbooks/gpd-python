import numpy as np

FORWARD_DIRECTION = 0
LATERAL_DIRECTION = 1


class Finger:
    """
    The representation of one finger of the robot hand
    """

    def __init__(self, f_width: float, hand_out_diameter: float, hand_depth: float, n_placements: int):
        """
        init a finger of hand
        :param f_width: width of finger
        :param hand_out_diameter: outer diameter of robot hand
        :param hand_depth: the depth of robot hand = length of the finger
        :param n_placements: number of placements for finger
        """
        self.finger_width = f_width
        self.hand_depth = hand_depth
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.center = None
        self.surface = None
        self.finger_spacing = None

    def find_possible_placements(self, points, bite):
        """
        find possible finger placements.
        finger placements need to be collision-free and contain at least one point in between the fingers
        :param points: points to check for the possible finger placements
        :param bite: how far the robot can be moved into the object ?
        :return:
        """
        # calculate top and bottom of the hand (top = finger tip, bottom = base)
        pass

