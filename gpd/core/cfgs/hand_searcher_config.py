from ..entity import HandGeometry


class HandSearcherConfig:
    """
    This class is the configuration for the hand search process
    """

    def __init__(self, hand_geometry: HandGeometry = None):
        # todo finish it
        self.radius = 0.001
        self.normal_radius = 0.001
        self.n_samples = 50
        self.range_rotation = 90
        self.n_rotations = 8
        self.n_finger_placements = 10
        self.friction_coeff = 20
        self.approach_step = 0.005
        self.hand_geometry: HandGeometry = hand_geometry
