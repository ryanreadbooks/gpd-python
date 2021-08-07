from ..entity import HandGeometry


class HandSearcherConfig:
    """
    This class is the configuration for the hand search process
    """

    def __init__(self):
        # todo finish it
        self.radius = 0.001
        self.normal_radius = 0.001
        self.n_samples = 30
        self.range_rotation = 90
        self.n_rotations = 8
        self.n_finger_placements = 10
        self.friction_coeff = 20
        self.hand_geometry: HandGeometry = HandGeometry()
