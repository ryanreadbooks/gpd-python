import yaml

from ..entity import HandGeometry


class HandSearcherConfig:
    """
    This class is the configuration for the hand search process
    """

    def __init__(self, cfg_path: str):
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
            grasp_generation_cfg = cfg['grasp_candidate_generation']
            self.radius = grasp_generation_cfg['neighbor_radius']
            self.normal_radius = grasp_generation_cfg['normals_radius']
            self.n_samples = grasp_generation_cfg['num_samples']
            self.n_rotations = grasp_generation_cfg['num_rotation']
            self.range_rotation = grasp_generation_cfg['range_rotation']
            self.n_dy = grasp_generation_cfg['num_dy']
            self.approach_step = grasp_generation_cfg['approach_step']
            self.friction_coeff = grasp_generation_cfg['friction_cone_angle']
            self.min_viable =  grasp_generation_cfg['min_viable']
            hand_geo_cfg = cfg['hand_geometry']
            self.hand_geometry = HandGeometry(finger_width=hand_geo_cfg['finger_width'],
                                              hand_outer_diameter=hand_geo_cfg['hand_outer_diameter'],
                                              hand_depth=hand_geo_cfg['hand_depth'],
                                              hand_height=hand_geo_cfg['hand_height'],
                                              init_bite=hand_geo_cfg['init_bite'])
