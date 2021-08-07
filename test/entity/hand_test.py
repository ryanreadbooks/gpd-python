import numpy as np
import open3d as o3d

from gpd.core import Hand
from gpd.core import LocalFrameCalculator, LocalFrame, PointSampler


def do_test():
    # read the point cloud from ply file
    cloud = o3d.io.read_point_cloud('../plys/can.ply')
    sampler = PointSampler(30, method=PointSampler.Uniformly)
    sampled_cloud, _ = sampler.sample(cloud)
    sampled_points = np.asarray(sampled_cloud.points)

    frame_calculator = LocalFrameCalculator(radius=0.01, cloud=cloud, normals_radius=0.01)
    frames = frame_calculator.calculate_local_frames(sampled_points)

    bottom_center = np.array([0., 0., 0.])
    test_hand = Hand(sampled_points[0], frames[0].as_matrix(), bottom_center,
                     0.01, 0.02, 0.12, 0.06)

    test_hand.check_square_collision(bottom_center, sampled_points, Hand.OpenRegion)


if __name__ == '__main__':
    do_test()
