from typing import List

import numpy as np
import open3d as o3d

from gpd.core import LocalFrameCalculator, PointSampler, HandSearcher, HandSearcherConfig, HandGeometry, Hand
from gpd.utils import timer


@timer
def do_main():
    # read the point cloud from ply file
    # cloud = o3d.io.read_point_cloud('plys/can.ply')
    cloud = o3d.io.read_point_cloud('/home/ryan/Codes/paper_codes/gpd/tutorials/krylon.pcd')

    # sample point from pointcloud first
    sampler = PointSampler(30, method=PointSampler.Uniformly)
    sampled_cloud, _ = sampler.sample(cloud)
    sampled_points = np.asarray(sampled_cloud.points)

    # then calculate the local reference frame for all sampled points
    print('sampled_points shape = ', sampled_points.shape)
    frame_calculator = LocalFrameCalculator(radius=0.01, cloud=cloud, normals_radius=0.01)
    frames = frame_calculator.calculate_local_frames(sampled_points)

    hand_search = HandSearcher(HandSearcherConfig(HandGeometry(0.01, 0.02, 0.12, 0.06, 0.01)))
    grasps: List[Hand] = hand_search.generate_grasps(sampled_points, frames)
    print(f'Len of result grasps = {len(grasps)}')
    for grasp in grasps:
        print(grasp)


if __name__ == '__main__':
    do_main()
