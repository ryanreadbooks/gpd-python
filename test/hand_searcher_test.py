import os, sys

# print(__file__)  # file relative path
# print(os.path.abspath(__file__))  # file absolute path
# print(os.path.dirname(os.path.abspath(__file__)))   # parent absolute path
Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)

import numpy as np
import open3d as o3d

from gpd.core import LocalFrameCalculator, PointSampler, HandSearcher, HandSearcherConfig, HandGeometry, Hand
from gpd.utils import timer
from gpd.core.visualization import DynamicVisualizer, StaticVisualizer, BaseVisualizer


@timer
def do_main():
    # np.random.seed(125)
    # read the point cloud from ply file
    # cloud = o3d.io.read_point_cloud('plys/glue.ply')
    cloud = o3d.io.read_point_cloud('plys/sugar_box.ply')
    # cloud = o3d.io.read_point_cloud('plys/merged_cloud.ply')
    cloud = cloud.voxel_down_sample(voxel_size=0.005)
    # cloud = o3d.io.read_point_cloud('plys/apple.ply')
    # cloud = o3d.io.read_point_cloud('plys/krylon.pcd')
    all_points = np.asarray(cloud.points) / 1.
    print(f'point shape = {all_points.shape}, max = {all_points.max()}, min = {all_points.min()}')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    vis = StaticVisualizer(True)
    vis.add_pointcloud(cloud, np.array([255, 20, 125]))
    # vis.add_pointcloud_by_array(all_points)
    config = HandSearcherConfig('../cfg/grasp_generation.yaml')
    # sample point from pointcloud first
    sampler = PointSampler(config.n_samples, method=PointSampler.Uniformly)
    sampled_cloud, _ = sampler.sample(cloud)
    sampled_points = np.asarray(sampled_cloud.points)
    vis.add_pointcloud_by_array(sampled_points, np.array([0, 125, 125]))

    # then calculate the local reference frame for all sampled points
    print('sampled_points shape = ', sampled_points.shape)
    frame_calculator = LocalFrameCalculator(radius=config.radius, cloud=cloud, normals_radius=config.normal_radius)
    frames = frame_calculator.calculate_local_frames(sampled_points)

    for frame in frames:
        vis.add_local_frame(frame)

    hand_search = HandSearcher(config)

    grasps = hand_search.generate_grasps(all_points, frames)
    print(f'Len of result grasps = {len(grasps)}')
    if len(grasps) == 0:
        print('No grasp candidates generated')
        return

    # only show a few
    # for grasp in np.random.choice(grasps, int(len(grasps) * 1), False):
    # for grasp in grasps:
    #     # vis.add_hand(grasp, BaseVisualizer.HandStyle_Line)
    #     vis.add_hand(grasp, BaseVisualizer.HandStyle_Cube)
    #     vis.add_local_frame(grasp.frame, BaseVisualizer.LocalFrameColorScheme_Transform)

    for i, grasp in enumerate(np.random.choice(grasps, int(len(grasps) * 0.5), False)):
        vis.add_hand(grasp, False)
        # vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx, invert=True), )
        vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx),
                           np.array([max(int(251 // (i + 1)), 0), min(20 * i, 255), min(255, i * 30 - 25)]))

    vis.show()
    # vis.render_to_img()


if __name__ == '__main__':
    do_main()
