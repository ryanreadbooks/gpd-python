import copy
import os, sys

# print(__file__)  # file relative path
# print(os.path.abspath(__file__))  # file absolute path
# print(os.path.dirname(os.path.abspath(__file__)))   # parent absolute path
import pickle

Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)

import numpy as np
import open3d as o3d

from gpd.core import LocalFrameCalculator, PointSampler, HandSearcher, HandSearcherConfig, HandGeometry, Hand, AntipodalGraspLabeling
from gpd.utils import timer
from gpd.core.visualization import StaticVisualizer, BaseVisualizer


@timer
def do_main():
    # read the point cloud from ply file
    # cloud = o3d.io.read_point_cloud('plys/glue.ply')
    # cloud = o3d.io.read_point_cloud('plys/apple.ply')
    cloud = o3d.io.read_point_cloud('plys/sugar_box.ply')
    # cloud = o3d.io.read_point_cloud('plys/mug.xyz')
    # cloud = o3d.io.read_point_cloud('plys/krylon.pcd')
    all_points = np.asarray(cloud.points) / 1.
    print(f'point shape = {all_points.shape}, max = {all_points.max()}, min = {all_points.min()}')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    vis = StaticVisualizer(True)
    vis.add_pointcloud(cloud, np.array([255, 20, 125]))

    config = HandSearcherConfig('../cfg/grasp_generation.yaml')
    n_positive_grasps = 0
    n_grasps_needed = 20
    max_search = 10
    cur_search = 0
    while n_positive_grasps < n_grasps_needed and cur_search < max_search:
        cur_search += 1
        # sample point from pointcloud first
        sampler = PointSampler(config.n_samples, method=PointSampler.Uniformly)
        sampled_cloud, _ = sampler.sample(cloud)
        sampled_points = np.asarray(sampled_cloud.points)
        # vis.add_pointcloud_by_array(sampled_points, np.array([0, 125, 125]))

        # then calculate the local reference frame for all sampled points
        print('sampled_points shape = ', sampled_points.shape)
        frame_calculator = LocalFrameCalculator(radius=config.radius, cloud=cloud, normals_radius=config.normal_radius)
        frames = frame_calculator.calculate_local_frames(sampled_points)

        hand_search = HandSearcher(config)

        grasps = hand_search.generate_grasps(all_points, frames)
        print(f'Debug: Len of result grasps = {len(grasps)}')

        use_force_closure = False
        if len(grasps) == 0:
            print('No grasp candidates generated')
            return
        else:
            # check the generated grasp is antipodal or not
            label_generator = AntipodalGraspLabeling(0.003, config.min_viable, config.friction_coeff)

            points = np.asarray(cloud.points)
            normals = np.asarray(cloud.normals)
            n = 0
            for grasp in grasps:
                if not use_force_closure:
                    points_t = (grasp.rotation.T @ (points - grasp.bottom_center).T).T
                    normals_t = (grasp.rotation @ normals.T).T
                    result = label_generator.label_grasp(grasp, points_t, normals_t)
                    if result == AntipodalGraspLabeling.FULL_GRASP:
                        n += 1
                        n_positive_grasps += 1
                        print(f'INFO: Found one positive grasp, now have {n_positive_grasps} positive grasps')
                        vis.add_hand(grasp, True)
                        # vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx), np.array([0, 255, 0]))
                        # print(grasp)
                else:
                    result = label_generator.label_grasp_force_closure(grasp, points, normals)
                    if result:
                        n += 1
                        n_positive_grasps += 1
                        vis.add_hand(grasp, True)

    if cur_search >= max_search:
        print('WARN: no grasp could be found')
        # if n > 0:
        #     vis.show()
    vis.show()


if __name__ == '__main__':
    do_main()
