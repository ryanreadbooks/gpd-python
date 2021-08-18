import os
import sys
Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)

import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt

from gpd.core import *
from gpd.core.visualization import DynamicVisualizer, StaticVisualizer, BaseVisualizer


if __name__ == '__main__':

    # cloud = o3d.io.read_point_cloud('plys/krylon.pcd')
    cloud = o3d.io.read_point_cloud('plys/sugar_box.ply')
    all_points = np.asarray(cloud.points) / 1.
    print(f'point shape = {all_points.shape}, max = {all_points.max()}, min = {all_points.min()}')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    vis = StaticVisualizer(True)
    vis.add_pointcloud(cloud, np.array([255, 20, 125]))
    # vis.add_pointcloud_by_array(all_points)
    hand_search_config = HandSearcherConfig('../cfg/grasp_generation.yaml')
    # sample point from pointcloud first
    sampler = PointSampler(hand_search_config.n_samples, method=PointSampler.Uniformly)
    sampled_cloud, _ = sampler.sample(cloud)
    sampled_points = np.asarray(sampled_cloud.points)
    vis.add_pointcloud_by_array(sampled_points, np.array([0, 125, 125]))

    # then calculate the local reference frame for all sampled points
    print('sampled_points shape = ', sampled_points.shape)
    frame_calculator = LocalFrameCalculator(radius=hand_search_config.radius, cloud=cloud, normals_radius=hand_search_config.normal_radius)
    frames = frame_calculator.calculate_local_frames(sampled_points)

    # for frame in frames:
    #     vis.add_local_frame(frame)

    hand_search = HandSearcher(hand_search_config)

    grasps = hand_search.generate_grasps(all_points, frames)
    print(f'Len of result grasps = {len(grasps)}')
    if len(grasps) == 0:
        print('No grasp candidates generated')
    else:
        grasp_image_generator = GraspImageGenerator(GraspImageConfig('../cfg/grasp_image.yaml'), hand_search_config.hand_geometry.hand_outer_diameter)

        # we take the first grasp to check
        grasp: Hand = grasps[0]
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        res = grasp_image_generator.generate_grasp_image(grasps[0], points, normals)
        print(res.shape)

        vis.add_hand(grasp, True)
        # vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx, invert=True), )
        # vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx), np.array([0, 255, 0]))
        # vis.show()
        # # try to visualize it
        # occupy_img = np.clip(res * 255, 0, 255).astype(int)
        norm_img = np.clip(res * 255, 0, 255).astype(int)
        # occupy_img1 = np.clip(occupy_pic1 * 255, 0, 255).astype(int)
        # norm_img1 = np.clip(norm_pic1 * 255, 0, 255).astype(int)
        # occupy_img2 = np.clip(occupy_pic2 * 255, 0, 255).astype(int)
        # norm_img2 = np.clip(norm_pic2 * 255, 0, 255).astype(int)
        # plt.figure(1)
        # plt.subplot(3, 2, 1)
        # plt.imshow(occupy_img[..., 0])
        # plt.subplot(3, 2, 2)
        # plt.imshow(norm_img)
        # plt.subplot(3, 2, 3)
        # plt.imshow(occupy_img1[..., 0])
        # plt.subplot(3, 2, 4)
        # plt.imshow(norm_img1)
        # plt.subplot(3, 2, 5)
        # plt.imshow(occupy_img2[..., 0])
        # plt.subplot(3, 2, 6)
        # plt.imshow(norm_img2)
        # plt.show()
