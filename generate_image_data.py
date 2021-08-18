from numpy.core.fromnumeric import shape
from gpd.core import *
from gpd.utils import *

from typing import List, Tuple
import os
import pathlib
import open3d as o3d
import h5py
import multiprocessing as mp
import glob
import random
import numpy as np


PointCloud = o3d.geometry.PointCloud

@timer
def generate_model_image_woker(paths: Tuple[str, str]):
    model_path, save_root = paths
    pid = os.getpid()
    print(pid, model_path, save_root)
    # we need at least 30 positive grasps per model object
    positive_grasps: List[Hand] = list()
    negative_grasps: List[Hand] = list()

    # load the model 
    cloud: PointCloud = o3d.io.read_point_cloud(model_path)
    maxbound = cloud.get_max_bound()
    max_v = np.max(maxbound)
    # downsample the cloud
    cloud.voxel_down_sample(voxel_size=max_v/100.)
    all_points = np.asarray(cloud.points)
    print(f'{pid} after downsampling, point size = {all_points.shape}')
    # search for positive and negative hand
    config = HandSearcherConfig('cfg/grasp_generation.yaml')
    n_positive_grasps = 0
    n_positive_grasps_needed = 20

    grasp_image_generator = GraspImageGenerator(GraspImageConfig('cfg/grasp_image.yaml'), 
                                    config.hand_geometry.hand_outer_diameter)

    while n_positive_grasps < n_positive_grasps_needed:
        # sample point from pointcloud first
        sampler = PointSampler(config.n_samples, method=PointSampler.Uniformly)
        sampled_cloud, _ = sampler.sample(cloud)
        sampled_points = np.asarray(sampled_cloud.points)

        # then calculate the local reference frame for all sampled points
        frame_calculator = LocalFrameCalculator(radius=config.radius, cloud=cloud, normals_radius=config.normal_radius)
        frames = frame_calculator.calculate_local_frames(sampled_points)

        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)

        hand_search = HandSearcher(config)

        grasps = hand_search.generate_grasps(all_points, frames)
        print(f'Debug: {pid} Len of result grasps = {len(grasps)}')

        if len(grasps) == 0:
            print(f'WARNING: {pid}No grasp candidates generated')
            continue
        else:
            # check the generated grasp is antipodal or not
            label_generator = AntipodalGraspLabeling(0.003, config.min_viable, config.friction_coeff)

            for grasp in grasps:
                points_t = (grasp.rotation.T @ (points - grasp.bottom_center).T).T
                normals_t = (grasp.rotation @ normals.T).T
                result = label_generator.label_grasp(grasp, points_t, normals_t)
                if result == AntipodalGraspLabeling.FULL_GRASP:
                    n_positive_grasps += 1
                    positive_grasps.append(grasp)
                    print(f'INFO: {pid} Found one positive grasp, now have {n_positive_grasps} positive grasps')
                else:
                    negative_grasps.append(grasp)
            print(f'n_positive_grasps = {n_positive_grasps} for now')

    # after generating positive and negative grasps, we generate the corresponding images
    lp = len(positive_grasps)
    ln = len(negative_grasps)
    print(f'{pid} positives = {lp}, negatives = {ln}')
    # balance negative and positive
    if lp > n_positive_grasps_needed:
        positive_grasps = random.sample(positive_grasps, n_positive_grasps_needed)
    if lp < ln:
        # sample negatives for balance
        new_ln = min(ln, lp * 2)
        negative_grasps = random.sample(negative_grasps, new_ln)
        print(f'{pid} after balancing, negatives = {new_ln}')

    model_path: pathlib.Path = pathlib.Path(model_path)
    h5_save_path = pathlib.Path(save_root) / model_path.name
    h5file_name = h5_save_path.with_suffix('.h5')
    h5file = h5py.File(str(h5file_name), 'w')
    print(f'{pid} Image generated, now saving them to h5 file {str(h5file_name)}...')
    train_group = h5file.create_group(name='train')
    test_group = h5file.create_group(name='test')
    train_normal_group = train_group.create_group(name='normal')
    train_image_group = train_group.create_group(name='image')
    test_normal_group = test_group.create_group(name='normal')
    test_image_group = test_group.create_group(name='image')

    random.shuffle(positive_grasps)
    random.shuffle(negative_grasps)
    r_train = 0.8
    positive_grasps_train = positive_grasps[:int(len(positive_grasps) * r_train)]
    positive_grasps_test = positive_grasps[int(len(positive_grasps) * r_train):]
    negative_grasps_train = negative_grasps[:int(len(negative_grasps) * r_train)]
    negative_grasps_test = negative_grasps[int(len(negative_grasps) * r_train):]

    num = 0
    # positive train data
    for i, positive_grasp in enumerate(positive_grasps_train):
        norm_pic, all_view_pic = grasp_image_generator.generate_grasp_image(positive_grasp, points, normals)
        # save it to hdf5 file
        normal_dataset = train_normal_group.create_dataset(name=f'{num}-normal', data=norm_pic)
        normal_dataset.attrs['label'] = 1
        normal_dataset.attrs['channel'] = 3
        all_view_dataset = train_image_group.create_dataset(name=f'{num}-image', data=all_view_pic)
        all_view_dataset.attrs['label'] = 1
        all_view_dataset.attrs['channel'] = 12
        num += 1
    # negative train data
    for i, positive_grasp in enumerate(negative_grasps_train):
        norm_pic, all_view_pic = grasp_image_generator.generate_grasp_image(positive_grasp, points, normals)
        # save it to hdf5 file
        normal_dataset = train_normal_group.create_dataset(name=f'{num}-normal', data=norm_pic)
        normal_dataset.attrs['label'] = 0
        normal_dataset.attrs['channel'] = 3
        all_view_dataset = train_image_group.create_dataset(name=f'{num}-image', data=all_view_pic)
        all_view_dataset.attrs['label'] = 0
        all_view_dataset.attrs['channel'] = 12
        num += 1
    # positive test data
    for i, negative_grasp in enumerate(positive_grasps_test):
        norm_pic, all_view_pic = grasp_image_generator.generate_grasp_image(negative_grasp, points, normals)
        normal_dataset = test_normal_group.create_dataset(name=f'{num}-normal', data=norm_pic)
        normal_dataset.attrs['label'] = 1
        normal_dataset.attrs['channel'] = 3
        all_view_dataset = test_image_group.create_dataset(name=f'{num}-image', data=all_view_pic)
        all_view_dataset.attrs['label'] = 1
        all_view_dataset.attrs['channel'] = 12
        num += 1
    # negative test data
    for i, positive_grasp in enumerate(negative_grasps_test):
        norm_pic, all_view_pic = grasp_image_generator.generate_grasp_image(positive_grasp, points, normals)
        # save it to hdf5 file
        normal_dataset = test_normal_group.create_dataset(name=f'{num}-normal', data=norm_pic)
        normal_dataset.attrs['label'] = 0
        normal_dataset.attrs['channel'] = 3
        all_view_dataset = test_image_group.create_dataset(name=f'{num}-image', data=all_view_pic)
        all_view_dataset.attrs['label'] = 0
        all_view_dataset.attrs['channel'] = 12
        num += 1

    h5file.close()
    print(f'{pid} ', str(h5file_name), ' done')


if __name__ == '__main__':
    model_paths = glob.glob('test/plys/*.*')
    # model_paths = ['test/plys/krylon.pcd', 'test/plys/mug.xyz']
    
    save_paths = ['test/images'] * len(model_paths)
    with mp.Pool(processes=12) as pool:
        pool.map(generate_model_image_woker, tuple(zip(model_paths, save_paths)))
    
    print('All done')
