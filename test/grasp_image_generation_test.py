import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from gpd.core import *
from gpd.core.visualization import DynamicVisualizer, StaticVisualizer, BaseVisualizer


def cal_projection(point_cloud_voxel, surface_normal, order, gripper_width):
    voxel_point_num = 50
    m_width_of_pic = 60
    margin = 1
    occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
    norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
    norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

    max_x = point_cloud_voxel[:, order[0]].max()
    min_x = point_cloud_voxel[:, order[0]].min()
    max_y = point_cloud_voxel[:, order[1]].max()
    min_y = point_cloud_voxel[:, order[1]].min()
    min_z = point_cloud_voxel[:, order[2]].min()

    tmp = max((max_x - min_x), (max_y - min_y))
    if tmp == 0:
        print("WARNING : the num of input points seems only have one, no possible to do learning on"
              "such data, please throw it away.  -- Hong zhuo")
        return occupy_pic, norm_pic
    # Here, we use the gripper width to cal the res:
    # 每个最小单元的宽度? 一共有gripper_width的空间用来存放m_width_of_pic - margin个单元，每个单元里面是用来存放在夹爪坐标系下的点的？
    res = gripper_width / (m_width_of_pic - margin)

    voxel_points_square_norm = []
    # 下面这6句应该是将每个点对应所在的voxel编号算出来
    # res应该就是每个voxel的尺寸
    # 后面加上m_width_of_pic / 2 是因为在grasp的坐标系中？
    x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
    y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
    z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
    x_coord_r = np.floor(x_coord_r).astype(int)
    y_coord_r = np.floor(y_coord_r).astype(int)
    z_coord_r = np.floor(z_coord_r).astype(int)
    # voxel_index 每个点对应哪个voxel
    voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
    # 提出voxel_index中重复的voxel
    coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
    K = len(coordinate_buffer)
    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=K, dtype=np.int64)
    feature_buffer = np.zeros(shape=(K, voxel_point_num, 6), dtype=np.float32)
    index_buffer = {}
    # 给每个index编号？
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

    # 遍历
    for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
        # 找到在index_buffer中的index值
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        # 给每个voxel安排进点，最多安排voxel_point_num个点，这样每个voxel就包含点的xyz坐标和法向量
        if number < voxel_point_num:
            # voxel
            feature_buffer[index, number, :3] = point
            feature_buffer[index, number, 3:6] = normal
            # 有一个点，对应位置+1
            number_buffer[index] += 1
    # feature_buffer shape = (K, 50, 6)
    # number_buffer应该是每个voxel中点的数量
    # 取所有每个voxel中的所有点的法向量求和然后除以这个voxel中点的个数 法向量的norm?
    voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1) / number_buffer[:, np.newaxis]
    voxel_points_square = coordinate_buffer

    if len(voxel_points_square) == 0:
        return occupy_pic, norm_pic
    x_coord_square = voxel_points_square[:, 0]
    y_coord_square = voxel_points_square[:, 1]
    norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
    occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
    occupy_max = occupy_pic.max()
    assert (occupy_max > 0)
    occupy_pic = occupy_pic / occupy_max
    return occupy_pic, norm_pic


if __name__ == '__main__':

    cloud = o3d.io.read_point_cloud('plys/krylon.pcd')
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

    # for frame in frames:
    #     vis.add_local_frame(frame)

    hand_search = HandSearcher(config)

    grasps = hand_search.generate_grasps(all_points, frames)
    print(f'Len of result grasps = {len(grasps)}')
    if len(grasps) == 0:
        print('No grasp candidates generated')
    else:
        grasp_image_generator = GraspImageGenerator(GraspImageConfig(''), 0.12)

        # we take the first grasp to check
        grasp: Hand = grasps[0]
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        points_t = (grasp.rotation.T @ (points - grasp.bottom_center).T).T
        normals_t = (grasp.rotation @ normals.T).T
        res = grasp_image_generator.generate_grasp_image(grasps[0], points, normals)
        print(res.shape)
        # occupy_pic, norm_pic = cal_projection(points_t[grasp.contained_pts_idx], normals_t[grasp.contained_pts_idx], np.array([0, 1, 2]), 0.12)
        # occupy_pic1, norm_pic1 = cal_projection(points_t[grasp.contained_pts_idx], normals_t[grasp.contained_pts_idx], np.array([1, 2, 0]), 0.12)
        # occupy_pic2, norm_pic2 = cal_projection(points_t[grasp.contained_pts_idx], normals_t[grasp.contained_pts_idx], np.array([0, 2, 1]), 0.12)
        #
        vis.add_hand(grasp, True)
        # vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx, invert=True), )
        vis.add_pointcloud(cloud.select_by_index(grasp.contained_pts_idx), np.array([0, 255, 0]))
        vis.show()
        # # try to visualize it
        # occupy_img = np.clip(occupy_pic * 255, 0, 255).astype(int)
        # norm_img = np.clip(norm_pic * 255, 0, 255).astype(int)
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
