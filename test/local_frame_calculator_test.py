import numpy as np
import open3d as o3d

from gpd.core import LocalFrameCalculator, LocalFrame, PointSampler
from gpd.utils import timer


@timer
def do_main():
    # read the point cloud from ply file
    cloud = o3d.io.read_point_cloud('plys/can.ply')

    # sample point from pointcloud first
    sampler = PointSampler(30, method=PointSampler.Uniformly)
    sampled_cloud, _ = sampler.sample(cloud)
    sampled_points = np.asarray(sampled_cloud.points)

    # then calculate the local reference frame for all sampled points
    print('sampled_points shape = ', sampled_points.shape)
    frame_calculator = LocalFrameCalculator(radius=0.01, cloud=cloud, normals_radius=0.01)
    frames = frame_calculator.calculate_local_frames(sampled_points)
    for frame in frames:
        print(frame)
        # frame.check_axis_orthogonal()
        print(np.linalg.norm(frame.normal_axis))
        print(np.linalg.norm(frame.binormal_axis))
        print(np.linalg.norm(frame.curvature_axis))
        print('-' * 20)


if __name__ == '__main__':
    do_main()
