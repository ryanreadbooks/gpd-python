from typing import List

import numpy as np
import open3d as o3d

from ..entity import LocalFrame, Hand
from .base_visualizer import BaseVisualizer


class DynamicVisualizer(BaseVisualizer):

    def __init__(self):
        super(DynamicVisualizer, self).__init__()
        self.visualizer = o3d.visualization.Visualizer()
        self.geometries = {}

    def create_window(self):
        self.visualizer.create_window()

    def destroy_window(self):
        self.visualizer.destroy_window()

    def _add_geometries(self, geo, name):
        if self.geometries.__contains__(name):
            self.visualizer.remove_geometry(self.geometries[name])
            self.visualizer.add_geometry(geo)
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
        else:
            self.visualizer.add_geometry(geo)
        self.geometries[name] = geo

    def add_pointcloud(self, name, pointcloud: o3d.geometry.PointCloud, color: np.ndarray = np.array([120, 20, 30])):
        cloud = pointcloud.paint_uniform_color(color / 255.)
        self._add_geometries(cloud, name)

    def add_pointcloud_by_array(self, name, points: np.ndarray, color: np.ndarray = np.array([120, 20, 30])):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).paint_uniform_color(color / 255.)
        self._add_geometries(cloud, name)
        self.visualizer.add_geometry(cloud)

    def add_local_frame(self, name, frame: LocalFrame, color_scheme=BaseVisualizer.LocalFrameColorScheme_Default):
        self._add_geometries(BaseVisualizer._create_local_frame(frame, color_scheme), name)

    def add_hand(self, name, hand: Hand, style=BaseVisualizer.HandStyle_Cube):
        self._add_geometries(BaseVisualizer._create_hand(hand, style), name)
