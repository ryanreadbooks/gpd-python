import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from .base_visualizer import BaseVisualizer
from ..entity import LocalFrame, Hand


class StaticVisualizer(BaseVisualizer):

    def __init__(self, need_coord=False):
        super(StaticVisualizer, self).__init__()
        self.geometries = []
        if need_coord:
            self.geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.01, [0, 0, 0]))

    def show(self):
        o3d.visualization.draw_geometries(self.geometries)

    def add_pointcloud(self, pointcloud: o3d.geometry.PointCloud, color: np.ndarray = np.array([120, 20, 30])):
        cloud = pointcloud.paint_uniform_color(color / 255.)
        self.geometries.append(cloud)

    def add_pointcloud_by_array(self, points: np.ndarray, color: np.ndarray = np.array([120, 20, 30])):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).paint_uniform_color(color / 255.)
        self.geometries.append(cloud)

    def add_local_frame(self, frame: LocalFrame, color_scheme=BaseVisualizer.LocalFrameColorScheme_Default):
        frame_lineset = BaseVisualizer._create_local_frame(frame, color_scheme)
        self.geometries.append(frame_lineset)

    def add_hand(self, hand: Hand, show_frame=False, style=BaseVisualizer.HandStyle_Cube):
        hand_lineset = super(StaticVisualizer, self)._create_hand(hand, style)
        self.geometries.append(hand_lineset)
        if show_frame:
            self.add_local_frame(hand.frame, BaseVisualizer.LocalFrameColorScheme_Transform)

    def render_to_img(self):
        if len(self.geometries) != 0:
            yellow = rendering.Material()
            yellow.base_color = [1.0, 0.75, 0.0, 1.0]
            yellow.shader = "defaultLit"
            render = rendering.OffscreenRenderer(1920, 1080, headless=True)
            for i, geometry in enumerate(self.geometries):
                render.scene.add_geometry(f'g-{i}', geometry, yellow)
            # render.setup_camera(60.0, [0,0,0], [0,10,0], [0,0,1])
            render.scene.scene.enable_sun_light(False)
            render.scene.show_axes(True)
            img = render.render_to_image()
            print('rendered image generated')
            o3d.io.write_image('scene.png', img)