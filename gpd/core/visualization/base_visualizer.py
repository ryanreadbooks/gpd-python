import numpy as np
import open3d as o3d

from ..entity import LocalFrame, Hand

TriangleMesh = o3d.geometry.TriangleMesh
LineSet = o3d.geometry.LineSet


class BaseVisualizer:
    LocalFrameColorScheme_Default = 'axis-default'
    LocalFrameColorScheme_Transform = 'axis-transformed'

    HandStyle_Cube = 'hand-style-cube'
    HandStyle_Line = 'hand-style-line'

    def __init__(self):
        pass

    @staticmethod
    def _create_hand(hand: Hand, style=HandStyle_Cube) -> LineSet:
        hand_line_set = None
        hand_points = hand.hand_points_vis
        if style == BaseVisualizer.HandStyle_Cube:
            hand_points_list = hand_points.tolist()
            hand_lines_right = [[1, 9], [1, 4], [9, 10], [4, 10], [9, 17], [10, 20], [1, 5], [4, 8], [5, 8], [17, 20]]
            hand_lines_left = [[2, 3], [2, 13], [3, 14], [13, 14], [2, 6], [3, 7], [13, 18], [14, 19], [6, 7], [18, 19]]
            hand_lines_bottom = [[5, 6], [7, 8], [17, 18], [19, 20]]
            hand_lines = hand_lines_left + hand_lines_right + hand_lines_bottom
            hand_line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(hand_points_list), lines=o3d.utility.Vector2iVector(hand_lines))
            left_colors = [[1.0, 0.64, 0.0] for _ in hand_lines_left]
            right_colors = [[0.27, 1.0, 0.0] for _ in hand_lines_right]
            bottom_colors = [[0.18, 0.31, 0.31] for _ in hand_lines_bottom]
            colors = left_colors + right_colors + bottom_colors
            hand_line_set.colors = o3d.utility.Vector3dVector(colors)
        else:
            p_right_58_mid = (hand_points[5] + hand_points[8]) / 2.
            p_left_67_mid = (hand_points[6] + hand_points[7]) / 2.
            p_right_14_mid = (hand_points[1] + hand_points[4]) / 2.
            p_left_23_mid = (hand_points[2] + hand_points[3]) / 2.
            origin = hand_points[0]
            points = [origin.tolist(), p_right_58_mid.tolist(), p_left_67_mid.tolist(),
                      p_right_14_mid.tolist(), p_left_23_mid.tolist()]
            lines = [[0, 1], [0, 2], [1, 3], [2, 4]]
            left_hand_color = [[1.0, 0.64, 0.0]]
            right_hand_color = [[0.27, 1.0, 0.0]]
            bottom_color = [[0.18, 0.31, 0.31], [0.18, 0.31, 0.31]]
            hand_line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
            colors = bottom_color + right_hand_color + left_hand_color
            hand_line_set.colors = o3d.utility.Vector3dVector(colors)

        return hand_line_set

    @staticmethod
    def _create_local_frame(frame: LocalFrame, color_scheme) -> LineSet:
        # the origin of local frame in the the world reference frame
        frame_origin: np.ndarray = frame.sample
        # orientation vector * length
        point_x = (frame.normal_axis * 0.01 + frame_origin).tolist()
        point_y = (frame.binormal_axis * 0.01 + frame_origin).tolist()
        point_z = (frame.curvature_axis * 0.01 + frame_origin).tolist()
        points = [frame_origin.tolist(), point_x, point_y, point_z]
        lines = [[0, 1], [0, 2], [0, 3]]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines))
        # x y z axes color
        colors = []
        if color_scheme == BaseVisualizer.LocalFrameColorScheme_Default:
            colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif color_scheme == BaseVisualizer.LocalFrameColorScheme_Transform:
            colors = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
        else:
            raise ValueError('No such color for local frame')
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set
