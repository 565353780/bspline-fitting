import numpy as np
import open3d as o3d
from typing import Union


def translateGeometries(
    translate: Union[list, np.ndarray], geometry_list: list
) -> bool:
    translate = np.ndarray(translate, dtype=float)

    for geometry in geometry_list:
        geometry.translate(translate)
    return True


def getLineSet(
    start: Union[list, np.ndarray],
    vectors: Union[list, np.ndarray],
    color: Union[list, np.ndarray],
) -> o3d.geometry.PointCloud:
    start = np.array(start, dtype=float)
    vectors = np.array(vectors, dtype=float)
    color = np.array(color, dtype=float)

    points = np.vstack([start.reshape(1, -1), start + vectors])
    lines = np.zeros([vectors.shape[0], 2], dtype=int)
    lines[:, 1] = np.arange(1, points.shape[0])
    colors = np.zeros([vectors.shape[0], 3], dtype=float)
    colors[:, :] = color

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def renderGeometries(geometry_list, window_name="Geometry List"):
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name)
    return True


def renderPoints(points: np.ndarray, window_name="Points"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return renderGeometries(pcd, window_name)
