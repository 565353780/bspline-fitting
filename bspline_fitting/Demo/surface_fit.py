import numpy as np
import open3d as o3d

from bspline_fitting.Module.fitter import Fitter


def demo():
    mesh_file_path = "/Users/fufu/Downloads/test.obj"

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    # o3d.visualization.draw_geometries([mesh])

    # Data set
    points = np.array(
        (
            (-5, -5, 0),
            (-2.5, -5, 0),
            (0, -5, 0),
            (2.5, -5, 0),
            (5, -5, 0),
            (7.5, -5, 0),
            (10, -5, 0),
            (-5, 0, 3),
            (-2.5, 0, 3),
            (0, 0, 3),
            (2.5, 0, 3),
            (5, 0, 3),
            (7.5, 0, 3),
            (10, 0, 3),
            (-5, 5, 0),
            (-2.5, 5, 0),
            (0, 5, 0),
            (2.5, 5, 0),
            (5, 5, 0),
            (7.5, 5, 0),
            (10, 5, 0),
            (-5, 7.5, -3),
            (-2.5, 7.5, -3),
            (0, 7.5, -3),
            (2.5, 7.5, -3),
            (5, 7.5, -3),
            (7.5, 7.5, -3),
            (10, 7.5, -3),
            (-5, 10, 0),
            (-2.5, 10, 0),
            (0, 10, 0),
            (2.5, 10, 0),
            (5, 10, 0),
            (7.5, 10, 0),
            (10, 10, 0),
        )
    )
    size_u = 5
    size_v = 7
    degree_u = 2
    degree_v = 3

    fitter = Fitter()

    fitter.fit(points, size_u, size_v, degree_u, degree_v)
    return True
