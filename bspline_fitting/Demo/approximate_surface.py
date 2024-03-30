import torch
import numpy as np
import open3d as o3d

from bspline_fitting.Method.fitting import approximate_surface
from bspline_fitting.Model.bspline_surface import BSplineSurface


def demo():
    mesh_file_path = "/Users/fufu/Downloads/test.obj"

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    pcd = mesh.sample_points_uniformly(10000, use_triangle_normal=True)

    abb = o3d.geometry.AxisAlignedBoundingBox(
        np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])
    )

    crop_pcd = pcd.crop(abb)

    points = np.asarray(crop_pcd.points)
    size_u = 5
    size_v = 7
    degree_u = 3
    degree_v = 3
    use_centripetal = False
    ctrlpts_size_u = None
    ctrlpts_size_v = None

    surf = approximate_surface(
        points,
        size_u,
        size_v,
        degree_u,
        degree_v,
        use_centripetal,
        ctrlpts_size_u,
        ctrlpts_size_v,
    )

    bspline_surface = BSplineSurface(
        degree_u,
        degree_v,
        size_u,
        size_v,
    )
    bspline_surface.setGradState(True)
    bspline_surface.loadParams(
        surf.data["knotvector"][0],
        surf.data["knotvector"][1],
        surf.data["control_points"],
    )

    sample_points = bspline_surface.toSamplePoints()

    loss = torch.mean(sample_points)
    loss.backward()
    print("bspline_surface.ctrlpts")
    print(bspline_surface.ctrlpts)
    print("bspline_surface.ctrlpts.grad")
    print(bspline_surface.ctrlpts.grad)

    evalpts = np.array(surf.evalpts)
    fitting_pcd = o3d.geometry.PointCloud()
    fitting_pcd.points = o3d.utility.Vector3dVector(evalpts)

    sample_points = sample_points.detach().clone().cpu().numpy()
    sample_fitting_pcd = o3d.geometry.PointCloud()
    sample_fitting_pcd.points = o3d.utility.Vector3dVector(sample_points)

    crop_pcd.translate([1, 0, 0])
    # fitting_pcd.translate([0, 1, 0])
    sample_fitting_pcd.translate([0, 1, 0])

    o3d.visualization.draw_geometries([mesh, crop_pcd, fitting_pcd, sample_fitting_pcd])
    return True
