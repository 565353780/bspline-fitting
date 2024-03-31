import os
import numpy as np
import open3d as o3d
from tqdm import trange
from copy import deepcopy

from bspline_fitting.Config.custom_path import mesh_file_path_dict

if __name__ == "__main__":
    mesh_file_path = mesh_file_path_dict["mac_airplane"]
    save_pcd_file_base_path = "./output/input_pcd/airplane"
    sample_point_num = 10000
    crop_abb_params = np.array(
        [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            [-0.1, -0.1, -0.1, 0.2, 0.2, 0.2],
        ],
        dtype=float,
    )

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    print("start sample points from mesh...")
    pcd = mesh.sample_points_uniformly(sample_point_num, use_triangle_normal=True)

    save_pcd_file_basename = save_pcd_file_base_path.split("/")[-1]
    save_pcd_folder_path = save_pcd_file_base_path[: -len(save_pcd_file_basename)]
    os.makedirs(save_pcd_folder_path, exist_ok=True)

    print("start convert mesh to crop pcd files...")
    for i in trange(crop_abb_params.shape[0]):
        crop_abb = o3d.geometry.AxisAlignedBoundingBox(
            crop_abb_params[i, :3], crop_abb_params[i, 3:]
        )

        copy_pcd = deepcopy(pcd)

        crop_pcd = copy_pcd.crop(crop_abb)
        crop_pcd.paint_uniform_color([1.0, 1.0, 1.0])

        o3d.io.write_point_cloud(
            save_pcd_file_base_path + "_" + str(i) + ".ply",
            crop_pcd,
            write_ascii=True,
            print_progress=True,
        )
