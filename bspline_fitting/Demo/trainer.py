import torch

import numpy as np
import open3d as o3d

from bspline_fitting.Config.custom_path import mesh_file_path_dict
from bspline_fitting.Module.trainer import Trainer


def demo():
    degree_u = 2
    degree_v = 2
    size_u = 7
    size_v = 7
    sample_num_u = 20
    sample_num_v = 20
    start_u = 0.0
    start_v = 0.0
    stop_u = 1.0
    stop_v = 1.0
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    warm_epoch_step_num = 20
    warm_epoch_num = 4
    finetune_step_num = 400
    lr = 5e-2
    weight_decay = 1e-4
    factor = 0.9
    patience = 1
    min_lr = 1e-3

    render = True
    render_freq = 1
    render_init_only = False

    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    mesh_name = "mac_airplane"

    if False:
        mesh_file_path = mesh_file_path_dict[mesh_name]
        mesh = o3d.io.read_triangle_mesh(mesh_file_path)
        pcd = mesh.sample_points_uniformly(10000, use_triangle_normal=True)
        abb = o3d.geometry.AxisAlignedBoundingBox(
            np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])
        )
        crop_pcd = pcd.crop(abb)
        gt_points = np.asarray(crop_pcd.points)

    if True:
        pcd_file_path = "./output/input_pcd/airplane_0.ply"
        crop_pcd = o3d.io.read_point_cloud(pcd_file_path)
        gt_points = np.asarray(crop_pcd.points)

    save_params_file_path = "./output/" + mesh_name + ".npy"
    save_pcd_file_path = "./output/" + mesh_name + ".ply"
    overwrite = True
    print_progress = True

    trainer = Trainer(
        degree_u,
        degree_v,
        size_u,
        size_v,
        sample_num_u,
        sample_num_v,
        start_u,
        start_v,
        stop_u,
        stop_v,
        idx_dtype,
        dtype,
        device,
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        render,
        render_freq,
        render_init_only,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.autoTrainBSplineSurface(gt_points)
    trainer.bspline_surface.saveParamsFile(save_params_file_path, overwrite)
    trainer.bspline_surface.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

    if trainer.o3d_viewer is not None:
        trainer.o3d_viewer.run()
    trainer.bspline_surface.renderSamplePoints()
    return True
