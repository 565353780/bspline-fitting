import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from bspline_fitting.Config.constant import EPSILON
from bspline_fitting.Loss.chamfer_distance import chamferDistance
from bspline_fitting.Method.pcd import getPointCloud
from bspline_fitting.Method.time import getCurrentTime
from bspline_fitting.Method.fitting import approximate_surface
from bspline_fitting.Model.bspline_surface import BSplineSurface
from bspline_fitting.Module.logger import Logger
from bspline_fitting.Module.o3d_viewer import O3DViewer


class Trainer(object):
    def __init__(
        self,
        degree_u: int = 3,
        degree_v: int = 3,
        size_u: int = 5,
        size_v: int = 7,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        start_u: float = 0.0,
        start_v: float = 0.0,
        stop_u: float = 1.0,
        stop_v: float = 1.0,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
        warm_epoch_step_num: int = 20,
        warm_epoch_num: int = 10,
        finetune_step_num: int = 400,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        factor: float = 0.9,
        patience: int = 1,
        min_lr: float = 1e-4,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.bspline_surface = BSplineSurface(
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
        )

        self.warm_epoch_step_num = warm_epoch_step_num
        self.warm_epoch_num = warm_epoch_num

        self.finetune_step_num = finetune_step_num

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        # TODO: can start from 0 and auto upperDegrees later

        self.initRecords()

        self.o3d_viewer = None
        if self.render:
            self.o3d_viewer = O3DViewer()
            self.o3d_viewer.createWindow()

        self.min_lr_reach_time = 0
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def updateBestParams(self, loss: Union[float, None] = None) -> bool:
        if loss is not None:
            if loss >= self.loss_min:
                return False

            self.loss_min = loss

        self.best_params_dict = {
            "degree_u": self.bspline_surface.degree_u,
            "degree_v": self.bspline_surface.degree_v,
            "size_u": self.bspline_surface.size_u,
            "size_v": self.bspline_surface.size_v,
            "knotvector_u": self.bspline_surface.knotvector_u.detach()
            .clone()
            .cpu()
            .numpy(),
            "knotvector_v": self.bspline_surface.knotvector_v.detach()
            .clone()
            .cpu()
            .numpy(),
            "ctrlpts": self.bspline_surface.ctrlpts.detach().clone().cpu().numpy(),
        }
        return True

    def loadParams(
        self,
        knotvector_u: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        knotvector_v: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        ctrlpts: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
    ) -> bool:
        self.bspline_surface.loadParams(knotvector_u, knotvector_v, ctrlpts)

        self.updateBestParams()
        return True

    def loadBestParams(
        self,
        knotvector_u: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        knotvector_v: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        ctrlpts: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
    ) -> bool:
        if isinstance(knotvector_u, list) or isinstance(knotvector_u, tuple):
            knotvector_u = np.array(knotvector_u)
        if isinstance(knotvector_u, np.ndarray):
            knotvector_u = torch.from_numpy(knotvector_u)

        if isinstance(knotvector_v, list) or isinstance(knotvector_v, tuple):
            knotvector_v = np.array(knotvector_v)
        if isinstance(knotvector_v, np.ndarray):
            knotvector_v = torch.from_numpy(knotvector_v)

        if isinstance(ctrlpts, list) or isinstance(ctrlpts, tuple):
            ctrlpts = np.array(ctrlpts)
        if isinstance(ctrlpts, np.ndarray):
            ctrlpts = torch.from_numpy(ctrlpts)

        self.best_params_dict["degree_u"] = self.bspline_surface.degree_u
        self.best_params_dict["degree_v"] = self.bspline_surface.degree_v
        self.best_params_dict["size_u"] = self.bspline_surface.size_u
        self.best_params_dict["size_v"] = self.bspline_surface.size_v
        self.best_params_dict["knotvector_u"] = knotvector_u
        self.best_params_dict["knotvector_v"] = knotvector_v
        self.best_params_dict["ctrlpts"] = ctrlpts
        return True

    def getLr(self, optimizer) -> float:
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def toTrainStepNum(self, scheduler: LRScheduler) -> int:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            return self.finetune_step_num

        if scheduler.T_mult == 1:
            warm_epoch_num = scheduler.T_0 * self.warm_epoch_num
        else:
            warm_epoch_num = int(
                scheduler.T_mult
                * (1.0 - pow(scheduler.T_mult, self.warm_epoch_num))
                / (1.0 - scheduler.T_mult)
            )

        return self.warm_epoch_step_num * warm_epoch_num

    def trainStep(
        self,
        optimizer: Optimizer,
        gt_points: torch.Tensor,
    ) -> Union[dict, None]:
        optimizer.zero_grad()

        detect_points = self.bspline_surface.toSamplePoints()

        fit_dists2, coverage_dists2 = chamferDistance(
            detect_points.reshape(1, -1, 3).type(gt_points.dtype),
            gt_points,
            "cuda" not in self.bspline_surface.device,
        )[:2]

        fit_dists = torch.mean(torch.sqrt(fit_dists2 + EPSILON))
        coverage_dists = torch.mean(torch.sqrt(coverage_dists2 + EPSILON))

        mean_fit_loss = torch.mean(fit_dists)
        mean_coverage_loss = torch.mean(coverage_dists)

        loss = mean_fit_loss + mean_coverage_loss

        loss.backward()

        optimizer.step()

        loss_dict = {
            "fit_loss": mean_fit_loss.detach().clone().cpu().numpy(),
            "coverage_loss": mean_coverage_loss.detach().clone().cpu().numpy(),
            "loss": loss.detach().clone().cpu().numpy(),
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["loss"])

            if self.getLr(optimizer) == self.min_lr:
                self.min_lr_reach_time += 1

            return self.min_lr_reach_time > self.patience

        current_warm_epoch = self.step / self.warm_epoch_step_num
        scheduler.step(current_warm_epoch)

        return current_warm_epoch >= self.warm_epoch_num

    def trainBSplineSurface(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        gt_points: torch.Tensor,
    ) -> bool:
        self.bspline_surface.setGradState(True)

        train_step_num = self.toTrainStepNum(scheduler)
        final_step = self.step + train_step_num

        print("[INFO][BSplineSurfaceModelOp::train]")
        print("\t start training ...")
        pbar = tqdm(total=final_step)
        pbar.update(self.step)
        while self.step < final_step:
            if self.render and self.step % self.render_freq == 0:
                assert self.o3d_viewer is not None
                with torch.no_grad():
                    self.o3d_viewer.clearGeometries()

                    mesh_abb_length = 1.0

                    gt_pcd = getPointCloud(gt_points.reshape(-1, 3).cpu().numpy())
                    gt_pcd.translate([-mesh_abb_length, 0, 0])
                    self.o3d_viewer.addGeometry(gt_pcd)

                    detect_points = (
                        self.bspline_surface.toSamplePoints()
                        .detach()
                        .clone()
                        .cpu()
                        .numpy()
                    )
                    pcd = getPointCloud(detect_points)
                    self.o3d_viewer.addGeometry(pcd)

                    """
                    for j in range(self.bspline_surface.mask_params.shape[0]):
                        view_cone = self.toO3DViewCone(j)
                        view_cone.translate([-mesh_abb_length, 0, 0])
                        self.o3d_viewer.addGeometry(view_cone)

                        # inv_sphere = self.toO3DInvSphere(j)
                        # inv_sphere.translate([-30, 0, 0])
                        # self.o3d_viewer.addGeometry(inv_sphere)
                    """

                    self.o3d_viewer.update()

                    if self.render_init_only:
                        self.o3d_viewer.run()
                        exit()

            loss_dict = self.trainStep(
                optimizer,
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            if self.logger.isValid():
                for key, item in loss_dict.items():
                    self.logger.addScalar("Train/" + key, item, self.step)
                self.logger.addScalar("Train/lr", self.getLr(optimizer), self.step)

            self.updateBestParams(loss_dict["loss"])

            pbar.set_description(
                "LOSS %.6f LR %.4f"
                % (
                    loss_dict["loss"],
                    self.getLr(optimizer) / self.lr,
                )
            )

            self.autoSaveBSplineSurface("train")

            if self.checkStop(optimizer, scheduler, loss_dict):
                break

            self.step += 1
            pbar.update(1)

        return True

    def autoTrainBSplineSurface(
        self,
        gt_points: Union[np.ndarray, list, tuple],
    ) -> bool:
        print("[INFO][Trainer::autoTrainBSplineSurface]")
        print("\t start auto train BSplineSurface...")
        print(
            "\t degree_u:",
            self.bspline_surface.degree_u,
            ", degree_v:",
            self.bspline_surface.degree_v,
            ", size_u:",
            self.bspline_surface.size_u,
            ", size_v:",
            self.bspline_surface.size_v,
        )

        if isinstance(gt_points, list) or isinstance(gt_points, tuple):
            gt_points = np.array(gt_points)

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
        target_sample_point_num = (
            4 * self.bspline_surface.sample_num_u * self.bspline_surface.sample_num_v
        )
        if target_sample_point_num < gt_points.shape[0]:
            try:
                downsample_pcd = gt_pcd.farthest_point_down_sample(target_sample_point_num)
            except:
                downsample_pcd = gt_pcd.uniform_down_sample(
                    int(gt_points.shape[0] / target_sample_point_num)
                )

            gt_points = np.asarray(downsample_pcd.points)

        max_point = np.max(gt_points, axis=0)
        min_point = np.min(gt_points, axis=0)
        center = (max_point + min_point) / 2.0
        scale = np.max(max_point - min_point)

        gt_points = (gt_points - center) / scale

        if False:
            surf = approximate_surface(
                gt_points,
                self.bspline_surface.size_u,
                self.bspline_surface.size_v,
                self.bspline_surface.degree_u,
                self.bspline_surface.degree_v,
            )
            ctrlpts = np.array(surf.data["control_points"])

            self.bspline_surface.loadParams(ctrlpts=ctrlpts)
        else:
            ctrlpts = np.zeros(
                [
                    self.bspline_surface.size_u - 1,
                    self.bspline_surface.size_v - 1,
                    3,
                ],
                dtype=float,
            )

            u_values = (
                np.arange(self.bspline_surface.size_u - 1)
                / (self.bspline_surface.size_u - 2)
            ) - 0.5

            v_values = (
                np.arange(self.bspline_surface.size_v - 1)
                / (self.bspline_surface.size_v - 2)
            ) - 0.5

            for i in range(self.bspline_surface.size_u - 1):
                ctrlpts[:, i, 0] = u_values
            for i in range(self.bspline_surface.size_v - 1):
                ctrlpts[i, :, 1] = v_values

            ctrlpts = ctrlpts.reshape(-1, 3) * 0.4

            self.bspline_surface.loadParams(ctrlpts=ctrlpts)

        gt_points = torch.from_numpy(gt_points)

        if self.bspline_surface.device == "cpu":
            gt_points_dtype = self.bspline_surface.dtype
        else:
            gt_points_dtype = torch.float32

        gt_points = (
            gt_points.type(gt_points_dtype)
            .to(self.bspline_surface.device)
            .reshape(1, -1, 3)
        )

        while True:
            optimizer = AdamW(
                [
                    # self.bspline_surface.knotvector_u,
                    # self.bspline_surface.knotvector_v,
                    self.bspline_surface.ctrlpts,
                ],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            warm_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
            finetune_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
            )

            self.trainBSplineSurface(optimizer, warm_scheduler, gt_points)
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr
            self.trainBSplineSurface(optimizer, finetune_scheduler, gt_points)

            break

        self.bspline_surface.ctrlpts.data = self.bspline_surface.ctrlpts.data * scale + torch.from_numpy(center)

        return True

    def autoSaveBSplineSurface(self, state_info: str) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_file_path = (
            self.save_result_folder_path
            + str(self.save_file_idx)
            + "_"
            + state_info
            + ".npy"
        )

        save_bspline_surface = deepcopy(self.bspline_surface)
        save_bspline_surface.loadParamsDict(self.best_params_dict)
        save_bspline_surface.saveParamsFile(save_file_path, True)

        self.save_file_idx += 1
        return True
