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
        return

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
        self.bspline_surface.ctrlpts.data = self.bspline_surface.ctrlpts.data * scale + torch.from_numpy(center)

        return True
