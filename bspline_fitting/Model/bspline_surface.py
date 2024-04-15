import os
import torch
import numpy as np
import open3d as o3d
from typing import Union

import bs_fit_cpp

from bspline_fitting.Method.check import checkShape
from bspline_fitting.Method.value import toTorchPoints
from bspline_fitting.Method.render import renderPoints
from bspline_fitting.Method.path import createFileFolder, removeFile, renameFile


class BSplineSurface(object):
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
    ) -> None:
        # Super Params
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.size_u = size_u
        self.size_v = size_v
        self.sample_num_u = sample_num_u
        self.sample_num_v = sample_num_v
        self.start_u = start_u
        self.start_v = start_v
        self.stop_u = stop_u
        self.stop_v = stop_v
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        # Diff Params
        self.knotvector_u = torch.zeros(
            self.size_u - self.degree_u - 1, dtype=self.dtype
        ).to(self.device)
        self.knotvector_v = torch.zeros(
            self.size_v - self.degree_v - 1, dtype=self.dtype
        ).to(self.device)
        self.ctrlpts = torch.zeros(
            [(self.size_u - 1) * (self.size_v - 1), 3], dtype=self.dtype
        ).to(self.device)

        self.reset()
        return

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        start_u: float = 0.0,
        start_v: float = 0.0,
        stop_u: float = 1.0,
        stop_v: float = 1.0,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cuda:0",
    ):
        degree_u = params_dict["degree_u"]
        degree_v = params_dict["degree_v"]
        size_u = params_dict["size_u"]
        size_v = params_dict["size_v"]

        bspline_surface = cls(
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

        bspline_surface.loadParamsDict(params_dict)

        return bspline_surface

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        start_u: float = 0.0,
        start_v: float = 0.0,
        stop_u: float = 1.0,
        stop_v: float = 1.0,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cuda:0",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
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

    def reset(self) -> bool:
        self.initParams()
        return True

    def setGradState(self, need_grad: bool) -> bool:
        self.knotvector_u.requires_grad_(need_grad)
        self.knotvector_v.requires_grad_(need_grad)
        self.ctrlpts.requires_grad_(need_grad)
        return True

    def initParams(self) -> bool:
        return True

    def loadParams(
        self,
        knotvector_u: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        knotvector_v: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        ctrlpts: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
    ) -> bool:
        if knotvector_u is not None:
            if isinstance(knotvector_u, list) or isinstance(knotvector_u, tuple):
                knotvector_u = np.array(knotvector_u)

            if not checkShape(knotvector_u.shape, self.knotvector_u.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for knotvector_u!")
                return False

            if isinstance(knotvector_u, np.ndarray):
                knotvector_u = torch.from_numpy(knotvector_u)

            self.knotvector_u.data = (
                knotvector_u.detach().clone().type(self.dtype).to(self.device)
            )

        if knotvector_v is not None:
            if isinstance(knotvector_v, list) or isinstance(knotvector_v, tuple):
                knotvector_v = np.array(knotvector_v)

            if not checkShape(knotvector_v.shape, self.knotvector_v.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for knotvector_v!")
                return False

            if isinstance(knotvector_v, np.ndarray):
                knotvector_v = torch.from_numpy(knotvector_v)

            self.knotvector_v.data = (
                knotvector_v.detach().clone().type(self.dtype).to(self.device)
            )

        if ctrlpts is not None:
            if isinstance(ctrlpts, list) or isinstance(ctrlpts, tuple):
                ctrlpts = np.array(ctrlpts)

            if not checkShape(ctrlpts.shape, self.ctrlpts.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for ctrlpts!")
                return False

            if isinstance(ctrlpts, np.ndarray):
                ctrlpts = torch.from_numpy(ctrlpts)

            self.ctrlpts.data = (
                ctrlpts.detach().clone().type(self.dtype).to(self.device)
            )

        return True

    def loadParamsDict(self, params_dict: dict) -> bool:
        knotvector_u = params_dict["knotvector_u"]
        knotvector_v = params_dict["knotvector_v"]
        ctrlpts = params_dict["ctrlpts"]

        self.loadParams(knotvector_u, knotvector_v, ctrlpts)

        return True

    def loadParamsFile(self, params_file_path: str) -> bool:
        if not os.path.exists(params_file_path):
            print("[ERROR][BSplineSurface::loadParamsFile]")
            print("\t params dict file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        if not self.loadParamsDict(params_dict):
            print("[ERROR][BSplineSurface::loadParamsFile]")
            print("\t loadParamsDict failed!")
            return False

        return True

    def toSamplePoints(self) -> torch.Tensor:
        full_knotvector_u = torch.zeros(
            [self.degree_u + self.size_u], dtype=self.knotvector_u.dtype
        ).to(self.knotvector_u.device)
        full_knotvector_v = torch.zeros(
            [self.degree_v + self.size_v], dtype=self.knotvector_v.dtype
        ).to(self.knotvector_v.device)

        full_knotvector_u[-self.degree_u - 1 :] = 1.0
        full_knotvector_v[-self.degree_v - 1 :] = 1.0

        sigmoid_knotvector_u = torch.sigmoid(self.knotvector_u)
        sigmoid_knotvector_v = torch.sigmoid(self.knotvector_v)

        sigmoid_knotvector_u_sum = torch.sum(sigmoid_knotvector_u)
        sigmoid_knotvector_v_sum = torch.sum(sigmoid_knotvector_v)

        normed_sigmoid_knotvector_u = sigmoid_knotvector_u / sigmoid_knotvector_u_sum
        normed_sigmoid_knotvector_v = sigmoid_knotvector_v / sigmoid_knotvector_v_sum

        for i in range(normed_sigmoid_knotvector_u.shape[0] - 1):
            full_knotvector_u[self.degree_u + 1 + i] = (
                full_knotvector_u[self.degree_u + i] + normed_sigmoid_knotvector_u[i]
            )
        for i in range(normed_sigmoid_knotvector_v.shape[0] - 1):
            full_knotvector_v[self.degree_v + 1 + i] = (
                full_knotvector_v[self.degree_v + i] + normed_sigmoid_knotvector_v[i]
            )

        sample_points = toTorchPoints(
            self.degree_u,
            self.degree_v,
            self.size_u - 1,
            self.size_v - 1,
            self.sample_num_u,
            self.sample_num_v,
            self.start_u,
            self.start_v,
            self.stop_u,
            self.stop_v,
            full_knotvector_u,
            full_knotvector_v,
            self.ctrlpts,
        )

        return sample_points

    def renderSamplePoints(self) -> bool:
        sample_points = self.toSamplePoints().detach().clone().cpu().numpy()
        print(sample_points.shape)

        renderPoints(sample_points)
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "size_u": self.size_u,
            "size_v": self.size_v,
            "knotvector_u": self.knotvector_u.detach().clone().cpu().numpy(),
            "knotvector_v": self.knotvector_v.detach().clone().cpu().numpy(),
            "ctrlpts": self.ctrlpts.detach().clone().cpu().numpy(),
        }
        return params_dict

    def saveParamsFile(
        self, save_params_file_path: str, overwrite: bool = False
    ) -> bool:
        if os.path.exists(save_params_file_path):
            if overwrite:
                removeFile(save_params_file_path)
            else:
                print("[WARN][BSplineSurface::saveParamsFile]")
                print("\t save params dict file already exist!")
                print("\t save_params_file_path:", save_params_file_path)
                return False

        params_dict = self.toParamsDict()

        createFileFolder(save_params_file_path)

        tmp_save_params_file_path = save_params_file_path[:-4] + "_tmp.npy"
        removeFile(tmp_save_params_file_path)

        np.save(tmp_save_params_file_path, params_dict)
        renameFile(tmp_save_params_file_path, save_params_file_path)
        return True

    def saveAsPcdFile(
        self,
        save_pcd_file_path: str,
        overwrite: bool = False,
        print_progress: bool = False,
        uniform_color=None,
    ) -> bool:
        if os.path.exists(save_pcd_file_path):
            if overwrite:
                removeFile(save_pcd_file_path)
            else:
                print("[ERROR][BSplineSurface::saveAsPcdFile]")
                print("\t save pcd file already exist!")
                print("\t save_pcd_file_path:", save_pcd_file_path)
                return False

        createFileFolder(save_pcd_file_path)

        points = self.toSamplePoints().detach().clone().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if uniform_color is not None:
            pcd.paint_uniform_color(uniform_color)

        if print_progress:
            print("[INFO][BSplineSurface::saveAsPcdFile]")
            print("\t start save as pcd file...")
        o3d.io.write_point_cloud(
            save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )
        return True
