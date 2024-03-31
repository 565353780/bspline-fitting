import torch
import numpy as np
import gradio as gr
import open3d as o3d

from bspline_fitting.Config.custom_path import mesh_file_path_dict
from bspline_fitting.Module.trainer import Trainer


linux_examples = [
    [mesh_file_path_dict["linux_bunny"]],
    [mesh_file_path_dict["linux_airplane"]],
    [mesh_file_path_dict["linux_plane"]],
    [mesh_file_path_dict["linux_1"]],
    [mesh_file_path_dict["linux_2"]],
]
mac_examples = [
    [mesh_file_path_dict["mac_bunny"]],
    [mesh_file_path_dict["mac_airplane"]],
    [mesh_file_path_dict["mac_plane"]],
    [mesh_file_path_dict["mac_chair_0"]],
    [mesh_file_path_dict["mac_chair_1"]],
    [mesh_file_path_dict["mac_chair_2"]],
]


def load_mesh(
    mesh_file_path: str,
    degree_u: int,
    degree_v: int,
    size_u: int,
    size_v: int,
    sample_num_u: int,
    sample_num_v: int,
    start_u: float,
    start_v: float,
    stop_u: float,
    stop_v: float,
):
    print("mesh_file_path:", mesh_file_path)
    return mesh_file_path


class Server(object):
    def __init__(self, port: int) -> None:
        self.port = port
        return

    def start(self) -> bool:
        mesh = gr.Model3D()

        degree_u = gr.Slider(0, 10, value=3, step=1, label="degree_u")
        degree_v = gr.Slider(0, 10, value=3, step=1, label="degree_v")
        size_u = gr.Slider(2, 10, value=7, step=1, label="size_u")
        size_v = gr.Slider(2, 10, value=7, step=1, label="size_v")
        sample_num_u = gr.Slider(2, 20, value=20, step=1, label="sample_num_u")
        sample_num_v = gr.Slider(2, 20, value=20, step=1, label="sample_num_v")
        start_u = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="start_u")
        start_v = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="start_v")
        stop_u = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="stop_u")
        stop_v = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="stop_v")

        iface = gr.Interface(
            fn=load_mesh,
            inputs=[
                mesh,
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
            ],
            outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
            examples=mac_examples,
        )

        iface.launch(server_name="0.0.0.0", server_port=self.port)
        return True
