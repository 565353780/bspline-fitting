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
    warm_epoch_step_num: int,
    warm_epoch_num: int,
    finetune_epoch_num: int,
    lr: float,
    weight_decay: float,
    factor: float,
    patience: float,
    min_lr: float,
):
    print("mesh_file_path:", mesh_file_path)
    return mesh_file_path


class Server(object):
    def __init__(self, port: int) -> None:
        self.port = port
        return

    def start(self) -> bool:
        with gr.Blocks() as iface:
            gr.Markdown("BSpline Fitting Demo")

            with gr.Row():
                with gr.Column():
                    input_mesh = gr.Model3D(label="3D Data to be fitted")

                    gr.Examples(examples=mac_examples, inputs=input_mesh)

                    submit_button = gr.Button("Fitting")

                output_mesh = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="Fitting BSpline Sample Points",
                )

            with gr.Accordion(label="BSpline Params", open=False):
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

            bspline_params = [
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
            ]

            with gr.Accordion(label="Fitting Params", open=False):
                warm_epoch_step_num = gr.Slider(
                    0, 40, value=20, step=1, label="warm_epoch_step_num"
                )
                warm_epoch_num = gr.Slider(
                    0, 10, value=4, step=1, label="warm_epoch_num"
                )
                finetune_epoch_num = gr.Slider(
                    0, 1000, value=400, step=1, label="finetune_epoch_num"
                )
                lr = gr.Slider(1e-6, 1.0, value=5e-2, step=1e-10, label="lr")
                weight_decay = gr.Slider(
                    1e-10, 1.0, value=1e-4, step=1e-10, label="weight_decay"
                )
                factor = gr.Slider(0.01, 0.99, value=0.9, step=0.01, label="factor")
                patience = gr.Slider(1, 100, value=1, step=1, label="patience")
                min_lr = gr.Slider(1e-10, 1.0, value=1e-3, step=1e-10, label="min_lr")

            fitting_params = [
                warm_epoch_step_num,
                warm_epoch_num,
                finetune_epoch_num,
                lr,
                weight_decay,
                factor,
                patience,
                min_lr,
            ]

            submit_button.click(
                fn=load_mesh,
                inputs=[input_mesh] + bspline_params + fitting_params,
                outputs=[output_mesh],
            )

        iface.launch(server_name="0.0.0.0", server_port=self.port)
        return True
