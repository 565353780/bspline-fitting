import os
import torch
import numpy as np
import gradio as gr
import open3d as o3d

from bspline_fitting.Method.render import toPlotFigure
from bspline_fitting.Module.trainer import Trainer


def renderInputData(input_pcd_file_path: str):
    pcd = o3d.io.read_point_cloud(input_pcd_file_path)

    gt_points = np.asarray(pcd.points)

    return toPlotFigure(gt_points)


def fitBSplineSurface(
    input_pcd_file_path: str,
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
    finetune_step_num: int,
    lr: float,
    weight_decay: float,
    factor: float,
    patience: int,
    min_lr: float,
):
    print("input_pcd_file_path:", input_pcd_file_path)
    if not os.path.exists(input_pcd_file_path):
        print("[ERROR][Server::fitBSplineSurface]")
        print("\t input pcd file not exist!")
        print("\t input_pcd_file_path:", input_pcd_file_path)
        return ""

    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    render = False
    render_freq = 1
    render_init_only = False

    save_result_folder_path = None
    save_log_folder_path = None

    input_pcd_file_name = input_pcd_file_path.split("/")[-1]
    save_pcd_file_path = "./output/" + input_pcd_file_name
    overwrite = True
    print_progress = True

    pcd = o3d.io.read_point_cloud(input_pcd_file_path)
    gt_points = np.asarray(pcd.points)

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
    trainer.bspline_surface.saveAsPcdFile(
        save_pcd_file_path, overwrite, print_progress, [0.0, 1.0, 0.0]
    )

    sample_points = (
        trainer.bspline_surface.toSamplePoints().detach().clone().cpu().numpy()
    )

    sample_plot_figure = toPlotFigure(sample_points)

    return save_pcd_file_path, sample_plot_figure


class Server(object):
    def __init__(self, port: int) -> None:
        self.port = port

        self.input_data = None
        return

    def start(self) -> bool:
        example_folder_path = "./output/input_pcd/"
        example_file_name_list = os.listdir(example_folder_path)

        examples = [
            example_folder_path + example_file_name
            for example_file_name in example_file_name_list
        ]

        with gr.Blocks() as iface:
            gr.Markdown("BSpline Fitting Demo")

            with gr.Row():
                with gr.Column():
                    input_pcd = gr.Model3D(label="3D Data to be fitted")

                    gr.Examples(examples=examples, inputs=input_pcd)

                    submit_button = gr.Button("Submit to server")

                output_pcd = gr.Model3D(label="BSpline Surface Sample Points")

            with gr.Row():
                with gr.Column():
                    visual_gt_plot = gr.Plot()

                    fit_button = gr.Button("Click to start fitting")

                visual_sample_plot = gr.Plot()

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
                finetune_step_num = gr.Slider(
                    0, 1000, value=400, step=1, label="finetune_step_num"
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
                finetune_step_num,
                lr,
                weight_decay,
                factor,
                patience,
                min_lr,
            ]

            submit_button.click(
                fn=renderInputData,
                inputs=[input_pcd],
                outputs=[visual_gt_plot],
            )

            fit_button.click(
                fn=fitBSplineSurface,
                inputs=[input_pcd] + bspline_params + fitting_params,
                outputs=[output_pcd, visual_sample_plot],
            )

        iface.launch(
            server_name="0.0.0.0",
            server_port=self.port,
            ssl_keyfile="./ssl/key.pem",
            ssl_certfile="./ssl/cert.pem",
            ssl_verify=False,
        )
        return True
