import torch
import bs_fit_cpp


def evaluate(
    datadict,
    start: list = [0.0, 0.0],
    stop: list = [1.0, 1.0],
    sample_size: list = [20, 20],
):
    degree = datadict["degree"]
    knotvector = datadict["knotvector"]
    ctrlpts = datadict["control_points"]
    size = datadict["size"]
    dimension = (
        datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
    )

    eval_points = bs_fit_cpp.toPoints(
        degree,
        knotvector[0],
        knotvector[1],
        ctrlpts,
        size,
        dimension,
        start,
        stop,
        sample_size,
    )

    return eval_points
