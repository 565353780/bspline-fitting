import torch
import bs_fit_cpp


def toTorchPoints(
    degree: list,
    u_knotvector: list,
    v_knotvector: list,
    ctrlpts: torch.Tensor,
    size: list,
    start: list,
    stop: list,
    sample_size: list,
) -> torch.Tensor:
    spans = bs_fit_cpp.toSpans(
        degree, u_knotvector, v_knotvector, size, start, stop, sample_size
    )

    basis = bs_fit_cpp.toBasis(
        degree, u_knotvector, v_knotvector, start, stop, sample_size, spans
    )

    dtype = ctrlpts.dtype
    device = ctrlpts.device

    eval_points = torch.zeros([sample_size[0] * sample_size[1], 3], dtype=dtype).to(
        device
    )

    for i in range(len(spans[0])):
        idx_u = spans[0][i] - degree[0]

        for j in range(len(spans[1])):
            idx_v = spans[1][j] - degree[1]

            spt = torch.zeros([3], dtype=dtype).to(device)

            for k in range(degree[0] + 1):
                temp = torch.zeros([3], dtype=dtype).to(device)

                for l in range(degree[1] + 1):
                    temp = (
                        temp
                        + basis[1][j][l] * ctrlpts[idx_v + l + (size[1] * (idx_u + k))]
                    )

                spt = spt + basis[0][i][k] * temp

            eval_points[i * len(spans[0]) + j] = spt

    return eval_points
