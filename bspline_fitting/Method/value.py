import torch
import bs_fit_cpp


def basis_function_torch(degree, knot_vector, span, knot):
    dtype = knot_vector.dtype
    device = knot_vector.device

    left = [torch.zeros([1], dtype=dtype).to(device) for _ in range(degree + 1)]
    right = [torch.zeros([1], dtype=dtype).to(device) for _ in range(degree + 1)]
    N = [torch.zeros([1], dtype=dtype).to(device) for _ in range(degree + 1)]

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot

        saved = torch.zeros([1], dtype=dtype).to(device)

        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def basis_functions_torch(degree, knot_vector, spans, knots):
    basis = []

    for i in range(len(spans)):
        span = spans[i]
        knot = knots[i]

        current_basis_function = basis_function_torch(degree, knot_vector, span, knot)

        basis.append(current_basis_function)

    return basis


def toTorchPoints(
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
    knotvector_u,
    knotvector_v,
    ctrlpts,
) -> torch.Tensor:
    knots_u = bs_fit_cpp.linspace(start_u, stop_u, sample_num_u)
    knots_v = bs_fit_cpp.linspace(start_v, stop_v, sample_num_v)

    knotvector_u_vec = knotvector_u.detach().clone().numpy().tolist()
    knotvector_v_vec = knotvector_v.detach().clone().numpy().tolist()

    spans_u = bs_fit_cpp.find_spans(degree_u, knotvector_u_vec, size_u, knots_u)
    spans_v = bs_fit_cpp.find_spans(degree_v, knotvector_v_vec, size_v, knots_v)

    basis_u = basis_functions_torch(degree_u, knotvector_u, spans_u, knots_u)
    basis_v = basis_functions_torch(degree_v, knotvector_v, spans_v, knots_v)

    dtype = ctrlpts.dtype
    device = ctrlpts.device

    eval_points = torch.zeros([sample_num_u * sample_num_v, 3], dtype=dtype).to(device)

    for i in range(len(spans_u)):
        idx_u = spans_u[i] - degree_u

        for j in range(len(spans_v)):
            idx_v = spans_v[j] - degree_v

            spt = torch.zeros([3], dtype=dtype).to(device)

            for k in range(degree_u + 1):
                temp = torch.zeros([3], dtype=dtype).to(device)

                for l in range(degree_v + 1):
                    temp = (
                        temp
                        + basis_v[j][l] * ctrlpts[idx_v + l + (size_v * (idx_u + k))]
                    )

                spt = spt + basis_u[i][k] * temp

            eval_points[i * len(spans_u) + j] = spt

    return eval_points
