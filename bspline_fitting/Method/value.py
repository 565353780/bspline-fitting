import numpy as np


def linspace(start: float, stop: float, num: int) -> np.ndarray:
    if abs(stop - start) <= 10e-8:
        return np.array([start])

    if num < 2:
        return np.array([start])

    div = num - 1
    delta = stop - start
    return start + np.arange(num, dtype=float) * delta / div


def find_span_linear(
    degree: int, knot_vector: np.ndarray, num_ctrlpts: int, knot: int
) -> int:
    span = degree + 1

    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1

    return span - 1


def find_spans(
    degree: int, knot_vector: np.ndarray, num_ctrlpts: int, knots: np.ndarray
) -> np.ndarray:
    spans = []
    for knot in knots:
        spans.append(find_span_linear(degree, knot_vector, num_ctrlpts, knot))
    return np.array(spans, dtype=int)


def basis_function(
    degree: int, knot_vector: np.ndarray, span: int, knot: int
) -> np.ndarray:
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)
    N = np.ones(degree + 1, dtype=float)

    left[1 : degree + 1] = knot - knot_vector[span - degree + 1 : span + 1][::-1]
    right[1 : degree + 1] = knot_vector[span + 1 : span + degree + 1] - knot

    for j in range(1, degree + 1):
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


def basis_functions(degree, knot_vector, spans, knots):
    basis = []
    for span, knot in zip(spans, knots):
        basis.append(basis_function(degree, knot_vector, span, knot))
    return np.array(basis)


def evaluate(datadict, param):
    start = param
    stop = param

    # Geometry data from datadict
    sample_size = datadict["sample_size"]
    degree = datadict["degree"]
    knotvector = datadict["knotvector"]
    ctrlpts = datadict["control_points"]
    size = datadict["size"]
    dimension = (
        datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
    )
    pdimension = datadict["pdimension"]
    precision = datadict["precision"]

    # Algorithm A3.5
    spans = [[] for _ in range(pdimension)]
    basis = [[] for _ in range(pdimension)]
    for idx in range(pdimension):
        knots = linspace(start[idx], stop[idx], sample_size[idx])
        spans[idx] = find_spans(degree[idx], knotvector[idx], size[idx], knots)
        basis[idx] = basis_functions(degree[idx], knotvector[idx], spans[idx], knots)

    eval_points = []
    for i in range(len(spans[0])):
        idx_u = spans[0][i] - degree[0]
        for j in range(len(spans[1])):
            idx_v = spans[1][j] - degree[1]
            spt = [0.0 for _ in range(dimension)]
            for k in range(0, degree[0] + 1):
                temp = [0.0 for _ in range(dimension)]
                for l in range(0, degree[1] + 1):
                    temp[:] = [
                        tmp + (basis[1][j][l] * cp)
                        for tmp, cp in zip(
                            temp, ctrlpts[idx_v + l + (size[1] * (idx_u + k))]
                        )
                    ]
                spt[:] = [pt + (basis[0][i][k] * tmp) for pt, tmp in zip(spt, temp)]

            eval_points.append(spt)

    return eval_points
