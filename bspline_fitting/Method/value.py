def linspace(start: float, stop: float, num: int) -> list:
    if abs(stop - start) <= 10e-8:
        return [start]

    if num < 2:
        return [start]

    div = num - 1
    delta = stop - start
    return [start + i * delta / div for i in range(num)]


def find_span_linear(
    degree: int, knot_vector: list, num_ctrlpts: int, knot: int
) -> int:
    span = degree + 1

    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1

    return span - 1


def find_spans(degree: int, knot_vector: list, num_ctrlpts: int, knots: list) -> list:
    spans = []
    for knot in knots:
        spans.append(find_span_linear(degree, knot_vector, num_ctrlpts, knot))
    return spans


def basis_function(degree: int, knot_vector: list, span: int, knot: int) -> list:
    left = [0.0 for _ in range(degree + 1)]
    right = [0.0 for _ in range(degree + 1)]
    N = [1.0 for _ in range(degree + 1)]

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


def basis_functions(degree: int, knot_vector: list, spans: list, knots: list) -> list:
    basis = []
    for span, knot in zip(spans, knots):
        basis.append(basis_function(degree, knot_vector, span, knot))
    return basis


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

    spans = [[] for _ in range(2)]
    basis = [[] for _ in range(2)]
    for idx in range(2):
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
