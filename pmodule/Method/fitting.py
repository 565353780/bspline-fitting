import numpy as np
from typing import Union

geomdl_exist = False
try:
    from geomdl.BSpline import Surface
    geomdl_exist = True
except:
    class Surface(object):
        def __init__(self) -> None:
            return

def compute_params_curve(points, centripetal=False):
    num_points = points.shape[0]

    cds = np.zeros(num_points + 1, dtype=float)
    cds[-1] = 1.0
    for i in range(1, num_points):
        distance = np.linalg.norm(points[i] - points[i - 1], ord=2, axis=0)

        if centripetal:
            cds[i] = np.sqrt(distance)
        else:
            cds[i] = distance

    d = np.sum(cds[1:-1])

    uk = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        uk[i] = np.sum(cds[: i + 1]) / d

    return uk


def compute_params_surface(points, size_u, size_v, centripetal=False):
    # Compute uk
    uk = np.zeros(size_u, dtype=float)

    # Compute for each curve on the v-direction
    uk_temp = []
    for v in range(size_v):
        pts_u = np.array([points[v + (size_v * u)] for u in range(size_u)])
        uk_temp.append(compute_params_curve(pts_u, centripetal))

    uk_temp = np.hstack(uk_temp)

    for u in range(size_u):
        knots_v = np.array([uk_temp[u + (size_u * v)] for v in range(size_v)])
        uk[u] = np.sum(knots_v) / size_v

    # Compute vl
    vl = np.zeros(size_v, dtype=float)

    # Compute for each curve on the u-direction
    vl_temp = []
    for u in range(size_u):
        pts_v = np.array([points[v + (size_v * u)] for v in range(size_v)])
        vl_temp.append(compute_params_curve(pts_v, centripetal))

    vl_temp = np.hstack(vl_temp)

    for v in range(size_v):
        knots_u = np.array([vl_temp[v + (size_v * u)] for u in range(size_u)])
        vl[v] = np.sum(knots_u) / size_u

    return uk, vl


def compute_knot_vector2(degree, num_dpts, num_cpts, params):
    # Start knot vector
    kv = np.zeros(degree + num_cpts + 1, dtype=float)
    kv[num_cpts:] = 1.0

    # Compute "d" value - Eqn 9.68
    d = 1.0 * num_dpts / (num_cpts - degree)
    # Find internal knots
    for j in range(1, num_cpts - degree):
        i = int(j * d)
        alpha = (j * d) - i
        temp_kv = ((1.0 - alpha) * params[i - 1]) + (alpha * params[i])
        kv[degree + j] = temp_kv

    return kv


def basis_function_one(degree, knot_vector, span, knot):
    # Special case at boundaries
    if (
        (span == 0 and knot == knot_vector[0])
        or (span == len(knot_vector) - degree - 2)
        and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    # Knot is outside of span range
    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = np.zeros(degree + span + 1, dtype=float)

    # Initialize the zeroth degree basis functions
    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    # Computing triangular table of basis functions
    for k in range(1, degree + 1):
        # Detecting zeros saves computations
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (
                knot_vector[span + k] - knot_vector[span]
            )

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            # Zero detection
            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    return N[0]


def doolittle(matrix_a):
    matrix_a = np.array(matrix_a, dtype=float)
    matrix_size = matrix_a.shape[0]

    # Initialize L and U matrices
    matrix_u = np.zeros([matrix_size, matrix_size], dtype=float)
    matrix_l = np.zeros([matrix_size, matrix_size], dtype=float)

    for i in range(0, matrix_size):
        for k in range(i, matrix_size):
            # Upper triangular (U) matrix
            matrix_u[i, k] = matrix_a[i, k] - matrix_l[i, :i].dot(matrix_u[:i, k])
            # Lower triangular (L) matrix
            if i == k:
                matrix_l[i, i] = 1.0
            else:
                matrix_l[k, i] = matrix_a[k, i] - matrix_l[k, :i].dot(matrix_u[:i, i])

                if matrix_u[i, i] == 0.0:
                    matrix_l[k, i] = 0.0
                else:
                    matrix_l[k, i] /= matrix_u[i, i]

    return matrix_l, matrix_u


def forward_substitution(matrix_l, matrix_b):
    matrix_size = matrix_b.shape[0]
    matrix_y = np.zeros(matrix_size, dtype=float)
    matrix_y[0] = matrix_b[0] / matrix_l[0, 0]

    for i in range(1, matrix_size):
        matrix_y[i] = matrix_b[i] - matrix_l[i, :i].dot(matrix_y[:i])
        matrix_y[i] /= matrix_l[i][i]
    return matrix_y


def backward_substitution(matrix_u, matrix_y):
    matrix_size = matrix_y.shape[0]

    matrix_x = np.zeros(matrix_size, dtype=float)

    matrix_x[matrix_size - 1] = (
        matrix_y[matrix_size - 1] / matrix_u[matrix_size - 1, matrix_size - 1]
    )
    for i in range(matrix_size - 2, -1, -1):
        matrix_x[i] = matrix_y[i] - matrix_u[i, i:matrix_size].dot(
            matrix_x[i:matrix_size]
        )
        matrix_x[i] /= matrix_u[i, i]
    return matrix_x


def approximate_surface(
    points: np.ndarray,
    size_u: int,
    size_v: int,
    degree_u: int,
    degree_v: int,
    use_centripetal: bool = False,
    ctrlpts_size_u: Union[int, None] = None,
    ctrlpts_size_v: Union[int, None] = None,
) -> Surface:
    if not geomdl_exist:
        print('[ERROR][fitting::approximate_surface]')
        print('\t geomdl not exist! please install it first!')
        return Surface()

    num_cpts_u = size_u - 1
    num_cpts_v = size_v - 1

    if ctrlpts_size_u is not None:
        num_cpts_u = ctrlpts_size_u
    if ctrlpts_size_v is not None:
        num_cpts_v = ctrlpts_size_v

    uk, vl = compute_params_surface(points, size_u, size_v, use_centripetal)

    kv_u = compute_knot_vector2(degree_u, size_u, num_cpts_u, uk)
    kv_v = compute_knot_vector2(degree_v, size_v, num_cpts_v, vl)

    matrix_nu = []
    for i in range(1, size_u - 1):
        m_temp = []
        for j in range(1, num_cpts_u - 1):
            m_temp.append(basis_function_one(degree_u, kv_u, j, uk[i]))
        matrix_nu.append(m_temp)
    matrix_nu = np.array(matrix_nu, dtype=float)

    matrix_ntu = matrix_nu.transpose(1, 0)

    matrix_ntnu = matrix_ntu.dot(matrix_nu)

    matrix_ntnul, matrix_ntnuu = doolittle(matrix_ntnu)

    # Fit u-direction
    ctrlpts_tmp = np.zeros([num_cpts_u * size_v, 3], dtype=float)
    for j in range(size_v):
        ctrlpts_tmp[j + (size_v * 0)] = points[j + (size_v * 0)]
        ctrlpts_tmp[j + (size_v * (num_cpts_u - 1))] = points[
            j + (size_v * (size_u - 1))
        ]

        pt0 = points[j + (size_v * 0)]  # Qzero
        ptm = points[j + (size_v * (size_u - 1))]  # Qm

        rku = []
        for i in range(1, size_u - 1):
            ptk = points[j + (size_v * i)]
            n0p = basis_function_one(degree_u, kv_u, 0, uk[i])
            nnp = basis_function_one(degree_u, kv_u, num_cpts_u - 1, uk[i])
            elem2 = n0p * pt0
            elem3 = nnp * ptm
            rku.append(ptk - elem2 - elem3)
        rku = np.array(rku)

        ru = np.zeros([num_cpts_u - 2, 3], dtype=float)
        for i in range(1, num_cpts_u - 1):
            ru_tmp = []
            for idx, pt in enumerate(rku):
                ru_tmp.append(basis_function_one(degree_u, kv_u, i, uk[idx + 1]) * pt)
            for d in range(3):
                for idx in range(len(ru_tmp)):
                    ru[i - 1][d] += ru_tmp[idx][d]
        ru = np.array(ru)

        for d in range(3):
            b = ru[:, d]
            y = forward_substitution(matrix_ntnul, b)
            x = backward_substitution(matrix_ntnuu, y)

            for i in range(1, num_cpts_u - 1):
                ctrlpts_tmp[j + (size_v * i), d] = x[i - 1]

    matrix_nv = []
    for i in range(1, size_v - 1):
        m_temp = []
        for j in range(1, num_cpts_v - 1):
            m_temp.append(basis_function_one(degree_v, kv_v, j, vl[i]))
        matrix_nv.append(m_temp)
    matrix_nv = np.array(matrix_nv, dtype=float)

    matrix_ntv = matrix_nv.transpose(1, 0)

    matrix_ntnv = matrix_ntv.dot(matrix_nv)

    matrix_ntnvl, matrix_ntnvu = doolittle(matrix_ntnv)

    ctrlpts = np.zeros([num_cpts_u * num_cpts_v, 3], dtype=float)
    for i in range(num_cpts_u):
        ctrlpts[0 + (num_cpts_v * i)] = ctrlpts_tmp[0 + (size_v * i)]
        ctrlpts[num_cpts_v - 1 + (num_cpts_v * i)] = ctrlpts_tmp[
            size_v - 1 + (size_v * i)
        ]

        pt0 = ctrlpts_tmp[0 + (size_v * i)]  # Qzero
        ptm = ctrlpts_tmp[size_v - 1 + (size_v * i)]  # Qm
        rkv = []
        for j in range(1, size_v - 1):
            ptk = ctrlpts_tmp[j + (size_v * i)]
            n0p = basis_function_one(degree_v, kv_v, 0, vl[j])
            nnp = basis_function_one(degree_v, kv_v, num_cpts_v - 1, vl[j])
            elem2 = n0p * pt0
            elem3 = nnp * ptm
            rkv.append(ptk - elem2 - elem3)
        rkv = np.array(rkv)

        rv = np.zeros([num_cpts_v - 2, 3], dtype=float)
        for j in range(1, num_cpts_v - 1):
            rv_tmp = []
            for idx, pt in enumerate(rkv):
                rv_tmp.append(basis_function_one(degree_v, kv_v, j, vl[idx + 1]) * pt)
            for d in range(3):
                for idx in range(len(rv_tmp)):
                    rv[j - 1][d] += rv_tmp[idx][d]

        for d in range(3):
            b = rv[:, d]
            y = forward_substitution(matrix_ntnvl, b)
            x = backward_substitution(matrix_ntnvu, y)
            for j in range(1, num_cpts_v - 1):
                ctrlpts[j + (num_cpts_v * i), d] = x[j - 1]

    surf = Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts_size_u = num_cpts_u
    surf.ctrlpts_size_v = num_cpts_v
    surf.ctrlpts = ctrlpts.tolist()
    surf.knotvector_u = kv_u
    surf.knotvector_v = kv_v

    return surf
