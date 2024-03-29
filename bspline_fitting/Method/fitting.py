import numpy as np

from geomdl import BSpline


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
    """Forward substitution method for the solution of linear systems.

    Solves the equation :math:`Ly = b` using forward substitution method
    where :math:`L` is a lower triangular matrix and :math:`b` is a column matrix.

    :param matrix_l: L, lower triangular matrix
    :type matrix_l: list, tuple
    :param matrix_b: b, column matrix
    :type matrix_b: list, tuple
    :return: y, column matrix
    :rtype: list
    """
    q = len(matrix_b)
    matrix_y = [0.0 for _ in range(q)]
    matrix_y[0] = float(matrix_b[0]) / float(matrix_l[0][0])
    for i in range(1, q):
        matrix_y[i] = float(matrix_b[i]) - sum(
            [matrix_l[i][j] * matrix_y[j] for j in range(0, i)]
        )
        matrix_y[i] /= float(matrix_l[i][i])
    return matrix_y


def backward_substitution(matrix_u, matrix_y):
    """Backward substitution method for the solution of linear systems.

    Solves the equation :math:`Ux = y` using backward substitution method
    where :math:`U` is a upper triangular matrix and :math:`y` is a column matrix.

    :param matrix_u: U, upper triangular matrix
    :type matrix_u: list, tuple
    :param matrix_y: y, column matrix
    :type matrix_y: list, tuple
    :return: x, column matrix
    :rtype: list
    """
    q = len(matrix_y)
    matrix_x = [0.0 for _ in range(q)]
    matrix_x[q - 1] = float(matrix_y[q - 1]) / float(matrix_u[q - 1][q - 1])
    for i in range(q - 2, -1, -1):
        matrix_x[i] = float(matrix_y[i]) - sum(
            [matrix_u[i][j] * matrix_x[j] for j in range(i, q)]
        )
        matrix_x[i] /= float(matrix_u[i][i])
    return matrix_x


def approximate_surface(points, size_u, size_v, degree_u, degree_v, **kwargs):
    # Keyword arguments
    use_centripetal = kwargs.get("centripetal", False)
    num_cpts_u = kwargs.get(
        "ctrlpts_size_u", size_u - 1
    )  # number of datapts, r + 1 > number of ctrlpts, n + 1
    num_cpts_v = kwargs.get(
        "ctrlpts_size_v", size_v - 1
    )  # number of datapts, s + 1 > number of ctrlpts, m + 1

    # Dimension
    dim = len(points[0])

    # Get uk and vl
    uk, vl = compute_params_surface(points, size_u, size_v, use_centripetal)

    # Compute knot vectors
    kv_u = compute_knot_vector2(degree_u, size_u, num_cpts_u, uk)
    kv_v = compute_knot_vector2(degree_v, size_v, num_cpts_v, vl)

    # Construct matrix Nu
    matrix_nu = []
    for i in range(1, size_u - 1):
        m_temp = []
        for j in range(1, num_cpts_u - 1):
            m_temp.append(basis_function_one(degree_u, kv_u, j, uk[i]))
        matrix_nu.append(m_temp)
    matrix_nu = np.array(matrix_nu, dtype=float)
    # Compute Nu transpose
    matrix_ntu = matrix_nu.transpose(1, 0)
    # Compute NTNu matrix
    matrix_ntnu = matrix_ntu.dot(matrix_nu)
    # Compute LU-decomposition of NTNu matrix
    matrix_ntnul, matrix_ntnuu = doolittle(matrix_ntnu)

    # Fit u-direction
    ctrlpts_tmp = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u * size_v)]
    for j in range(size_v):
        ctrlpts_tmp[j + (size_v * 0)] = list(points[j + (size_v * 0)])
        ctrlpts_tmp[j + (size_v * (num_cpts_u - 1))] = list(
            points[j + (size_v * (size_u - 1))]
        )
        # Compute Rku - Eqn. 9.63
        pt0 = points[j + (size_v * 0)]  # Qzero
        ptm = points[j + (size_v * (size_u - 1))]  # Qm
        rku = []
        for i in range(1, size_u - 1):
            ptk = points[j + (size_v * i)]
            n0p = basis_function_one(degree_u, kv_u, 0, uk[i])
            nnp = basis_function_one(degree_u, kv_u, num_cpts_u - 1, uk[i])
            elem2 = [c * n0p for c in pt0]
            elem3 = [c * nnp for c in ptm]
            rku.append([a - b - c for a, b, c in zip(ptk, elem2, elem3)])
        # Compute Ru - Eqn. 9.67
        ru = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u - 2)]
        for i in range(1, num_cpts_u - 1):
            ru_tmp = []
            for idx, pt in enumerate(rku):
                ru_tmp.append(
                    [p * basis_function_one(degree_u, kv_u, i, uk[idx + 1]) for p in pt]
                )
            for d in range(dim):
                for idx in range(len(ru_tmp)):
                    ru[i - 1][d] += ru_tmp[idx][d]
        # Get intermediate control points
        for d in range(dim):
            b = [pt[d] for pt in ru]
            y = forward_substitution(matrix_ntnul, b)
            x = backward_substitution(matrix_ntnuu, y)
            for i in range(1, num_cpts_u - 1):
                ctrlpts_tmp[j + (size_v * i)][d] = x[i - 1]

    # Construct matrix Nv
    matrix_nv = []
    for i in range(1, size_v - 1):
        m_temp = []
        for j in range(1, num_cpts_v - 1):
            m_temp.append(basis_function_one(degree_v, kv_v, j, vl[i]))
        matrix_nv.append(m_temp)
    matrix_nv = np.array(matrix_nv, dtype=float)
    # Compute Nv transpose
    matrix_ntv = matrix_nv.transpose(1, 0)
    # Compute NTNv matrix
    matrix_ntnv = matrix_ntv.dot(matrix_nv)
    # Compute LU-decomposition of NTNv matrix
    matrix_ntnvl, matrix_ntnvu = doolittle(matrix_ntnv)

    # Fit v-direction
    ctrlpts = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u * num_cpts_v)]
    for i in range(num_cpts_u):
        ctrlpts[0 + (num_cpts_v * i)] = list(ctrlpts_tmp[0 + (size_v * i)])
        ctrlpts[num_cpts_v - 1 + (num_cpts_v * i)] = list(
            ctrlpts_tmp[size_v - 1 + (size_v * i)]
        )
        # Compute Rkv - Eqs. 9.63
        pt0 = ctrlpts_tmp[0 + (size_v * i)]  # Qzero
        ptm = ctrlpts_tmp[size_v - 1 + (size_v * i)]  # Qm
        rkv = []
        for j in range(1, size_v - 1):
            ptk = ctrlpts_tmp[j + (size_v * i)]
            n0p = basis_function_one(degree_v, kv_v, 0, vl[j])
            nnp = basis_function_one(degree_v, kv_v, num_cpts_v - 1, vl[j])
            elem2 = [c * n0p for c in pt0]
            elem3 = [c * nnp for c in ptm]
            rkv.append([a - b - c for a, b, c in zip(ptk, elem2, elem3)])
        # Compute Rv - Eqn. 9.67
        rv = [[0.0 for _ in range(dim)] for _ in range(num_cpts_v - 2)]
        for j in range(1, num_cpts_v - 1):
            rv_tmp = []
            for idx, pt in enumerate(rkv):
                rv_tmp.append(
                    [p * basis_function_one(degree_v, kv_v, j, vl[idx + 1]) for p in pt]
                )
            for d in range(dim):
                for idx in range(len(rv_tmp)):
                    rv[j - 1][d] += rv_tmp[idx][d]
        # Get intermediate control points
        for d in range(dim):
            b = [pt[d] for pt in rv]
            y = forward_substitution(matrix_ntnvl, b)
            x = backward_substitution(matrix_ntnvu, y)
            for j in range(1, num_cpts_v - 1):
                ctrlpts[j + (num_cpts_v * i)][d] = x[j - 1]

    # Generate B-spline surface
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts_size_u = num_cpts_u
    surf.ctrlpts_size_v = num_cpts_v
    surf.ctrlpts = ctrlpts
    surf.knotvector_u = kv_u
    surf.knotvector_v = kv_v

    return surf
