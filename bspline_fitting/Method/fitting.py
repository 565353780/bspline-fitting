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
    kv = [0.0 for _ in range(degree + 1)]

    # Compute "d" value - Eqn 9.68
    d = float(num_dpts) / float(num_cpts - degree)
    # Find internal knots
    for j in range(1, num_cpts - degree):
        i = int(j * d)
        alpha = (j * d) - i
        temp_kv = ((1.0 - alpha) * params[i - 1]) + (alpha * params[i])
        kv.append(temp_kv)

    # End knot vector
    kv += [1.0 for _ in range(degree + 1)]

    return kv


def basis_function_one(degree, knot_vector, span, knot):
    """Computes the value of a basis function for a single parameter.

    Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis function, :math:`N_{i,p}`
    :rtype: float
    """
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

    N = [0.0 for _ in range(degree + span + 1)]

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


def matrix_transpose(m):
    """Transposes the input matrix.

    The input matrix :math:`m` is a 2-dimensional array.

    :param m: input matrix with dimensions :math:`(n \\times m)`
    :type m: list, tuple
    :return: transpose matrix with dimensions :math:`(m \\times n)`
    :rtype: list
    """
    num_cols = len(m)
    num_rows = len(m[0])
    m_t = []
    for i in range(num_rows):
        temp = []
        for j in range(num_cols):
            temp.append(m[j][i])
        m_t.append(temp)
    return m_t


def matrix_multiply(mat1, mat2):
    """Matrix multiplication (iterative algorithm).

    The running time of the iterative matrix multiplication algorithm is :math:`O(n^{3})`.

    :param mat1: 1st matrix with dimensions :math:`(n \\times p)`
    :type mat1: list, tuple
    :param mat2: 2nd matrix with dimensions :math:`(p \\times m)`
    :type mat2: list, tuple
    :return: resultant matrix with dimensions :math:`(n \\times m)`
    :rtype: list
    """
    n = len(mat1)
    p1 = len(mat1[0])
    p2 = len(mat2)
    assert p1 == p2, "Column - row size mismatch"

    try:
        # Matrix - matrix multiplication
        m = len(mat2[0])
        mat3 = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    mat3[i][j] += float(mat1[i][k] * mat2[k][j])
    except TypeError:
        # Matrix - vector multiplication
        mat3 = [0.0 for _ in range(n)]
        for i in range(n):
            for k in range(p2):
                mat3[i] += float(mat1[i][k] * mat2[k])
    return mat3


def doolittle(matrix_a):
    """Doolittle's Method for LU-factorization.

    :param matrix_a: Input matrix (must be a square matrix)
    :type matrix_a: list, tuple
    :return: a tuple containing matrices (L,U)
    :rtype: tuple
    """
    # Initialize L and U matrices
    matrix_u = [[0.0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]
    matrix_l = [[0.0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]

    # Doolittle Method
    for i in range(0, len(matrix_a)):
        for k in range(i, len(matrix_a)):
            # Upper triangular (U) matrix
            matrix_u[i][k] = float(
                matrix_a[i][k]
                - sum([matrix_l[i][j] * matrix_u[j][k] for j in range(0, i)])
            )
            # Lower triangular (L) matrix
            if i == k:
                matrix_l[i][i] = 1.0
            else:
                matrix_l[k][i] = float(
                    matrix_a[k][i]
                    - sum([matrix_l[k][j] * matrix_u[j][i] for j in range(0, i)])
                )
                # Handle zero division error
                try:
                    matrix_l[k][i] /= float(matrix_u[i][i])
                except ZeroDivisionError:
                    matrix_l[k][i] = 0.0

    return matrix_l, matrix_u


def lu_decomposition(matrix_a):
    """LU-Factorization method using Doolittle's Method for solution of linear systems.

    Decomposes the matrix :math:`A` such that :math:`A = LU`.

    The input matrix is represented by a list or a tuple. The input matrix is **2-dimensional**, i.e. list of lists of
    integers and/or floats.

    :param matrix_a: Input matrix (must be a square matrix)
    :type matrix_a: list, tuple
    :return: a tuple containing matrices L and U
    :rtype: tuple
    """
    # Check if the 2-dimensional input matrix is a square matrix
    q = len(matrix_a)
    for idx, m_a in enumerate(matrix_a):
        if len(m_a) != q:
            raise ValueError(
                "The input must be a square matrix. "
                + "Row "
                + str(idx + 1)
                + " has a size of "
                + str(len(m_a))
                + "."
            )

    # Return L and U matrices
    return doolittle(matrix_a)


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
    # Compute Nu transpose
    matrix_ntu = matrix_transpose(matrix_nu)
    # Compute NTNu matrix
    matrix_ntnu = matrix_multiply(matrix_ntu, matrix_nu)
    # Compute LU-decomposition of NTNu matrix
    matrix_ntnul, matrix_ntnuu = lu_decomposition(matrix_ntnu)

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
    # Compute Nv transpose
    matrix_ntv = matrix_transpose(matrix_nv)
    # Compute NTNv matrix
    matrix_ntnv = matrix_multiply(matrix_ntv, matrix_nv)
    # Compute LU-decomposition of NTNv matrix
    matrix_ntnvl, matrix_ntnvu = lu_decomposition(matrix_ntnv)

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
