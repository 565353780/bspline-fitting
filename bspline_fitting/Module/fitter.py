import numpy as np
import matplotlib.pyplot as plt

from bspline_fitting.Method.fitting import approximate_surface

class Fitter(object):
    def __init__(self) -> None:
        return

    def fit(
        self, points: np.ndarray, size_u: int, size_v: int, degree_u: int, degree_v: int
    ) -> bool:
        surf = approximate_surface(points, size_u, size_v, degree_u, degree_v)
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=3, ctrlpts_size_v=4)

        evalpts = np.array(surf.evalpts)
        pts = np.array(points)
        ax = plt.axes(projection='3d')
        ax.scatter(evalpts[:, 0], evalpts[:, 1], evalpts[:, 2])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red")
        plt.show()
        return True
