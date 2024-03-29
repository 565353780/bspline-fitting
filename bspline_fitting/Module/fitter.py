import numpy as np
from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis


class Fitter(object):
    def __init__(self) -> None:
        return

    def fit(
        self, points: np.ndarray, size_u: int, size_v: int, degree_u: int, degree_v: int
    ) -> bool:
        surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=3, ctrlpts_size_v=4)

        # Extract curves from the approximated surface
        surf_curves = construct.extract_curves(surf)
        plot_extras = [
            dict(points=surf_curves["u"][0].evalpts, name="u", color="cyan", size=5),
            dict(points=surf_curves["v"][0].evalpts, name="v", color="magenta", size=5),
        ]

        # Plot the interpolated surface
        surf.delta = 0.05
        surf.vis = vis.VisSurface()
        surf.render(extras=plot_extras)

        # # Visualize data and evaluated points together
        # import numpy as np
        # import matplotlib.pyplot as plt
        # evalpts = np.array(surf.evalpts)
        # pts = np.array(points)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter(evalpts[:, 0], evalpts[:, 1], evalpts[:, 2])
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red")
        # plt.show()
        return True
