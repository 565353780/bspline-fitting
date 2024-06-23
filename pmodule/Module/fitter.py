import numpy as np
from typing import Union

from bspline_fitting.Method.fitting import approximate_surface


class Fitter(object):
    def __init__(self) -> None:
        return

    def fit(
        self,
        points: np.ndarray,
        size_u: int,
        size_v: int,
        degree_u: int,
        degree_v: int,
        use_centripetal: bool = False,
        ctrlpts_size_u: Union[int, None] = None,
        ctrlpts_size_v: Union[int, None] = None,
    ):
        return approximate_surface(
            points,
            size_u,
            size_v,
            degree_u,
            degree_v,
            use_centripetal,
            ctrlpts_size_u,
            ctrlpts_size_v,
        )
