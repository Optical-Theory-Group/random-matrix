"""Module for integrating functions over regions defined by Mode objects"""

from typing import Callable, Any
import numpy as np
import numpy.typing as npt
import quadpy
import scipy
import itertools
import numba

def integrate_simplex(
    function: Callable[[..., Any], Any],
    vertices: npt.NDArray[np.complex128],
    degree: int = 5,
) -> np.complex128:
    num_vertices, num_dimensions = np.shape(vertices)
    scheme = quadpy.tn.grundmann_moeller(num_dimensions, degree)
    return scheme.integrate(function, vertices)


def f(x):
    return x[0] ** 2 + x[1] ** 2


triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


test = integrate_simplex(f, triangle)
print(test)
