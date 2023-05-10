import inspect

import numpy as np
import scipy

from random_matrix.utils import integration_utils
from random_matrix.utils.types import FloatLike


# 1D integral
def test_product_integral_1d() -> None:
    def function(x: float) -> float:
        return 2 * x

    domain = {"x": [0.0, 1.0]}

    integral = integration_utils.product_integral(function, domain)
    assert np.isclose(integral, 1.0)


# 2D integral
def test_product_integral_2d() -> None:
    def function(x: float, y: float) -> float:
        return 4 * x * y

    domain = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    integral = integration_utils.product_integral(function, domain)
    assert np.isclose(integral, 1.0)


# High dimensional integral
def test_product_integral_highd() -> None:
    def function(
        x: float, y: float, z: float, a: float, b: float, c: float
    ) -> float:
        return 2 * x * 2 * y * 2 * z * 2 * a * 2 * b * 2 * c

    domain = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "z": [0.0, 1.0],
        "a": [0.0, 1.0],
        "b": [0.0, 1.0],
        "c": [0.0, 1.0],
    }

    integral = integration_utils.product_integral(function, domain)
    assert np.isclose(integral, 1.0)


# More complex integral
def test_error_function_integral() -> None:
    def gaussian(x: float) -> float:
        return np.exp(-(x**2))

    domain = {"x": [0.0, 1.0]}
    integral = integration_utils.product_integral(gaussian, domain)
    actual = np.sqrt(np.pi) / 2.0 * scipy.special.erf(1.0)
    assert np.isclose(integral, actual)

