import inspect

import numpy as np
import scipy

from random_matrix.utils import function_utils
from random_matrix.utils.types import FloatLike


# vectorizing arguments
def test_vectorize_arguments() -> None:
    def function(x: float, y: float, z: float) -> float:
        return x * y * z

    assert len(inspect.signature(function).parameters) == 3
    vectorized = function_utils.vectorize_arguments(function)
    assert len(inspect.signature(vectorized).parameters) == 1


# adding functions
def test_add_functions() -> None:
    def f1(x: float) -> float:
        return 2 * x

    def f2(x: float) -> float:
        return x**2

    def f3(x: float) -> float:
        return np.sin(x)

    functions = [f1, f2, f3]
    func_sum = function_utils.add_functions(functions)

    def true_sum(x: float) -> float:
        return 2 * x + x**2 + np.sin(x)

    random_args = np.random.randn(1000)
    np.testing.assert_allclose(true_sum(random_args), func_sum(random_args))


# adding functions
def test_multiply_functions() -> None:
    def f1(x: float) -> float:
        return 2 * x

    def f2(x: float) -> float:
        return x**2

    def f3(x: float) -> float:
        return np.sin(x)

    functions = [f1, f2, f3]
    func_prod = function_utils.multiply_functions(functions)

    def true_prod(x: float) -> float:
        return 2 * x * x**2 * np.sin(x)

    random_args = np.random.randn(1000)
    np.testing.assert_allclose(true_prod(random_args), func_prod(random_args))


def test_fix_last_components() -> None:
    def f(k_inc: FloatLike, k_sca: FloatLike) -> FloatLike:
        return k_inc + k_sca

    f_reduced = function_utils.fix_last_components(f, (1, 1))

    for _ in range(10**3):
        k_inc = np.random.randn(3)
        k_inc = k_inc / np.linalg.norm(k_inc)
        k_out = np.random.randn(3)
        k_out = k_out / np.linalg.norm(k_out)
        k_inc[2] = np.abs(k_inc[2])
        k_out[2] = np.abs(k_out[2])
        kappa_inc = k_inc[0:2]
        kappa_out = k_out[0:2]
        np.testing.assert_allclose(
            f(k_inc, k_out), f_reduced(kappa_inc, kappa_out)
        )

    f_reduced = function_utils.fix_last_components(f, (1, -1))

    for _ in range(10**3):
        k_inc = np.random.randn(3)
        k_inc = k_inc / np.linalg.norm(k_inc)
        k_out = np.random.randn(3)
        k_out = k_out / np.linalg.norm(k_out)
        k_inc[2] = np.abs(k_inc[2])
        k_out[2] = -np.abs(k_out[2])
        kappa_inc = k_inc[0:2]
        kappa_out = k_out[0:2]
        np.testing.assert_allclose(
            f(k_inc, k_out), f_reduced(kappa_inc, kappa_out)
        )


def test_equate_arguments() -> None:
    def f(x, y, z):
        return (x + y) * z

    g = function_utils.equate_arguments(f)

    random_args = np.random.randn(10**3)
    for x in random_args:
        assert np.isclose(f(x, x, x), g(x))
