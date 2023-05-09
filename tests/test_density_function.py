import numpy as np
import pytest

from random_matrix.statistics.density_function import (
    DeltaDensityFactor,
    DensityFunction,
    DensityFunctionTerm,
    RegularDensityFactor,
)
from random_matrix.utils.types import FloatLike, MathematicalFunction


# Regular factor
def test_regular_density_factor() -> None:
    def function(x: FloatLike, m: FloatLike) -> FloatLike:
        return 2 * x * 2 * m

    domain = {"x": [0.0, 1.0], "m": [0.0, 1.0]}

    density = RegularDensityFactor(function, domain)
    assert density.density == function
    assert density.domain == domain
    assert np.isclose(density.integral, 1.0)


# Dirac factor
def test_dirac_density_factor() -> None:
    delta_functions = {"x": 1.0, "m": 1.2}
    density = DeltaDensityFactor(delta_functions)

    assert density.density == delta_functions
    assert np.isclose(density.integral, 1.0)


# Empty term
def test_empty_term() -> None:
    with pytest.raises(Exception, match="You have both as None"):
        empty_term = DensityFunctionTerm()


# Mixed term
def test_mixed_term() -> None:
    def function(x: float, y: float, z: float) -> float:
        return 2 * x * 2 * y * 2 * z

    domain = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}

    deltas = {"a": 0.1, "b": 0.2}
    delta_factor = 0.3

    reg = RegularDensityFactor(function, domain)
    delt = DeltaDensityFactor(deltas, delta_factor)
    term = DensityFunctionTerm(reg, delt)

    assert term.regular == reg
    assert term.delta == delt
    assert np.isclose(term.integral, 0.3)


# non-normalized density from function
def test_non_normalized_regular() -> None:
    with pytest.raises(Exception, match="is not normalized"):

        def function(x):
            return x**2

        domain = {"x": [0.0, 1.0]}
        density = DensityFunction.from_function(function, domain)


# from deltas
def test_from_deltas() -> None:
    deltas = {"x": 0.1, "y": 0.2}
    density = DensityFunction.from_deltas(deltas)
    print(density.terms[0])
    assert np.isclose(density.integral, 1.0)
    assert len(density.terms) == 1
    assert density.terms[0].delta.density == deltas
