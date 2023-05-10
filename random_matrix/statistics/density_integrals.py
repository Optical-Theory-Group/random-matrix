"""Module for computing statistical quantities associated with functions."""

import copy
import functools
import inspect

import numpy as np
import numpy.typing as npt
import quadpy

from random_matrix.statistics.density_function import (
    DensityFunction,
    DeltaDensityFactor,
)
from random_matrix.utils.types import MathematicalFunction, FloatLike
from random_matrix.utils import integration_utils, function_utils


def integrate_by_delta(
    function: MathematicalFunction, delta_density_factor: DeltaDensityFactor
) -> MathematicalFunction:
    """Integrate a function by a product of delta functions, yielding a new
    function.

    Note: the returned function has only key work arguments.
    """

    delta_function = delta_density_factor.density

    def integrated_function(**kwargs: FloatLike) -> FloatLike:
        new_kwargs = {**kwargs, **delta_function}
        return function(**new_kwargs)

    # Update signature of integrated_function to contain new variables
    integrated_vars = set(delta_function.keys())
    total_vars = function_utils.get_function_variables(function)
    remaining_vars = total_vars - integrated_vars
    if len(remaining_vars) > 0:
        new_signature = inspect.Signature(
            [
                inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for var in remaining_vars
            ]
        )
        integrated_function.__signature__ = new_signature

    return integrated_function


def integrate_by_density(
    function: MathematicalFunction, density: DensityFunction
) -> MathematicalFunction:
    """Integrate a function by a probability density function, thus yielding
    a new function."""

    # Check that density_variables is a proper subset of function_variables
    function_variables = integration_utils.get_function_variables(function)
    density_variables = density.variables
    if not density_variables <= function_variables:
        raise ValueError(
            f"The density function variables: {density_variables} must be a "
            f"proper subset of the function variables {function_variables}."
        )

    # Loop over terms. Results will be added
    partial_results = []
    for term in density.terms:
        integrated_function = copy.copy(function)

        # Integrate over delta functions
        delta_distribution = term.delta
        if delta_distribution is not None:
            integrated_function = integrate_by_delta(
                integrated_function, delta_distribution
            )

        # Integrate over regular density function factor
        regular_distribution = term.regular
        if regular_distribution is not None:
            integrated_function = integrate_by_regular(
                integrated_function, regular_distribution
            )

        partial_results.append(integrated_function)

    # Add together results
    result = function_utils.add_functions(partial_results)
    return result


d = DeltaDensityFactor({"a": 1.0, "b": 2.0})


def f(a, b, c):
    return a + b + c


g = integrate_by_delta(f, d)
