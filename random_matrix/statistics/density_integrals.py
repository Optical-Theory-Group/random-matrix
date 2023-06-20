"""Module for computing statistical quantities associated with functions."""

import copy
import functools
import inspect
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import quadpy
import numba

from random_matrix.statistics import density_function
from random_matrix.utils import function_utils, integration_utils
from random_matrix.utils.types import (
    FloatLike,
    MathematicalFunction,
)


def integrate_by_delta_density_function(
    function: MathematicalFunction,
    delta_density_function: dict[str, FloatLike],
) -> MathematicalFunction:
    """Integrates a function by a product of delta functions, fixing the
    values of the arguments of the original function to those dictated
    by the delta functions.

    Parameters:
    -----------
    function:
        The function to be integrated.
    delta_density_function:
        Delta functions in the integrand, represented by a dictionary of
        variable-value pairs.

    Returns:
    --------
    integrated_function:
        The result of the integration.
    """

    function_variables = function_utils.get_function_variables(function)
    integration_variables = set(delta_density_function.keys())
    remaining_variables = [
        var for var in function_variables if var not in integration_variables
    ]
    num_remaining_variables = len(remaining_variables)

    # args here are the remaning variables after integration
    def integrated_function(*args: FloatLike) -> FloatLike:
        if len(args) != num_remaining_variables:
            raise ValueError(
                f"Please give {num_remaining_variables} positional arguments "
                f"for variables {remaining_variables}"
            )
        new_args = []
        old_args = iter(args)
        for arg in function_variables:
            if arg in integration_variables:
                new_args.append(delta_density_function.get(arg))
            else:
                new_args.append(next(old_args))
        return function(*new_args)

    # Update signature of integrated_function to contain new variables
    if len(remaining_variables) > 0:
        integrated_function.__signature__ = function_utils.get_new_signature(
            remaining_variables
        )
    return integrated_function


def integrate_by_delta_density_factor(
    function: MathematicalFunction,
    delta_density_factor: density_function.DeltaDensityFactor,
) -> MathematicalFunction:
    """Equivalent to "integrate by delta functions", but for a
    DeltaDensityFactor object as an input.

    Note that the constant factor is also enforced.
    """

    delta_functions = delta_density_factor.density_function
    constant_factor = delta_density_factor.const_factor
    integrated_function = integrate_by_delta_density_function(
        function, delta_functions
    )
    return function_utils.multiply_function_by_constant(
        integrated_function, constant_factor
    )


def integrate_by_regular_density_function(
    function: MathematicalFunction,
    regular_density_function: MathematicalFunction,
    integration_domain: dict[str, list[FloatLike]],
    scheme: Any | None = None,
) -> MathematicalFunction:
    """Integrate a function by a regular (non-Dirac) probability density
    function, yielding its expected value.

    Parameters:
    -----------
    function:
        The function to be integrated.
    regular_density_function:
        The pdf against which the function is integrated.
    integration_domain:
        The integration domain as a dictionary
    scheme:
        Cubature scheme used in integration

    Returns:
    --------
    integrated_function:
        The integral of function * density over the domain.
    """

    function_variables = function_utils.get_function_variables(function)
    integration_variables = function_utils.get_function_variables(
        regular_density_function
    )
    remaining_variables = [
        var for var in function_variables if var not in integration_variables
    ]
    num_remaining_variables = len(remaining_variables)

    # For every choice of the non-integration variables, we return a function
    # that gives the integrand, which is a function of the integration
    # variables.
    def get_integrand(*remaining_args: FloatLike) -> MathematicalFunction:
        def integrand(*integration_args: FloatLike) -> FloatLike:
            # new_args is what needs to be passed to the function being
            # integrated against
            remaining_args_iter = iter(remaining_args)
            integration_args_iter = iter(integration_args)
            new_args = []

            for arg in function_variables:
                if arg in integration_variables:
                    new_args.append(next(integration_args_iter))
                else:
                    new_args.append(next(remaining_args_iter))
            return function(*new_args) * regular_density_function(
                *integration_args
            )

        # Give the inner function a signature containing its integration
        # variables
        integrand.__signature__ = function_utils.get_new_signature(
            integration_variables
        )
        return integrand  # type: ignore

    def integrated_function(*args: FloatLike) -> FloatLike:
        if len(args) != num_remaining_variables:
            raise ValueError(
                f"Please give {num_remaining_variables} positional arguments "
                f"for variables {remaining_variables}"
            )
        local_integrand = get_integrand(*args)
        return integration_utils.basic_product_integral(
            local_integrand, integration_domain, scheme
        )

    # Update signature of integrated_function to contain new variables
    if len(remaining_variables) > 0:
        integrated_function.__signature__ = function_utils.get_new_signature(
            remaining_variables
        )

    return integrated_function  # type: ignore


def integrate_by_regular_density_factor(
    function: MathematicalFunction,
    regular_density_factor: density_function.RegularDensityFactor,
    scheme: Any | None = None,
) -> MathematicalFunction:
    density = regular_density_factor.density_function
    domain = regular_density_factor.domain
    return integrate_by_regular_density_function(
        function, density, domain, scheme
    )


def integrate_by_density(
    function: MathematicalFunction,
    density: density_function.DensityFunction,
    scheme: Any | None = None,
) -> MathematicalFunction:
    """Integrate a function by a general probability density function,
    including both regular functions and Dirac delta functions.

    Parameters:
    -----------
    function:
        The function to be integrated.
    density:
        The pdf against which the function is integrated, represented as a
        DensityFunction object.
    scheme:
        Cubature scheme used in integration

    Returns:
    --------
    integrated_function:
        The integral of function * density.
    """

    # Check that density_variables is a proper subset of function_variables
    function_variables = set(function_utils.get_function_variables(function))
    density_variables = set(density.variables)
    if not density_variables <= function_variables:
        raise TypeError(
            f"The density function variables: {density_variables} must be a "
            f"proper subset of the function variables {function_variables}."
        )

    # Loop over terms. Results will be added
    partial_results = []
    for term in density.terms:
        integrated_function = copy.copy(function)

        # Integrate over delta functions
        delta_distribution = term.delta_factor
        if delta_distribution is not None:
            integrated_function = integrate_by_delta_density_factor(
                integrated_function, delta_distribution
            )

        # Integrate over regular density function factor
        regular_distribution = term.regular_factor
        if regular_distribution is not None:
            integrated_function = integrate_by_regular_density_factor(
                integrated_function, regular_distribution, scheme
            )

        partial_results.append(integrated_function)

    # Add together results
    result = function_utils.add_functions(partial_results)
    return result
