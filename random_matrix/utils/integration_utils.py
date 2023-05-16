"""Utility functions that assist with numerical integration"""

import copy
import inspect
from typing import Any, Callable

import numpy as np
import quadpy

from random_matrix.utils import function_utils
from random_matrix.utils.types import FloatLike, MathematicalFunction


def get_integration_domain_from_dict(
    function: MathematicalFunction, domain_dict: dict[str, list[FloatLike]]
) -> list[list[FloatLike]]:
    integration_variables = list(inspect.signature(function).parameters)
    new_domain = [domain_dict[var] for var in integration_variables]
    return new_domain


def basic_product_integral(
    function: MathematicalFunction,
    integration_domain: list[list[FloatLike]] | dict[str, list[FloatLike]],
    scheme: Any | None = None,
) -> FloatLike:
    """Integrate a function over a hypercube of dimension n, where n is the
    number of arguments.

    Parameters:
    -----------
    function:
        The function to be integrated.
    integration_domain:
        The integration domain as a list of lists. E.g. if the arguments
        come in the order (x,y) and the domain is 0 < x < 1, 1 < y < 2, then
        this should be [[0.0, 1.0], [1.0, 2.0]]. If it is given as a dict, this
        will first be converted.

    Returns:
    --------
    integral:
        The integral of function over the domain
    """

    # Convert domain to list if given as a dict
    if isinstance(integration_domain, dict):
        integration_domain = get_integration_domain_from_dict(
            function, integration_domain
        )

    # If the function has multiple arguments, replace its arguments
    # by a vector. This is to make quadpy happy.
    num_args = len(inspect.signature(function).parameters)
    if num_args > 1:
        function = function_utils.vectorize_arguments(function)

    dim = len(integration_domain)
    match dim:
        case 1:
            scheme = quadpy.c1.gauss_legendre(10) if scheme is None else scheme            
            integral = scheme.integrate(function, *integration_domain)
        case 2:
            scheme = (
                quadpy.c2.get_good_scheme(10) if scheme is None else scheme
            )
            integral = scheme.integrate(
                function, quadpy.c2.rectangle_points(*integration_domain)
            )
        case _:
            scheme = quadpy.cn.phillips(dim) if scheme is None else scheme
            integral = scheme.integrate(
                function,
                quadpy.cn.ncube_points(*integration_domain),
            )

    return integral  # type: ignore
