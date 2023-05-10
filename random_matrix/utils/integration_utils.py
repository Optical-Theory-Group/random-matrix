"""Utility functions that assist with numerical integration"""

import inspect
from typing import Any, Callable
import copy

import numpy as np
import quadpy

from random_matrix.utils.types import FloatLike, MathematicalFunction
from random_matrix.utils import function_utils


def product_integral(
    function: MathematicalFunction,
    integration_domain: dict[str, list[FloatLike]],
    degree: int = 5,
) -> FloatLike:
    """Integrate a probability density function over its arguments. Used for
    density normalization checking."""

    # If the function has multiple arguments, replace its arguments
    # by a vector.
    num_args = len(inspect.signature(function).parameters)
    if num_args > 1:
        function = function_utils.vectorize_arguments(function)

    print(function)
    domains = list(integration_domain.values())
    dim = len(domains)
    match dim:
        case 1:
            scheme = quadpy.c1.gauss_legendre(degree)
            integral = scheme.integrate(function, *domains)
        case _:
            scheme = quadpy.cn.stroud_cn_3_3(dim)
            integral = scheme.integrate(
                function,
                quadpy.cn.ncube_points(*domains),
            )
    return integral  # type: ignore
