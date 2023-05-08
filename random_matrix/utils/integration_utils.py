"""Utility functions that assist with numerical integration"""

import numpy as np
import quadpy
import inspect
from typing import Callable, Any
from random_matrix.utils.types import DensityFunction, FloatLike
from enum import Enum


# def get_integration_variables(
#     function: Callable[[Any], Any], integration_domain: dict[str, FloatLike]
# ) -> tuple[list[str], list[str]]:
#     """Determine integration variables given a function and"""
#     domain_variables = list(integration_domain.keys())
#     function_variables = list(inspect.getfullargspec(function).args)
#     return domain_variables, function_variables


# def check_variable_compatability(
#     domain_variables: list[str], function_variables: list[str]
# ) -> None:
#     """Compare integration domain variables with those defined in the
#     function"""

#     domain_variables_set = set(domain_variables)
#     function_variables_set = set(function_variables)
#     if not domain_variables_set == function_variables_set:
#         raise ValueError(
#             f"Given integration domain variables "
#             f"{domain_variables} incompatible with function "
#             f"variables {function_variables}"
#         )


def integrate_density_function(
    density_function: DensityFunction,
    integration_domain: dict[str, list[FloatLike]],
    degree: int = 5,
) -> FloatLike:
    """Integrate a probability density function over its arguments. Used for
    density normalization checking."""

    domains = list(integration_domain.values())
    dim = len(domains)
    match dim:
        case 1:
            scheme = quadpy.c1.gauss_legendre(degree)
            integral = scheme.integrate(density_function, *domains)
        case _:
            scheme = quadpy.cn.stroud_cn_3_3(dim)
            integral = scheme.integrate(
                density_function,
                quadpy.cn.ncube_points(*domains),
            )
    return integral  # type: ignore
