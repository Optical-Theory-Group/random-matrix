"""Utility functions that assist with numerical integration"""

import inspect
from typing import Any, Callable
import math
import numpy as np
import quadpy
import cupy as cp
import scipy
from random_matrix.utils import function_utils
from random_matrix.utils.types import Numeric, MathematicalFunction


def get_integration_domain_from_dict(
    function: MathematicalFunction, domain_dict: dict[str, list[Numeric]]
) -> list[list[Numeric]]:
    integration_variables = list(inspect.signature(function).parameters)
    new_domain = [domain_dict[var] for var in integration_variables]
    return new_domain


def basic_product_integral(
    function: MathematicalFunction,
    integration_domain: list[list[Numeric]] | dict[str, list[Numeric]],
    scheme: Any | None = None,
) -> Numeric:
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

    scheme:
        The cubature scheme. If not provided, defaults will be used.

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
        function = function_utils.vectorize_functions_args(function)

    dim = len(integration_domain)
    match dim:
        case 1:
            scheme = quadpy.c1.gauss_legendre(10) if scheme is None else scheme
            integral = scheme.integrate(function, *integration_domain)
        case 2:
            scheme = quadpy.c2.get_good_scheme(10) if scheme is None else scheme
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


def basic_triangle_integral(
    function: MathematicalFunction,
    integration_domain: Numeric,
    scheme: Any | None = None,
) -> Numeric:
    """Integrates a function of 2 variables over a triangle (or collection
    of triangles)

    Parameters:
    -----------
    function:
        The function to be integrated.
    integration_domain:
        The integration domain as a list of lists. E.g. if the arguments
        come in the order (x,y) and the domain is 0 < x < 1, 1 < y < 2, then
        this should be [[0.0, 1.0], [1.0, 2.0]]. If it is given as a dict, this
        will first be converted.
    scheme:
        The cubature scheme. If not provided, defaults will be used.

    Returns:
    --------
    integral:
        The integral of function over the domain
    """

    # If the function has multiple arguments, replace its arguments
    # by a vector. This is to make quadpy happy.
    num_args = len(inspect.signature(function).parameters)
    if num_args > 1:
        function = function_utils.vectorize_arguments(function)

    if scheme is None:
        scheme = quadpy.t2.get_good_scheme(3)

    # Reshape domain stack if necessary
    # again to make quadpy happy
    if np.ndim(integration_domain == 3) and np.shape(integration_domain)[1:] == (3, 2):
        integration_domain = np.transpose(integration_domain, (1, 0, 2))

    integral = scheme.integrate(function, integration_domain)

    return integral


def basic_simplex_integral(
    function: MathematicalFunction,
    integration_domain: Numeric,
    scheme: Any | None = None,
    dim: int = 6,
) -> Numeric:
    """Integrates a function of 2 variables over a triangle (or collection
    of triangles)

    Parameters:
    -----------
    function:
        The function to be integrated.
    integration_domain:
        The integration domain as a list of lists. E.g. if the arguments
        come in the order (x,y) and the domain is 0 < x < 1, 1 < y < 2, then
        this should be [[0.0, 1.0], [1.0, 2.0]]. If it is given as a dict, this
        will first be converted.
    scheme:
        The cubature scheme. If not provided, defaults will be used.

    Returns:
    --------
    integral:
        The integral of function over the domain
    """

    # If the function has multiple arguments, replace its arguments
    # by a vector. This is to make quadpy happy.
    num_args = len(inspect.signature(function).parameters)
    if num_args > 1:
        function = function_utils.vectorize_arguments(function)

    if scheme is None:
        scheme = quadpy.tn.grundmann_moeller(dim, 3)

    # Reshape domain stack if necessary
    # again to make quadpy happy
    if np.ndim(integration_domain == 3) and np.shape(integration_domain)[1:] == (7, 6):
        integration_domain = np.transpose(integration_domain, (1, 0, 2))

    integral = scheme.integrate(function, integration_domain)
    # if np.any(np.isinf(integral)):
    #     print("Nan after integration")
    # else:
    #     print("All good!")

    return integral


def test_func(x):
    return x[:, 0] ** 2 + x[:, 1] ** 2


def hull_surface_integral(
    function: MathematicalFunction,
    hull: scipy.spatial.ConvexHull,
    scheme: Any | None = None,
    use_gpu: bool = False,
) -> np.ndarray | cp.ndarray:
    """Compute the integral of a function over the surface of a convex hull. It
    is assumed that the function returns a scalar output, i.e. maps R^n -> R.

    function: The function to be integrated. This function must be vectorized
    to accept arguments of shape

    N x n

    where n-1 is the number of dimensions of the simplical facets of the
    surface (alternatively, n is the number of dimensions of the ambient space
    in which the surface lies. For example, for the surface of a sphere, n=3)

    hull: The convex hull object that defines the surface.
    scheme: Integration scheme to be used.
    use_gpu: If true, use cupy instead of numpy
    """
    # Pick appropriate array module
    xp = cp if use_gpu else np

    # Pick integration scheme
    # For future developers: one can use Cayley-Menger determinants for d > 2
    num_dimensions = hull.points.shape[1] - 1
    if num_dimensions > 2:
        raise NotImplementedError(
            "Integration of surfaces in d > 2 dimensions"
            "not currently supported."
        )

    if scheme is None:
        # The second parameter to quadpy's scheme method here is about
        # accuracy of the scheme, not the number of spatial dimensions.
        scheme = quadpy.tn.grundmann_moeller(num_dimensions, 3)

    barycentric_weights = xp.asarray(scheme.points)
    weights = xp.asarray(scheme.weights)

    # Generate integration points
    num_simplices = len(hull.simplices)
    num_weights = len(weights)
    simplical_points = xp.asarray(hull.points)[
        xp.asarray(hull.simplices)
    ].transpose(0, 2, 1)
    integration_points = simplical_points @ barycentric_weights
    reshaped_points = integration_points.transpose(0, 2, 1).reshape(
        -1, num_dimensions + 1
    )

    # Find simplex volumes using cross product
    v1s = simplical_points[:, :, 1] - simplical_points[:, :, 0]
    v2s = simplical_points[:, :, 2] - simplical_points[:, :, 0]
    cross_products = xp.cross(v1s, v2s)
    areas = 0.5 * xp.sqrt(xp.sum(cross_products**2, axis=1))

    # Output
    function_output = function(reshaped_points)
    weighted_output = (
        function_output
        * xp.tile(weights, num_simplices)
        * xp.repeat(areas, num_weights)
    )
    integral = xp.sum(weighted_output)
    return integral


def simplex_integral(
    function: Callable,
    simplices: Numeric,
    scheme: Any | None = None,
    use_gpu: bool = False,
) -> np.ndarray | cp.ndarray:
    """Compute the integral of a function over a (collection of) simplex/ices.

    function: The function to be integrated. This function must be vectorized
    to accept arguments of shape

    N x (n+1) x n

    where n is the number of dimensions of the simplices

    simplices: The simplices with shape as described above.
    scheme: Integration scheme to be used.
    use_gpu: If true, use cupy instead of numpy
    """
    xp = cp.get_array_module(simplices)
    # Pick integration scheme
    num_dimensions = simplices.shape[2]
    if scheme is None:
        # The second parameter to quadpy's scheme method here is about
        # accuracy of the scheme, not the number of spatial dimensions.
        scheme = quadpy.tn.grundmann_moeller(num_dimensions, 1)

    # barycentric weights should be of shape M x (n+1),
    # where n is the number of dimensions (vertices in the simplex)
    # M is the number of points in the cubature scheme
    barycentric_weights = xp.asarray(scheme.points).T
    weights = xp.asarray(scheme.weights)

    # Find simplex volumes using Cayley-Menger determinants
    contents = xp.abs(
        xp.linalg.det(simplices[:, 1:, :] - simplices[:, 0, None, :])
        / math.factorial(num_dimensions)
    )

    # Generate integration points
    # Integration points should be of size N x M x n
    # Number of simplices x number of integration poits per simplex x dimension
    integration_points = barycentric_weights @ simplices

    # Reduce dimensionality of integration pont array for calculation
    # efficiency
    num_simplices, points_per_simplex, _ = integration_points.shape
    reshaped_integration_points = integration_points.reshape(
        num_simplices * points_per_simplex, num_dimensions
    )
    # Output
    function_output = function(reshaped_integration_points)
    _, output_size = function_output.shape
    reshaped_function_output = function_output.reshape(
        num_simplices, points_per_simplex, output_size
    )
    (function_output)
    print(printreshaped_function_output)
    weighted_output = (
        reshaped_function_output
        * contents[:, np.newaxis, np.newaxis]
        * weights[np.newaxis, :, np.newaxis]
    )

    # Sum over the points per simplex axis
    integral = xp.sum(weighted_output, axis=1)
    return integral
