"""Utility functions that manipulate mathematical functions

Also includes some special functions.
"""

import copy
import functools
import inspect
import math
from typing import Any, Callable
import cupy as cp
import numpy as np

from random_matrix.utils.types import Numeric, MathematicalFunction


def get_new_signature(variables: list[str]) -> inspect.Signature:
    new_signature = inspect.Signature(
        [
            inspect.Parameter(var, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for var in variables
        ]
    )
    return new_signature


def get_arg_index(function: Callable[..., Any], arg: str) -> int:
    """Get the index of a certain positional argument."""

    return function.__code__.co_varnames.index(arg)


def get_function_variables(function: MathematicalFunction) -> list[str]:
    """Return a list of the variables that a function takes."""

    return list(inspect.signature(function).parameters)


def vectorize_arguments(
    function: MathematicalFunction,
) -> MathematicalFunction:
    """Return a function that is equivalent to the original function, but whose
    arguments have been changed into a vector.

    Note: only works on functions that has purely positional arguments.
    """

    def vectorized_function(args: list[Numeric]) -> Numeric:
        return function(*args)

    return vectorized_function


def add_functions(
    functions: list[MathematicalFunction],
) -> MathematicalFunction:
    """Add together a list of functions, returning a new function."""

    if not isinstance(functions, list):
        raise ValueError("Input to add_functions must be a list of functions.")
    if len(functions) == 1:
        return functions[0]

    @functools.wraps(functions[0])
    def function_sum(*args: Numeric, **kwargs: Numeric) -> Numeric:
        return sum(f(*args, **kwargs) for f in functions)

    return function_sum


def multiply_functions(
    functions: list[MathematicalFunction],
) -> MathematicalFunction:
    """Multiply together a list of functions, returning a new function."""

    if not isinstance(functions, list):
        raise ValueError("Input to multiply_functions must be a list ofvfunctions.")
    if len(functions) == 1:
        return functions[0]

    @functools.wraps(functions[0])
    def function_product(*args: Numeric, **kwargs: Numeric) -> Numeric:
        return math.prod(f(*args, **kwargs) for f in functions)

    return function_product


def multiply_function_by_constant(
    function: MathematicalFunction, constant: Numeric
) -> MathematicalFunction:
    """Return a new function that is a constant multiplied by an old
    function."""

    if np.isclose(constant, 1.0):
        return function

    @functools.wraps(function)
    def new_function(*args: Numeric) -> MathematicalFunction:
        return constant * function(*args)

    return new_function


def fix_last_components(
    function: MathematicalFunction, signs: tuple[..., int]
) -> MathematicalFunction:
    """Method for reducing a function of a 3-vector to that of a 2-vector of
    its first two components.

    The final component is given such that the vector has a norm of 1.
    """

    # args of fixed_function is expected to be a collection of 2 vectors
    @functools.wraps(function)
    def fixed_function(*args: Numeric) -> Numeric:
        third_components = [
            sign * np.sqrt(1 - arg[0] ** 2 - arg[1] ** 2)
            for sign, arg in zip(signs, args)
        ]

        three_vectors = np.array(
            [
                (*arg, third_component)
                for arg, third_component in zip(args, third_components)
            ]
        )

        return function(*three_vectors)

    return fixed_function


def equate_arguments(function: MathematicalFunction) -> MathematicalFunction:
    """Given a function of the form f(x,y,z,...), return a function
    g(a) = f(a,a,a,...)"""

    num_variables = len(get_function_variables(function))

    def new_function(arg: Numeric) -> Numeric:
        original_function_args = (arg for _ in range(num_variables))
        return function(*original_function_args)

    return new_function


def vectorize_functions_args(
    function: MathematicalFunction,
) -> MathematicalFunction:
    """Vectorize function prepartion for integration"""

    def vectorized_function(x):
        num_dim = np.ndim(x)
        num_inputs = np.shape(x)[-1]
        match num_dim:
            case 0:
                # x is a scalar
                return function(x)

            case 1:
                # args are given as an array rather than as separate positional
                # arguments. here we simply unpack them
                return function(*x)

            case 2:
                # In this case, multiple collections of args have been given
                # in a matrix e.g.
                # [x1, x2, x3, ...]
                # [m1, m1, m3, ...]
                # [...         ...]
                #
                # Each column is one set of args. We need to loop over columns
                # and collect the results

                # Assuming the output is a vector, we need to determine its
                # length
                first_input = x[:, 0]
                first_output = function(*first_input)
                match np.ndim(first_output):
                    case 0:
                        output_array = np.zeros((num_inputs))
                        output_array[0] = first_output
                        for col_index in range(1, num_inputs):
                            col = x[:, col_index]
                            out = function(*col)
                            output_array[col_index] = out
                    case 1:
                        length = len(first_output)

                        output_array = np.zeros(
                            (length, num_inputs), dtype=first_output.dtype
                        )
                        output_array[:, 0] = first_output

                        # Loop over reamining inputs
                        for col_index in range(1, num_inputs):
                            col = x[:, col_index]
                            out = function(*col)
                            output_array[:, col_index] = out

            case 3:
                if np.shape(x)[0] == 2:
                    # This is the most common case for high dimensional integration
                    # for more information ,see
                    # https://github.com/sigma-py/quadpy/wiki/Dimensionality-of-input-and-output-arrays
                    kx_array = x[0]
                    ky_array = x[1]

                    nx, ny = np.shape(kx_array)
                    output_array = np.zeros((4, nx, ny), dtype=np.complex128)
                    for i in range(nx):
                        for j in range(ny):
                            kx = kx_array[i, j]
                            ky = ky_array[i, j]
                            out = function(kx, ky)
                            output_array[:, i, j] = out
                else:
                    k1_x_array = x[0]
                    k1_y_array = x[1]
                    k2_x_array = x[2]
                    k2_y_array = x[3]
                    d_x_array = x[4]
                    d_y_array = x[5]
                    nx, ny = np.shape(k1_x_array)
                    output_array = np.zeros((16, nx, ny), dtype=np.complex128)
                    for i in range(nx):
                        for j in range(ny):
                            k1_x = k1_x_array[i, j]
                            k1_y = k1_y_array[i, j]
                            k2_x = k2_x_array[i, j]
                            k2_y = k2_y_array[i, j]
                            d_x = d_x_array[i, j]
                            d_y = d_y_array[i, j]
                            out = function(k1_x, k1_y, k2_x, k2_y, d_x, d_y)
                            output_array[:, i, j] = out

        return output_array

    return vectorized_function

