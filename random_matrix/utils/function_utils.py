"""Utility functions that manipulate mathematical functions

Also includes some special functions.
"""

import copy
import functools
import inspect
import math
from typing import Any, Callable

import numpy as np

from random_matrix.utils.types import FloatLike, MathematicalFunction


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

    def vectorized_function(args: list[FloatLike]) -> FloatLike:
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
    def function_sum(*args: FloatLike, **kwargs: FloatLike) -> FloatLike:
        return sum(f(*args, **kwargs) for f in functions)

    return function_sum


def multiply_functions(
    functions: list[MathematicalFunction],
) -> MathematicalFunction:
    """Multiply together a list of functions, returning a new function."""

    if not isinstance(functions, list):
        raise ValueError(
            "Input to multiply_functions must be a list ofvfunctions."
        )
    if len(functions) == 1:
        return functions[0]

    @functools.wraps(functions[0])
    def function_product(*args: FloatLike, **kwargs: FloatLike) -> FloatLike:
        return math.prod(f(*args, **kwargs) for f in functions)

    return function_product


def multiply_function_by_constant(
    function: MathematicalFunction, constant: FloatLike
) -> MathematicalFunction:
    """Return a new function that is a constant multiplied by an old
    function."""

    if np.isclose(constant, 1.0):
        return function

    @functools.wraps(function)
    def new_function(*args: FloatLike) -> MathematicalFunction:
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
    def fixed_function(*args: FloatLike) -> FloatLike:
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

    def new_function(arg: FloatLike) -> FloatLike:
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
                # This is the most common case for high dimensional integration
                # for more information ,see
                # https://github.com/sigma-py/quadpy/wiki/Dimensionality-of-input-and-output-arrays
                ki_array = x[0]
                kj_array = x[1]

                nx, ny = np.shape(ki_array)
                output_array = np.zeros((4, nx, ny), dtype=np.complex128)
                for i in range(nx):
                    for j in range(ny):
                        ki = ki_array[i, j]
                        kj = kj_array[i, j]
                        out = function(ki, kj)
                        output_array[:, i, j] = out

        return output_array

    return vectorized_function
