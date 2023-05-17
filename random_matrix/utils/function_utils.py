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

    @functools.wraps(function)
    def new_function(arg: FloatLike) -> FloatLike:
        original_function_args = (arg for _ in range(num_variables))
        return function(*original_function_args)

    return new_function
