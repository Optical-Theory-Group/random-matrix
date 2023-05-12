"""Utility functions that manipulate mathematical functions"""

import inspect
import copy
from typing import Any, Callable
import functools

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
        raise ValueError(
            "Input to add_functions must be a list of functions."
        )
    if len(functions) == 1:
        return functions[0]

    @functools.wraps(functions[0])
    def function_sum(*args: FloatLike, **kwargs: FloatLike) -> FloatLike:
        return sum(f(*args, **kwargs) for f in functions)

    return function_sum


def multiply_function_by_constant(
    function: MathematicalFunction, constant: FloatLike
) -> MathematicalFunction:
    """Return a new function that is a constant multiplied by an old
    function."""

    @functools.wraps(function)
    def new_function(*args: FloatLike) -> MathematicalFunction:
        return constant * function(*args)

    return new_function
