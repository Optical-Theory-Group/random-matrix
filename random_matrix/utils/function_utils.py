"""Utility functions that manipulate mathematical functions"""

import inspect
import copy
from typing import Any, Callable

import numpy as np

from random_matrix.utils.types import FloatLike, MathematicalFunction


def get_arg_index(function: Callable[..., Any], arg: str) -> int:
    return function.__code__.co_varnames.index(arg)


def get_function_variables(function: MathematicalFunction) -> list[str]:
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

    def function_sum(*args: FloatLike, **kwargs: FloatLike) -> FloatLike:
        evaluations = np.array([f(*args, **kwargs) for f in functions])
        return np.sum(evaluations)

    return function_sum
