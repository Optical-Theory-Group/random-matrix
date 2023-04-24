"""
Module containing custom array types used throughout the library for type
checking.
"""

from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=np.generic, covariant=True)

Vector = np.ndarray[tuple[int], np.dtype[T]]
Matrix = np.ndarray[tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[tuple[int, ...], np.dtype[T]]
