"""
Module containing custom types used throughout the library for type
checking.
"""
import typing

import numpy as np

Numeric = typing.TypeVar("Numeric", bound=np.generic)
