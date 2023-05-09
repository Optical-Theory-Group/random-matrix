"""Generic types used throughout library for static type checking"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Generic float-like type
FloatLike = (
    float
    | complex
    | np.float64
    | np.complex128
    | npt.NDArray[np.float64]
    | npt.NDArray[np.complex128]
)


@runtime_checkable
class MathematicalFunction(Protocol):
    def __call__(self, *args: FloatLike) -> FloatLike:
        ...
