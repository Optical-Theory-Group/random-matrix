"""Generic types used throughout library for static type checking"""

import numpy as np
import numpy.typing as npt
from typing import Protocol

# Generic float-like type
FloatLike = (
    float
    | complex
    | np.float64
    | np.complex128
    | npt.NDArray[np.float64]
    | npt.NDArray[np.complex128]
)


# Type hinting for distributions
class MathematicalFunction(Protocol):
    def __call__(self, *args: FloatLike) -> FloatLike:
        ...
