"""Generic types used throughout library for static type checking"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass

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
@dataclass
class Parameters(Protocol):
    ...


@runtime_checkable
class MathematicalFunction(Protocol):
    def __call__(self, *args: FloatLike, **kwargs: FloatLike) -> FloatLike:
        ...


@runtime_checkable
@dataclass
class AMatrixFunction(Protocol):
    particle_type: str

    def __call__(
        self, k_inc: FloatLike, k_sca: FloatLike, params: Parameters
    ) -> FloatLike:
        ...
