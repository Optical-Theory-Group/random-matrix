"""Generic types used throughout library for static type checking"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Generic numeric type
Numeric = int | float | complex | np.float64 | np.complex128 | np.ndarray


@runtime_checkable
@dataclass
class Parameters(Protocol): ...


@runtime_checkable
class MathematicalFunction(Protocol):
    def __call__(self, *args: Numeric, **kwargs: Numeric) -> Numeric: ...


@runtime_checkable
@dataclass
class AMatrixFunction(Protocol):
    particle_type: str

    def __call__(
        self, k_inc: Numeric, k_sca: Numeric, params: Parameters
    ) -> Numeric: ...
