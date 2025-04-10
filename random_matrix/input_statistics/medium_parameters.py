from dataclasses import dataclass, field

import numpy as np

from random_matrix.utils.types import Numeric


@dataclass(slots=True)
class MediumParameters:
    """Parameters that describe the properies of the background medium."""

    wavelength: float
    number_density: float
    slab_thickness: float

    n: float = field(init=False)
    L: float = field(init=False)
    k: float = field(init=False)

    mean_const_factor: Numeric = field(init=False)
    cov_const_factor: Numeric = field(init=False)

    def __post_init__(self) -> None:
        # Physics abbreviations
        self.n = self.number_density
        self.L = self.slab_thickness

        # Derived quantities
        self.k = 2 * np.pi / self.wavelength

        # Constant factors used in integration
        self.mean_const_factor = 2 * np.pi * self.n * self.L / self.k**2
        self.cov_const_factor = self.n * self.L / self.k**2
