from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class MediumParameters:
    """Parameters that describe the properies of the background medium."""

    wavelength: float
    number_density: float
    slab_thickness: float

    def __post_init__(self):
        # Physics abbreviations
        self.n = self.number_density
        self.L = self.slab_thickness

        # Derived quantities
        self.k = 2 * np.pi / self.wavelength

        # Constant factors used in integration
        self.mean_const_factor = 2 * np.pi * self.n * self.L / self.k**2
