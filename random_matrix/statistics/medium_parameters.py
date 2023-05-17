from dataclasses import dataclass
import numpy as np

@dataclass
class MediumParameters:
    n: float
    k: float
    L: float

    def __post_init__(self):
        self.mean_const_factor = 2 * np.pi * self.n * self.L / self.k**2
