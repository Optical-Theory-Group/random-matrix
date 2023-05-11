from random_matrix.statistics import density_function
from random_matrix.utils.types import FloatLike, MathematicalFunction
from dataclasses import dataclass


@dataclass
class ParticleStatistics(density_function.DensityFunction):
    """Subclass of DensityFunction for keeping track of components of the
    total particle probability density function specific to a certain type
    of particle.

    Attributes
    ----------
    terms:
        Terms of the particle parameter density function.
    A_matrix:
        The A matrix function associated with the particles.
    mixing_ratio:
        The concentration of the particles in the medium relative to other
        particle types, expressed as a fraction of unity.
    """

    A_matrix: MathematicalFunction
    mixing_ratio: float


@dataclass
class MediumStatistics:
    pass


def test(x: FloatLike, y: FloatLike) -> FloatLike:
    return x + y


domain = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
mixing_ratio = 0.2
regular_factor = density_function.RegularDensityFactor(test, domain)
term = density_function.DensityFunctionTerm(regular=regular_factor)
my = ParticleStatistics([term], test, 0.2)
