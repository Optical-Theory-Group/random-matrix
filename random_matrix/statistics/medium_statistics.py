from dataclasses import dataclass, field

import numpy as np

from random_matrix.statistics import density_function, density_integrals
from random_matrix.utils import function_utils
from random_matrix.utils.types import FloatLike, MathematicalFunction


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
    particle_type:
        A string describing the type of particle in the medium
    """

    a_matrix: MathematicalFunction
    a_product: MathematicalFunction
    a_product_conj: MathematicalFunction
    mixing_ratio: float = 1.0

    particle_type: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._check_a_matrix_variable_consistency()
        self.particle_type = self._get_particle_type()

    def _check_a_matrix_variable_consistency(self) -> None:
        """Check that density function variables are a proper subset of
        A matrix variables"""

        a_matrix_variables = function_utils.get_function_variables(
            self.a_matrix
        )
        if not self.variables <= set(a_matrix_variables):
            raise ValueError(
                f"Density function variables: {self.variables} are "
                f"inconsistent with the A_matrix function variables: "
                f"{set(a_matrix_variables)}. Density function variables must "
                f"be a proper subset of the A_matrix function variables."
            )

    def _get_particle_type(self) -> str:
        particle_type: str = self.a_matrix.particle_type
        if particle_type is None:
            raise ValueError(
                "A matrix function has no assicated particle type. Please "
                "provide one."
            )
        return particle_type

    @property
    def integral(self) -> FloatLike:
        return self.mixing_ratio * sum(term.integral for term in self.terms)

    def get_mean_a_matrix(self) -> MathematicalFunction:
        mean_a = density_integrals.integrate_by_density(self.a_matrix, self)
        mean_a = function_utils.multiply_function_by_constant(
            mean_a, self.mixing_ratio
        )
        return mean_a

    def get_covariance_a_matrix(self) -> MathematicalFunction:
        covariance_a = density_integrals.integrate_by_density(
            self.a_product_conj, self
        )
        covariance_a = function_utils.multiply_function_by_constant(
            covariance_a, self.mixing_ratio**2
        )
        return covariance_a

    def get_pseudo_covariance_a_matrix(self) -> MathematicalFunction:
        pseudo_covariance_a = density_integrals.integrate_by_density(
            self.a_product, self
        )
        pseudo_covariance_a = function_utils.multiply_function_by_constant(
            pseudo_covariance_a, self.mixing_ratio**2
        )
        return pseudo_covariance_a


@dataclass
class MediumStatistics:
    particle_terms: list[ParticleStatistics]

    def __post_init__(self) -> None:
        self._check_particle_statistics_normalization()

    def _check_particle_statistics_normalization(self) -> None:
        integral = self.density_integral
        if not np.isclose(integral, 1.0):
            raise ValueError(
                f"Probability density assoicated with physicsal parameters is "
                f"not normalized. Its integral is {integral}."
            )

    @property
    def density_integral(self) -> FloatLike:
        return sum(term.integral for term in self.particle_terms)

    def get_mean_a_matrix(self) -> MathematicalFunction:
        """Computes <A> over the contained particle terms and A functions"""

        partial_results = []
        for particle_term in self.particle_terms:
            partial_results.append(particle_term.get_mean_a_matrix())
        return function_utils.add_functions(partial_results)

    def get_covariance_a_matrix(self) -> MathematicalFunction:
        """Computes <AA*> over the contained particle terms and A functions"""

        partial_results = []
        for particle_term in self.particle_terms:
            partial_results.append(particle_term.get_covariance_a_matrix())
        return function_utils.add_functions(partial_results)

    def get_pseudo_covariance_a_matrix(self) -> MathematicalFunction:
        """Computes <AA> over the contained particle terms and A functions"""

        partial_results = []
        for particle_term in self.particle_terms:
            partial_results.append(
                particle_term.get_pseudo_covariance_a_matrix()
            )
        return function_utils.add_functions(partial_results)
