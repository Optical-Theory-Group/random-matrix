"""Classes for describing the statistical properties of the particles in the
scattering medium"""

from dataclasses import dataclass, field
from typing import Protocol
from random_matrix.utils.types import FloatLike, DensityFunction
from random_matrix.utils import integration_utils
import numpy as np


@dataclass
class ParticleStatisticsTerm:
    """Class for one term in the particle physical properties probability
    density function.

    The total probability density function for the physical properties
    associated with the particles should be given by the sum of one or more
    ParticleStatisticsTerm objects. Separating the density function into terms
    allows for straightforward handling of delta functions. Consider for
    example the following density function for two parameters "x" and "m":

    f(x, m) = 1/3*delta(x-1)*delta(m-1.2) + 2/3*delta(x-2)*delta(m-1.5)

    This would correspond to mixture of particles with non-random
    parameters x=1, m=1.2 and x=2, m=1.5 in the ratio 1:2. Each of the two
    terms in the above expression would be represented by an individual
    ParticleStatisticsTerm object.


    Attributes
    ----------
    residual_density : DensityFunction
        Probability density function of remaining variables after delta
        functions have been accounted for. For example, if

        f(x, m) = delta(x-1)*m/2 for 0.0 < m < 1.0,

        then residual_density would be the function

        g(m) = m/2
    residual_density_domain : dict[str, list[float]]
        Integration domains for the residual integration variables. In the
        above example, this would be equal to

        {"m": [0.0, 1.0]}
    delta_distributions: dict[str, FloatLike]
        dictionary containing Dirac delta distributions and the values at which
        the peak occurs. In the above example, this would be equal to
        {"x": 1.0}
    delta_factor : float
        A constant factor that allows for normalisation of the density function
        when multiple terms with delta functions are present. In the example at
        the beginning of the class docstring, this would be equal to either
        1/3 or 2/3 depending on the term.
    integral : float
        The integral of the given density function term over the given domain
    """

    residual_density: DensityFunction | None = None
    residual_density_domain: dict[str, list[FloatLike]] = field(
        default_factory=dict
    )
    delta_distributions: dict[str, FloatLike] = field(default_factory=dict)
    delta_factor: FloatLike = 1.0

    params: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_input()

        delta_params, density_params = self._get_params()
        self._check_params(delta_params, density_params)
        self.params = delta_params | density_params

    @staticmethod
    def _validate_input() -> None:
        pass

    def _get_params(self) -> tuple[set[str], set[str]]:
        """Return sets containing all of the physical parameters
        from the residual density and dirac distributions."""

        delta_params = set(self.delta_distributions.keys())
        density_params = set(self.residual_density_domain.keys())
        return delta_params, density_params

    @staticmethod
    def _check_params(
        delta_params: set[str], density_params: set[str]
    ) -> None:
        """Check that physical parameters are not repeated in the delta
        functions and residual density function."""

        if delta_params & density_params != set():
            raise ValueError(
                f"Physical parameters in the delta functions and "
                f"density function must not overlap. You have "
                f"{delta_params} and {density_params}"
            )

    @property
    def integral(self) -> FloatLike:
        """Compute the integral of the probability density function term"""

        # Dirac deltas always integrate to 1.0, so we can begin with the
        # delta_factor
        integral = self.delta_factor

        # If there is no residual density, we don't need to integrate anything
        # else
        if self.residual_density is None:
            return integral

        integral *= integration_utils.integrate_density_function(
            self.residual_density, self.residual_density_domain
        )
        return integral


@dataclass
class ParticleStatistics:
    """Container class for one or more ParticleStatisticsTerm obejcts

    Attributes
    ----------
    terms : list[ParticleStatisticsTerm]
        A list containing ParticleStatisticsTerm objects. The true, overall
        density is the sum of these terms.
    integral : float
        The integral of the given density function, computed by summing over
        terms.
    """

    terms: list[ParticleStatisticsTerm]

    def __post_init__(self) -> None:
        # If only a single term is given, put it into an array
        if isinstance(self.terms, ParticleStatisticsTerm):
            self.terms = [self.terms]
        self._check_param_consistency()
        self._check_normalization()

    def _check_param_consistency(self) -> None:
        """Check that the physical parameters within each term object are
        consistent."""

        # If there is only a single term, there's nothing to check
        if len(self.terms) == 1:
            pass

        param_sets = [term.params for term in self.terms]
        reference = param_sets[0]
        if not all(s == reference for s in param_sets[1:]):
            raise ValueError(
                "Inconsistnet physical parameters in "
                "terms contained in ParticleStatistics object"
            )

    def _check_normalization(self) -> None:
        """Check that the given probability density function is normalized
        i.e. its integral is 1.0"""

        integral = self.integral
        if not np.isclose(integral, 1.0):
            raise ValueError(
                f"Given probability density function is not "
                f"normalized. Its integral is {integral}"
            )

    @property
    def integral(self) -> FloatLike:
        """Compute the total integral of the probability density function"""

        integral: FloatLike = 0.0
        for term in self.terms:
            integral += term.integral
        return integral
