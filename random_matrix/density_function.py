"""Classes for describing probability density functions containing Dirac delta
distributions.

Complex probability density functions consisting of combinations of delta
functions and regular functions should be built up gradually. Consider, for
example the following density function, defined over the region
0 < x < 1, 0 < y < 1.

f(x, y) = 0.3*delta(x-0.4)delta(y-0.3) + 0.2*delta(x-0.1)*2y + 0.5*2x*2y

This consists of three terms separated by + symbols. These can be analysed
separately

Term 1:
    DeltaDensityFactor with
        density = {"x": 0.4, "y": 0.3}
        factor = 0.3
    No RegularDensityFactor

Term 2:
    DeltaDensityFactor with
        density = {"x": 0.1}
        factor = 0.2
    RegularDensityFactor with
        density: g(y) = 2*y
        domain = {"y": [0.0,1.0]}

    Note that one could instead have the delta factor = 1.0 and the regular
    denstiy g(y) = 2*y * 0.2

Term 3:
    No DeltaDensityFactor
    RegularDensityFactor with
        density: g(x,y) = 0.5*2x*2y
        domain = {"x": [0.0,1.0], "y": [0.0,1.0]}

The DensityFunction object can be formed by passing the three
DensityFunctionTerm objects to its constructor as a list.
"""

from dataclasses import dataclass, field
from typing import Self
from random_matrix.utils.types import FloatLike, MathematicalFunction
from random_matrix.utils import integration_utils
import numpy as np


@dataclass
class RegularDensityFactor:
    """Factor of a term in a probability density function that is a regular
    mathematical function.

    Attributes
    ----------
    density:
        The factor of the density function.
    domain:
        The integration domain for each variable
    integral:
        The integral of the density over the domain
    """

    density: MathematicalFunction
    domain: dict[str, list[FloatLike]]

    @property
    def integral(self) -> FloatLike:
        integral = integration_utils.product_integral(
            self.density, self.domain
        )
        return integral


@dataclass
class DeltaDensityFactor:
    """Factor of a term in a probability density function that is a product of
    Dirac delta distributions. Note that this class only supports delta
    functions of the form

    delta(variable - constant)

    Attributes
    ----------
    density:
        Delta functions as a dict of variable-constant string-float pairs.
    factor:
        Constant factor multiplying the product of delta functions.
    integral:
        The integral of the density over the domain. In this case, this is just
        the factor. We assume, of course, that the imaginary integration domain
        includes the delta function peak.
    """

    density: dict[str, FloatLike]
    factor: FloatLike = 1.0

    @property
    def integral(self) -> FloatLike:
        integral = self.factor
        return integral


@dataclass
class DensityFunctionTerm:
    """A single term in the probability density function, potentially
    containing both regular and delta factors, which are implicitly
    multiplied together.

    Attributes
    ----------
    residual:
        The regular factor of the term.
    delta:
        The delta factor of the term.
    params:
        A set of all of the variables involved in the term.
    integral:
        The total integral of the term.
    """

    regular: RegularDensityFactor | None = None
    delta: DeltaDensityFactor | None = None

    params: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_input()
        delta_params, regular_params = self._get_params()
        self._check_params(delta_params, regular_params)
        self.params = delta_params | regular_params

    def _validate_input(self) -> None:
        if self.regular is None and self.delta is None:
            raise ValueError(
                "At least one of regular or delta must be "
                "given. You have both as None."
            )

    def _get_params(self) -> tuple[set[str], set[str]]:
        """Return sets containing all of the physical parameters
        from the residual density and dirac distributions."""

        delta_params = (
            set() if self.delta is None else set(self.delta.density.keys())
        )
        regular_params = (
            set() if self.regular is None else set(self.regular.domain.keys())
        )
        return delta_params, regular_params

    @staticmethod
    def _check_params(
        delta_params: set[str], regular_params: set[str]
    ) -> None:
        """Check that physical parameters are not repeated in the delta
        functions and residual density function."""

        if delta_params & regular_params != set():
            raise ValueError(
                f"Physical parameters in the delta functions and "
                f"density function must not overlap. You have "
                f"{delta_params} and {regular_params}"
            )

    @property
    def integral(self) -> FloatLike:
        """Compute the integral of the probability density function term"""
        regular = 1.0 if self.regular is None else self.regular.integral
        delta = 1.0 if self.delta is None else self.delta.integral
        return regular * delta


@dataclass
class DensityFunction:
    """A probability density function as a collection of terms.

    Attributes
    ----------
    terms:
        A list containing DensityFunctionTerm objects. The total density is
        the sum of these terms.
    integral : float
        The integral of the total density function.

    Methods
    ----------
    from_function:
        Convenience constructor that creates an instance for a given density
        function and domain without any delta functions.
    from_deltas:
        Convenience constructor that creates an instance for a product of
        delta functions without a residual regular density function.

    """

    terms: list[DensityFunctionTerm]

    def __post_init__(self) -> None:
        # If only a single term is given, put it into an array
        if isinstance(self.terms, DensityFunctionTerm):
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
        return sum(term.integral for term in self.terms)

    @classmethod
    def from_function(
        cls, function: MathematicalFunction, domain: dict[str, list[FloatLike]]
    ) -> Self:
        """Create an instance with only a single density function and no
        deltas"""
        regular = RegularDensityFactor(function, domain)
        term = DensityFunctionTerm(regular, None)
        density = cls([term])
        return density

    @classmethod
    def from_deltas(cls, deltas: dict[str, FloatLike]) -> Self:
        """Create an instance with only delta functions"""
        delta = DeltaDensityFactor(deltas)
        term = DensityFunctionTerm(None, delta)
        density = cls([term])
        return density
