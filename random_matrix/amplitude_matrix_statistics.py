from random_matrix import amplitude_matrix, particle_statistics

"""Statistical properties of the particles in the medium"""
from dataclasses import dataclass, field
import functools
from typing import Protocol, Any
import copy
import inspect

import numpy as np
import numpy.typing as npt
import quadpy



@dataclass
class AmplitudeMatrixStatistics:
    amplitude_matrix: amplitude_matrix.AmplitudeMatrix
    particle_statistics: particle_statistics.ParticleStatistics

    def __post_init__(self) -> None:
        self.mean = self._get_mean()

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.amplitude_matrix(*args, **kwargs)

    def _get_mean(self) -> None:
        expected_num_params = len(
            set(inspect.getfullargspec(self.a_matrix).args)
        )
        print(expected_num_params)

        partial_functions = []

        for term in self.particle_statistics.terms:
            # If delta distributions are present, enforce their values
            delta_integrated = copy.copy(self.a_matrix)
            if len(term.delta_distributions) > 0:
                delta_integrated = functools.partial(
                    delta_integrated, **term.delta_distributions
                )

            # Scale by delta_factor if it isn't 1.0
            if not np.isclose(term.delta_factor, 1.0):
                delta_integrated = functools.partial(
                    lambda k, *args, **kwargs: k
                    * delta_integrated(*args, **kwargs),
                    k=term.delta_factor,
                )

            # If there is no residual density, we have our function
            if term.residual_density is None:
                partial_functions.append(delta_integrated)
                continue

            # There is a residual density function after the deltas
            # We thus need to integrate
            total_args = set(inspect.getfullargspec(delta_integrated).args)
            residual_args = set(term.residual_density_domain.keys())
            wavevector_args = list(total_args.difference(residual_args))


def amp_matrix(
    k_inc: float, k_sca: float, params: dict[str, np.float64] = {}
) -> float:
    x = params.get("x", 1.0)
    m = params.get("m", 1.0)
    return (k_inc + k_sca) * x * m


def residual_density(x):
    return x / 2.0


particle_statistics_term = ParticleStatisticsTerm(
    residual_density=None,
    residual_density_domain={"x": [0.0, 1.0]},
    delta_distributions={"m": 1.0},
)
particle_statistics = ParticleStatistics(particle_statistics_term)
a_matrix = AmplitudeMatrixStatistics(amp_matrix, particle_statistics)
