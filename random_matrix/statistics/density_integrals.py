"""AmplitudeMatrixStatistics class that """

import copy
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import quadpy

from random_matrix.amplitude_matrix.amplitude_matrix import AmplitudeMatrix
from random_matrix.statistics.density_function import DensityFunction
from random_matrix.utils.types import MathematicalFunction


@dataclass
class AmplitudeMatrixStatistics:
    amplitude_matrix: MathematicalFunction
    particle_statistics: DensityFunction

    def get_statistics(self) -> None:
        # self._get_mean()
        pass

    def _get_mean(self) -> None:
        expected_num_params = len(
            set(inspect.getfullargspec(self.a_matrix).args)
        )

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
