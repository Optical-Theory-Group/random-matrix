from random_matrix import particle_statistics
from random_matrix.utils.types import DensityFunction
import numpy as np


def test_particle_statistics_term() -> None:
    # No delta functions
    def distribution(x: float, m: float) -> float:
        return x / 2 * m / 2

    domain = {"x": [0.0, 1.0], "m": [0.0, 1.0]}

    particle_statistics_term = particle_statistics.ParticleStatisticsTerm(
        residual_density=distribution, residual_density_domain=domain
    )

    assert particle_statistics_term.delta_distributions == {}
    assert np.isclose(particle_statistics_term.delta_factor, 1.0)
    assert particle_statistics_term.params == {"x", "m"}
    assert particle_statistics_term.residual_density_domain == domain

    # One delta functions
    def distribution_m(m: float) -> float:
        return m / 2

    delta_functions = {"x": 1.0}
    domain = {"m": [0.0, 1.0]}
    particle_statistics_term = particle_statistics.ParticleStatisticsTerm(
        residual_density=distribution,
        residual_density_domain=domain,
        delta_distributions=delta_functions,
    )

    assert particle_statistics_term.delta_distributions == delta_functions
    assert particle_statistics_term.residual_density_domain == domain
    assert particle_statistics_term.params == {"m", "x"}

    # Two delta functions
    delta_functions = {"x": 1.0, "m": 2.0}
    particle_statistics_term = particle_statistics.ParticleStatisticsTerm(
        delta_distributions=delta_functions,
    )

    assert particle_statistics_term.delta_distributions == delta_functions
    assert particle_statistics_term.residual_density_domain == {}
    assert particle_statistics_term.residual_density is None
    assert particle_statistics_term.params == {"m", "x"}


test_particle_statistics_term()
