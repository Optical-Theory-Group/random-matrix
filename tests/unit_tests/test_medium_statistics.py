import pytest

from random_matrix.statistics import density_function, medium_statistics


def test_particle_statistics_variable_consistency():
    def A_matrix(k1, k2, x, m):
        return (k1 + k2) * x * m

    A_matrix.particle_type = "test"
    mixing_ratio = 1.0

    def density_function_good(x, m):
        return 2 * x * 2 * m

    domain_good = {"x": [0.0, 1.0], "m": [0.0, 1.0]}

    def density_function_bad(x, m, c):
        return 2 * x * 2 * m * 2 * c

    domain_bad = {"x": [0.0, 1.0], "m": [0.0, 1.0], "c": [0.0, 1.0]}

    # Good case. No errors
    term = density_function.DensityFunctionTerm.from_regular(
        density_function_good, domain_good
    )
    particle_stats = medium_statistics.ParticleStatistics(
        term, A_matrix, mixing_ratio
    )

    with pytest.raises(ValueError, match="inconsistent with the A_matrix"):
        # Bad case.
        term = density_function.DensityFunctionTerm.from_regular(
            density_function_bad, domain_bad
        )
        particle_stats = medium_statistics.ParticleStatistics(
            term, A_matrix, mixing_ratio
        )


test_particle_statistics_variable_consistency()
