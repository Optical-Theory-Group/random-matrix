import numpy as np

from random_matrix.modes import mode_grid, mode_grid_generator
from random_matrix.statistics.index_finder import IndexFinder
from random_matrix.statistics.integration_task import IntegrationTaskPreparer
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.statistics.density_function import (
    DensityFunction,
    DensityFunctionTerm,
    DeltaDensityFactor,
    RegularDensityFactor,
)
from random_matrix.statistics.scattering_statistics import (
    InputStatisticsManager,
)
from random_matrix.utils import function_utils
from random_matrix.statistics import density_function, density_integrals

print("Preparing Grid")
my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.2, 0.2),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
#my_grid.plot()

print("Preparing integrals")
wavelength = 500e-9
slab_thickness = 1.177e-6
number_density = 4.737e18
medium_parameters = MediumParameters(
    wavelength=wavelength,
    number_density=number_density,
    slab_thickness=slab_thickness,
)


def density(x, m):
    return -2 * (m - 2) * -2 * (x - 1)


term = DensityFunctionTerm.from_regular(
    density, {"x": [0.0, 1.0], "m": [1.0, 2.0]}
)
term = DensityFunctionTerm.from_delta({"x": 1.0, "m": 1.2})
A_matrix = isotropic_sphere.get_A_scattering_plane
particle_statistics = ParticleStatistics(term, A_matrix)
medium_statistics = MediumStatistics([particle_statistics])
input_statistics_manager = InputStatisticsManager(
    medium_parameters, medium_statistics, my_grid
)

print("Calculating statistics")
mean_S = input_statistics_manager.get_statistics()
real = np.real(mean_S)
imag = np.imag(mean_S)
print("Done")
