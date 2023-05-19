import numpy as np

from random_matrix.modes import mode_grid, mode_grid_generator
from random_matrix.statistics.index_finder import IndexFinder
from random_matrix.statistics.integration_task import IntegrationTaskPreparer
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.amplitude_matrix import test
from random_matrix.statistics.density_function import (
    DensityFunction,
    DensityFunctionTerm,
)
from random_matrix.statistics.scattering_statistics import (
    InputStatisticsManager,
)

print("Preparing Grid")
my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.1, 0.1),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.00]),
)


print("Preparing tasks")
medium_parameters = MediumParameters(1.0, 1.0, 1.0)

term = DensityFunctionTerm.from_delta({"x": 1.0, "m": 1.0})
a_matrix = test.get_A
particle_statistics = ParticleStatistics(term, a_matrix)
medium_statistics = MediumStatistics([particle_statistics])
input_statistics_manager = InputStatisticsManager(
    medium_parameters, medium_statistics, my_grid
)

print("Calculating statistics")
result_list = input_statistics_manager.get_statistics()
print("Done")

