import numpy as np

from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.modes import mode_grid, mode_grid_generator
from random_matrix.statistics import density_function, density_integrals
from random_matrix.statistics.density_function import (DeltaDensityFactor,
                                                       DensityFunction,
                                                       DensityFunctionTerm,
                                                       RegularDensityFactor)
from random_matrix.statistics.index_finder import IndexFinder
from random_matrix.statistics.integration_task import IntegrationTaskPreparer
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.statistics.medium_statistics import (MediumStatistics,
                                                        ParticleStatistics)
from random_matrix.statistics.scattering_statistics import \
    InputStatisticsManager
from random_matrix.utils import function_utils

print("Preparing Grid")
my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.1715, 0.1715),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
#my_grid.plot()

print("Preparing integrals")
wavelength = 500e-9
slab_thickness = 1.8992695221776513e-06
number_density = 5.921762640653617e+17
medium_parameters = MediumParameters(
    wavelength=wavelength,
    number_density=number_density,
    slab_thickness=slab_thickness,
)


term = DensityFunctionTerm.from_delta({"x": 2.0, "m": 1.2})
A_matrix = isotropic_sphere.get_A
particle_statistics = ParticleStatistics(term, A_matrix)
medium_statistics = MediumStatistics([particle_statistics])
input_statistics_manager = InputStatisticsManager(
    medium_parameters, medium_statistics, my_grid
)


indices = input_statistics_manager._get_indices()
tasks = input_statistics_manager._get_integration_tasks(indices)
t_task = tasks.tasks[0]
t_result = t_task.execute_task()

for i, pair in enumerate(t_result.sub_block_locations):
    if pair == (0,0):
        print(i)
        break

integral = t_result.integral[i][0]
weight = my_grid.by_index(0).weight
print(integral)
# print("Calculating statistics")
# mean_S = input_statistics_manager.get_statistics()
# real = np.real(mean_S)
# imag = np.imag(mean_S)
# print("Done")


