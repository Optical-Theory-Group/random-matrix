import warnings

import numpy as np

from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.input_statistics.density_function import DensityFunctionTerm
from random_matrix.input_statistics.input_statistics_manager import \
    InputStatisticsManager
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics, ParticleStatistics)
from random_matrix.modes import mode_grid_generator

np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.4, 0.4),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)

wavelength = 500e-9
slab_thickness = 1.8992695221776513e-06
number_density = 5.921762640653617e17
medium_parameters = MediumParameters(
    wavelength=wavelength,
    number_density=number_density,
    slab_thickness=slab_thickness,
)


term = DensityFunctionTerm.from_delta({"x": 2.0, "m": 1.2})
particle_statistics = ParticleStatistics(
    term,
    isotropic_sphere.get_A,
    isotropic_sphere.get_A_product,
    isotropic_sphere.get_A_product_conj,
)
medium_statistics = MediumStatistics([particle_statistics])

input_statistics_manager = InputStatisticsManager(
    medium_parameters, medium_statistics, my_grid, use_logger=True
)

out = input_statistics_manager.get_statistics()
input_statistics_manager.index_finder.show_report()
input_statistics_manager.shape_classifier.show_report()
input_statistics_manager.show_report()
