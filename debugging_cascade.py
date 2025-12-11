import numpy as np
from random_matrix.amplitude_matrix import (
    isotropic_sphere,
)
from random_matrix.input_statistics.density_function import (
    DensityFunctionTerm,
)
import multiprocess as mp
from random_matrix.input_statistics.input_statistics_manager import (
    InputStatisticsManager,
)
from random_matrix.input_statistics.integration_task import (
    IntegrationTaskConfig,
)
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.utils import matrix_utils
from random_matrix.modes import mode_grid_factory
import traceback
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------

wavelength = 550e-9
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
integration_task_config = IntegrationTaskConfig(integration_method="lattice")

side_lengths = [0.5]
my_grid = mode_grid_factory.from_dr_dt(
    0.25, 2 * np.pi / 8, 1.0, False, 0.0, True, False
)
my_grid.plot(show_indices=True)

ism = InputStatisticsManager(
    "cascade_test",
    medium_parameters,
    medium_statistics,
    my_grid,
    integration_task_config,
)
mpm = ism.get_matrix_pool_manager()
# print("Finishing...")
