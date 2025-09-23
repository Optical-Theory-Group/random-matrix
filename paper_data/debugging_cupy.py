from pathlib import Path
import sys
import os

# Add parent directory to Python path
project_root = Path("/home/nbyrnes/code/random-matrix/")  # <-- adjust this
sys.path.insert(0, str(project_root))

import pickle
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import shapely
import tqdm
import quadpy
import warnings
from random_matrix.amplitude_matrix import (
    isotropic_sphere,
    scattering_geometry,
)
from random_matrix.input_statistics import density_function, density_integrals
from random_matrix.input_statistics.density_function import (
    DeltaDensityFactor,
    DensityFunction,
    DensityFunctionTerm,
    RegularDensityFactor,
)
from random_matrix.input_statistics.index_finder import IndexFinder
from random_matrix.input_statistics.input_statistics_manager import (
    InputStatisticsManager,
)
from random_matrix.input_statistics.integration_task import (
    IntegrationTaskPreparer,
    IntegrationTaskConfig,
)
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.modes import mode_grid, mode_grid_factory
from random_matrix.utils import (
    array_utils,
    function_utils,
    geometry_utils,
    integration_utils,
    matrix_utils,
    special_functions,
)
from random_matrix.scattering_matrix import sampler
import cupy as cp

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

# 2D version
particle_statistics_2d = ParticleStatistics(
    term,
    isotropic_sphere.get_A,
    isotropic_sphere.get_A_product,
    isotropic_sphere.get_A_product_conj,
)
medium_statistics_2d = MediumStatistics([particle_statistics_2d])

my_grid = mode_grid_factory.from_tiling(
    tiling_type="rectangles",
    side_length=(0.3, 0.3),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot()

use_np_config = IntegrationTaskConfig(use_gpu=False)

num_modes = my_grid.num_propagating
print(f"Starting num_modes = {num_modes}")
# 2D
simulation_name = f"debugging"
input_statistics_manager_2d = InputStatisticsManager(
    simulation_name,
    medium_parameters,
    medium_statistics_2d,
    my_grid,
    supplied_indices=None,
    use_dirac_density=False,
    integration_method="midpoint",
    covariance_cubature_scheme=None,
    integration_task_config=use_np_config,
)
pool = input_statistics_manager_2d.get_matrix_pool()

use_cupy = True

pool.populate_single_pool(
    num_matrices=100, populate_single_pool_M=True, use_cupy=use_cupy
)

analysis_points = [i * 100 for i in range(20)]
num_samples = 100


def get_t_values_np(matrices):
    dat = []
    for mat in matrices:
        t = matrix_utils.get_block(mat, "t")
        prod = np.conj(t.T) @ t
        eigs = np.linalg.eigvalsh(prod)
        dat = np.concatenate((np.array(dat), eigs))
    return dat


def get_t_values_cp(matrices):
    dat = []
    for mat in matrices:
        t = matrix_utils.get_block(mat, "t")
        prod = cp.conj(t.T) @ t
        eigs = cp.linalg.eigvalsh(prod)
        dat = cp.concatenate((cp.array(dat), eigs))
    return dat


analysis_functions = {"eigs": get_t_values_cp}
data = pool.cascade(
    num_samples,
    analysis_points,
    analysis_functions,
    use_transfer_matrices=False,
)
