from pathlib import Path
import sys

# Add parent directory to Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

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

MODE_GRIDS_FILENAME = Path(
    "/home/nbyrnes/code/random-matrix/paper_data/mode_grids_rectangular.pkl"
)

if MODE_GRIDS_FILENAME.exists():
    # Load existing mode grids
    with MODE_GRIDS_FILENAME.open("rb") as f:
        mode_grids = pickle.load(f)
    print("Mode grid successfully loaded")
else:
    raise ValueError("Grids do not exist")

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

# 3D version
particle_statistics_3d = ParticleStatistics(
    term,
    isotropic_sphere.get_A_3d,
    isotropic_sphere.get_A_product_3d,
    isotropic_sphere.get_A_product_conj_3d,
)
medium_statistics_3d = MediumStatistics([particle_statistics_3d])


# Non GPU Config
use_np_config = IntegrationTaskConfig(use_gpu=False)

for grid in mode_grids:

    num_modes = grid.num_propagating
    print(f"Starting num_modes = {num_modes}")
    # 2D
    simulation_name = f"2d_np_rectangular_midpoint_{num_modes}"
    input_statistics_manager_2d = InputStatisticsManager(
        simulation_name,
        medium_parameters,
        medium_statistics_2d,
        grid,
        supplied_indices=None,
        use_dirac_density=False,
        integration_method="midpoint",
        covariance_cubature_scheme=None,
        integration_task_config=use_np_config,
    )
    integration_result_list, duration = (
        input_statistics_manager_2d.get_statistics()
    )

    # 3D
    simulation_name = f"3d_np_rectangular_midpoint_{num_modes}"
    input_statistics_manager_3d = InputStatisticsManager(
        simulation_name,
        medium_parameters,
        medium_statistics_3d,
        grid,
        supplied_indices=None,
        use_dirac_density=False,
        integration_method="midpoint",
        covariance_cubature_scheme=None,
        integration_task_config=use_np_config,
    )
    integration_result_list, duration = (
        input_statistics_manager_3d.get_statistics()
    )
