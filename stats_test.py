import pickle
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import shapely
import tqdm
import quadpy
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
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
    "/home/nbyrnes/code/random-matrix/paper_data/mode_grids.pkl"
)
with MODE_GRIDS_FILENAME.open("rb") as f:
    mode_grids = pickle.load(f)

first = [
    "pp,pp",
    "pp,pe",
    "pp,ep",
    "pp,ee",
    "pe,pe",
    "pe,pe",
    "pe,ep",
    "pe,ee",
    "ep,ep",
    "ep,ee",
    "ee,ee",
]
second = [
    "t,t",
    "t,r",
    "t,t2",
    "t,r2",
    "r,r",
    "r,t2",
    "r,r2",
    "t2,t2",
    "t2,r2",
    "r2,r2",
]

my_grid = mode_grids[-1]
print(my_grid.num_propagating)
incident_index = 0

wavelength = 550e-9
slab_thickness = 1.8992695221776513e-06
number_density = 5.921762640653617e17
medium_parameters = MediumParameters(
    wavelength=wavelength,
    number_density=number_density,
    slab_thickness=slab_thickness,
)
term = DensityFunctionTerm.from_delta({"x": 2, "m": 1.2})

# 2D version
particle_statistics = ParticleStatistics(
    term,
    isotropic_sphere.get_A,
    isotropic_sphere.get_A_product,
    isotropic_sphere.get_A_product_conj,
)
medium_statistics = MediumStatistics([particle_statistics])


# Set up the relevant indices
propagating_indices = my_grid.propagating_indices
quads_a = [(j, incident_index, j, incident_index) for j in propagating_indices]
quads_b = [
    (-incident_index, j, -incident_index, j) for j in propagating_indices
]
quads = quads_a + quads_b

supplied_indices = {
    "covariance": {key1: {key2: [] for key2 in second} for key1 in first}
}
for key in ["t,t", "r,r"]:
    supplied_indices["covariance"]["pp,pp"][key] = quads

# 2D NUMPY
use_np_config = IntegrationTaskConfig(use_gpu=False)

simulation_name = f"test_auto_correlations"
input_statistics_manager = InputStatisticsManager(
    simulation_name,
    medium_parameters,
    medium_statistics,
    my_grid,
    supplied_indices=None,
    use_dirac_density=False,
    integration_method="midpoint",
    covariance_cubature_scheme=None,
    integration_task_config=use_np_config,
)
integration_result_list, duration = input_statistics_manager.get_statistics()