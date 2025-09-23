import pickle
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import shapely
import tqdm
import quadpy

from random_matrix.amplitude_matrix import isotropic_sphere
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
from random_matrix.scattering_matrix.matrix_pool import SinglePool


my_pool = SinglePool(
    "debugging",
    parent_data_dir="/home/nbyrnes/code/random-matrix/paper_data/data",
)
S_matrices = my_pool.generate_matrices(10)
print(S_matrices.shape)

S = S_matrices[0]
size_of_S = len(S) // 2

r = S[0:size_of_S, 0:size_of_S]
t = S[size_of_S:, 0:size_of_S]
t2 = S[0:size_of_S, size_of_S:]
r2 = S[size_of_S:, size_of_S:]

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(np.abs(r))
ax[0,1].imshow(np.abs(t2))
ax[1,0].imshow(np.abs(t))
ax[1,1].imshow(np.abs(r2))
