import pickle
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import shapely
import tqdm

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


seed = 0
np.random.seed(seed)
side_length = 0.16

warnings.filterwarnings("ignore")
my_grid = mode_grid_factory.from_tiling(
    tiling_type="rectangles",
    side_length=(side_length, side_length),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
print(my_grid.num_propagating)
my_grid.plot(
    show_indices=False, savefig="/home/nbyrnes/code/random-matrix/figtest.svg"
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
    medium_parameters, medium_statistics, my_grid
)

start = time.perf_counter()
integration_result_list, mean_S, cov, sigma, chol = (
    input_statistics_manager.get_statistics()
)
end = time.perf_counter()
print(end-start)
assert False

cov_dense = cov.todense()
S_matrices = S_sampler(mean_S, chol, 10**4)
nx, ny, num = S_matrices.shape
nxh = int(nx/2)
nyh = int(ny/2)

cov_data = np.zeros((1296,1296),dtype=np.complex128)
for i in range(num):
    S = S_matrices[:,:,i]
    r = S[0:nxh, 0:nxh]
    t = S[nxh:, 0:nxh]
    t2 =S[0:nxh, nxh:]
    r2 = S[nxh:, nxh:]

    t_list = np.zeros(0)
    r_list = np.zeros(0)
    t2_list = np.zeros(0)
    r2_list = np.zeros(0)

    for row in range(int(36/4)):
        for col in range(int(36/4)):
            t_block = np.ravel(t[row*2:(row+1)*2,col*2:(col+1)*2])
            t_list = np.concatenate((t_list, t_block))

            r_block = np.ravel(r[row*2:(row+1)*2,col*2:(col+1)*2])
            r_list = np.concatenate((r_list, r_block))

            t2_block = np.ravel(t2[row*2:(row+1)*2,col*2:(col+1)*2])
            t2_list = np.concatenate((t2_list, t2_block))

            r2_block = np.ravel(r2[row*2:(row+1)*2,col*2:(col+1)*2])
            r2_list = np.concatenate((r2_list, r2_block))

    S_linear = np.concatenate((r_list, t_list, t2_list, r2_list))
    new_cov = np.outer(S_linear, np.conj(S_linear))
    cov_data = cov_data + new_cov
cov_data = cov_data/num

fig, ax= plt.subplots(1,2)
im1 = ax[0].imshow(np.abs(cov_dense[0:100,0:100]),vmin = 1e-6, vmax=5e-5)
ax[0].set_title("True")
im2 = ax[1].imshow(np.abs(cov_data[0:100,0:100]),vmin = 1e-6, vmax=5e-5)
ax[1].set_title("Generated")
fig.colorbar(im1, ax=ax[0], orientation='vertical')
fig.colorbar(im2, ax=ax[1], orientation='vertical')

fig, ax = plt.subplots(1,2)
ax[0].set_ylim(0,0.0001)
ax[1].set_ylim(0,0.0001)
dat1 = np.diag(cov_dense[0:200])
dat2 = np.diag(cov_data[0:200])
ax[0].plot(range(len(dat1)), dat1)
ax[1].plot(range(len(dat2)), dat2,color="tab:orange")