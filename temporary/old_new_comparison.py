import pathlib
import time

import matplotlib.pyplot as plt
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
from random_matrix.utils import function_utils, matrix_utils


def mode_sample_cartesian(dx, dy):
    """
    Produces a cartesian grid of wavevector modes for given lattice spacings dx and dy
    """
    mode_list = []
    Nx = int(np.ceil(1 / dx))
    Ny = int(np.ceil(1 / dy))
    x = np.linspace(-Nx * dx, Nx * dx, 2 * Nx + 1)
    y = np.linspace(Ny * dy, -Ny * dy, 2 * Ny + 1)
    xv, yv = np.meshgrid(x, y)
    ny, nx = np.shape(xv)

    # Loop through mode centres
    for j in range(ny):
        for i in range(nx):
            x = xv[j, i]
            y = yv[j, i]

            if x**2 + y**2 < 1:
                mode_list.append(np.array([x, y]))

    n_modes = len(mode_list)
    weight = np.pi / n_modes

    return mode_list, weight


fig, ax = plt.subplots()

# -----------------------------------------------------------------------------
# Old data
# -----------------------------------------------------------------------------

# old_data_path = pathlib.Path(
#     "/var/home/niall/Code/Science/random-matrix/data/old_simulations/mean_t.npy"
# )

mode_list = mode_sample_cartesian(0.1715, 0.1715)[0]
# mean_t_old = np.load(old_data_path.resolve())
mean_t_old = statistics
old_ys = []
old_xs = []

for mode, mean in zip(mode_list, mean_t_old):
    # Get x data
    new_x = np.linalg.norm(mode)
    old_xs.append(new_x)

    # Get y data
    new_y = np.linalg.norm(mean)
    old_ys.append(new_y)

ax.scatter(old_xs, old_ys, color="black", marker="+", s=200.0)

# -----------------------------------------------------------------------------
# New data
# -----------------------------------------------------------------------------

my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.1715, 0.1715),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)

wavelength = 500e-09
slab_thickness = 1.9789234379375613e-06
number_density = 7.40220330081702e+16

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
mean_S = input_statistics_manager.get_statistics()

ys = []
xs = []
indices = range(-68, 68 + 1, 1)

for index in indices:
    # Get x data
    mode = my_grid.by_index(index)
    mode_center = mode.center
    new_x = np.linalg.norm(mode_center)
    xs.append(new_x)

    # Get y data
    q, p = matrix_utils.get_sub_block_indices(
        "t", (index, index), True, my_grid.num_propagating
    )
    sub_block = mean_S[q : q + 2, p : p + 2]
    new_y = np.linalg.norm(sub_block)
    ys.append(new_y)

new = ys[0]

# PLOT
ax.scatter(xs, ys, color="blue", marker="o", s=50.0)
# ax.set_ylim([0.0, 0.01])


for i, oldx in enumerate(old_xs):
    if np.isclose(oldx, 0):
        break


for j, x in enumerate(xs):
    if np.isclose(x, 0):
        break