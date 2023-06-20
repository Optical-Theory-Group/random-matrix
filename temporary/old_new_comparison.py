import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from random_matrix.amplitude_matrix import mie_sphere
from random_matrix.modes import mode_grid, mode_grid_generator
from random_matrix.statistics import density_function, density_integrals
from random_matrix.statistics.density_function import (
    DeltaDensityFactor,
    DensityFunction,
    DensityFunctionTerm,
    RegularDensityFactor,
)
from random_matrix.statistics.index_finder import IndexFinder
from random_matrix.statistics.integration_task import IntegrationTaskPreparer
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.statistics.scattering_statistics import (
    InputStatisticsManager,
)
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


# -----------------------------------------------------------------------------
# Old data
# -----------------------------------------------------------------------------

# old_data_path = pathlib.Path(
#     "/var/home/niall/Code/Science/random-matrix/data/old_simulations/mean_t.npy"
# )
# mean_t_old = np.load(old_data_path.resolve())
mode_list, weight = mode_sample_cartesian(0.1715, 0.1715)
mean_t_old = mean_t
cov_t_old = cov_t
old_ys = []
old_xs = []

for mode, mean in zip(mode_list, mean_t_old):
    # Get x data
    new_x = np.linalg.norm(mode)
    old_xs.append(new_x)
    
    # Get y data
    new_y = np.linalg.norm(mean)
    old_ys.append(new_y)
    
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
term = DensityFunctionTerm.from_delta({"x": 4.0, "m": 1.2})
a_matrix = mie_sphere.get_A
a_prod = mie_sphere.get_A_product
a_prod_conj = mie_sphere.get_A_product_conj
particle_statistics = ParticleStatistics(term, a_matrix, a_prod, a_prod_conj)
medium_statistics = MediumStatistics([particle_statistics])
input_statistics_manager = InputStatisticsManager(
    medium_parameters, medium_statistics, my_grid
)

indices = input_statistics_manager._get_indices()
tasks = input_statistics_manager._get_integration_tasks(indices)

mean_task = tasks.tasks[0]
cov_task = tasks.tasks[4]

mean_result = mean_task.execute_task()
cov_result = cov_task.execute_task()

# MEAN GRAPH
new_xs = []
new_ys = []

for integral, location in zip(
    mean_result.integral, mean_result.sub_block_locations
):
    integral = np.linalg.norm(integral.reshape(2, 2))
    i = location[0]

    weight = my_grid.by_index(i).weight
    integral = integral / weight

    center = my_grid.by_index(i).center
    radius = np.linalg.norm(center)
    new_xs.append(radius)
    new_ys.append(integral)

fig, ax = plt.subplots()
ax.scatter(new_xs, new_ys, color="blue", marker="o", s=50.0)
ax.scatter(old_xs, old_ys, color="black", marker="+", s=200.0)
ax.set_xlabel("|kappa|")
ax.set_ylabel("Mean")

# ----------------------------------
# COV GRAPH
# -----------------------------------
old_ys = []
old_xs = []

for mode, cov in zip(mode_list, cov_t_old):
    # Get x data
    new_x = np.linalg.norm(mode)
    old_xs.append(new_x)
    
    # Get y data
    new_y = np.trace(cov)
    old_ys.append(new_y)


# ---
# NEW
# MEAN GRAPH
new_xs = []
new_ys = []

for integral, location in zip(
    cov_result.integral, cov_result.sub_block_locations
):
    integral = np.trace(integral.reshape(4, 4))
    i = location[1]

    weight = my_grid.by_index(i).weight
    integral = integral / (weight*my_grid.by_index(0).weight)

    center = my_grid.by_index(i).center
    radius = np.linalg.norm(center)
    new_xs.append(radius)
    new_ys.append(integral*2*np.pi)


fig, ax = plt.subplots()
ax.scatter(new_xs, new_ys, color="blue", marker="o", s=50.0)
ax.scatter(old_xs, old_ys, color="black", marker="+", s=200.0)
ax.set_xlabel("|kappa|")
ax.set_ylabel("Cov")
