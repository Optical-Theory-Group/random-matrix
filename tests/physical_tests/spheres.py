import time

import matplotlib.pyplot as plt
import numpy as np
from old_code.functions import getT

from random_matrix.amplitude_matrix import isotropic_sphere, mie_sphere
from random_matrix.input_statistics import density_function, density_integrals
from random_matrix.input_statistics.density_function import (
    DeltaDensityFactor, DensityFunction, DensityFunctionTerm,
    RegularDensityFactor)
from random_matrix.input_statistics.index_finder import IndexFinder
from random_matrix.input_statistics.input_statistics_manager import \
    InputStatisticsManager
from random_matrix.input_statistics.integration_task import \
    IntegrationTaskPreparer
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics, ParticleStatistics)
from random_matrix.modes import mode_grid, mode_grid_factory
from random_matrix.utils import function_utils

print("Preparing Grid")
# my_grid = mode_grid_generator.from_tiling(
#     tiling_type="rectangles",
#     side_length=(0.1715, 0.1715),
#     r_lim=1.2,
#     grid_wave_type="propagating",
#     rotation_angle=0.0,
#     translation_vector=np.array([0.0, 0.0]),
# )

kxs = np.linspace(-1.0, 1.0, 10**4)

scattering_angle = []
parallel = []
perp = []
unpol = []

x = np.array([[3]])
m = np.array([[1.33]])
ki_x = np.array([[0]])
ki_y = np.array([[0]])
ki_z = np.array([[1]])

for kx in kxs:
    kz = np.sqrt(1 - kx**2)

    kj_x = np.array([[kx]])
    kj_y = np.array([[0.0]])
    kj_z = np.array([[kz]])
    A = mie_sphere.get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A = np.reshape(A, (2, 2))

    par = np.linalg.norm(A[:, 0]) ** 2
    per = np.linalg.norm(A[:, 1]) ** 2
    unp = (par + per) / 2

    scattering_angle.append(np.arccos(kz) * 180 / np.pi)
    parallel.append(par)
    perp.append(per)
    unpol.append(unp)

    kj_z = np.array([[-kz]])
    A = mie_sphere.get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A = np.reshape(A, (2, 2))

    par = np.linalg.norm(A[:, 0]) ** 2
    per = np.linalg.norm(A[:, 1]) ** 2
    unp = (par + per) / 2

    scattering_angle.append(np.arccos(-kz) * 180 / np.pi)
    parallel.append(par)
    perp.append(per)
    unpol.append(unp)

parallel = np.array(parallel) / max(parallel)
perp = np.array(perp) / max(perp)
unpol = np.array(unpol) / max(unpol)

fig, ax = plt.subplots()
ax.scatter(scattering_angle, parallel, color="tab:blue")
ax.scatter(scattering_angle, perp, color="tab:orange")
# ax.scatter(scattering_angle, unpol, color="tab:green")
ax.set_yscale("log")
ax.set_xlim(0, 180)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("Scattering angle")
fig.savefig("/var/home/niall/Code/Science/random-matrix/tests/physical_tests/mie.png")
