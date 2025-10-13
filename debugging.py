from pathlib import Path
import sys
import os


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
    side_length=(0.07, 0.07),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot()
print(f"Num modes = {my_grid.num_propagating}")

indices = my_grid.propagating_indices
num_indices = len(indices)

elements = [
    (index_i * num_indices + index_j, i, j)
    for index_i, i in enumerate(indices)
    for index_j, j in enumerate(indices)
]


total = my_grid.num_propagating**2 * (my_grid.num_propagating**2 + 1) / 2
total_auto = my_grid.num_propagating**2
total_mem = total - total_auto

print(f"Number to check: {total}")

i, j, u, v = (0, 0, 0, 0)
mode_i = my_grid.by_index(i).vertices
mode_j = my_grid.by_index(j).vertices
mode_u = my_grid.by_index(u).vertices
mode_v = my_grid.by_index(v).vertices

cartesian_product = geometry_utils.iterated_cartesian_product(
    [mode_i, mode_j, mode_u, mode_v]
)


# Test the Minkowski heuristic
# ------------------------------------------------

start = time.perf_counter()

# Find the centroids
mean_i = np.mean(mode_i, axis=0)
mean_j = np.mean(mode_j, axis=0)
centre_ij = np.mean(np.vstack((mean_i, mean_j)), axis=0)
mean_u = np.mean(mode_u, axis=0)
mean_v = np.mean(mode_v, axis=0)
centre_uv = np.mean(np.vstack((mean_u, mean_v)), axis=0)

# Find the difference space associated with centre_ij
mode_j_ref = geometry_utils.reflect_through_point(mode_j, centre_ij)
ij_intersect = geometry_utils.intersection(mode_i, mode_j_ref)
new_ij = 2 * geometry_utils.translate_points(ij_intersect, -centre_ij)

# Find the difference space associated with centre_uv
mode_v_ref = geometry_utils.reflect_through_point(mode_v, centre_uv)
uv_intersect = geometry_utils.intersection(mode_u, mode_v_ref)
new_uv = 2 * geometry_utils.translate_points(uv_intersect, -centre_uv)

# Find the intersection of the difference spaces and get
# its area
ijuv_intersect = geometry_utils.intersection(new_ij, new_uv)
area = geometry_utils.get_polygon_area(ijuv_intersect)
end = time.perf_counter()
print(f"Minkowski method took: {end -start}")
print(f"Would take in total: {(end - start) * total / 60**2 :.2f} hours")
print(f"Would take for auto: {(end - start) * total_auto / 60**2 :.2f} hours")
print(f"Would take for mems: {(end - start) * total_mem / 60**2 :.2f} hours")
# Testthe cdd method

start = time.perf_counter()

# Get the integration domain
columns_to_keep = [0, 1, 2, 3, 4, 5]

reduced_intersection = geometry_utils.get_intersection_vertices(
    cartesian_product
)[:, columns_to_keep]
reduced_hull = scipy.spatial.ConvexHull(
    reduced_intersection, qhull_options="QJ"
)
end = time.perf_counter()
print(f"cdd method took: {end -start}")
print(f"Would take in total: {(end - start) * total / 60**2 :.2f} hours")
print(f"Would take for auto: {(end - start) * total_auto / 60**2 :.2f} hours")
print(f"Would take for mems: {(end - start) * total_mem / 60**2 :.2f} hours")


################


start = time.perf_counter()

# Find the centroids
mean_i = np.mean(mode_i, axis=0)
mean_j = np.mean(mode_j, axis=0)
centre_ij = np.mean(np.vstack((mean_i, mean_j)), axis=0)
mean_u = np.mean(mode_u, axis=0)
mean_v = np.mean(mode_v, axis=0)
centre_uv = np.mean(np.vstack((mean_u, mean_v)), axis=0)
match = np.allclose(centre_ij, centre_uv)
end = time.perf_counter()
print(f"Match method took: {end -start}")
print(f"Would take in total: {(end - start) * total / 60**2 :.2f} hours")
print(f"Would take for auto: {(end - start) * total_auto / 60**2 :.2f} hours")
print(f"Would take for mems: {(end - start) * total_mem / 60**2 :.2f} hours")
