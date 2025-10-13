from pathlib import Path
import sys
import os
import warnings
# warnings.filterwarnings("error")
# Add parent directory to Python path
project_root = Path("/home/nbyrnes/code/random-matrix/")  # <-- adjust this
sys.path.insert(0, str(project_root))
from scipy.sparse.linalg import eigsh
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
from random_matrix.amplitude_matrix import isotropic_sphere
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
    side_length=(0.2, 0.2),
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
simulation_name = f"6d_test_midpoint"
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

chol_path = input_statistics_manager_2d.simulation_path / "chol.npz"
cov_path = input_statistics_manager_2d.simulation_path / "cov.npz"
pseudo_cov_path = input_statistics_manager_2d.simulation_path / "pseudo_cov.npz"

cov = scipy.sparse.load_npz(cov_path).todense()
pseudo_cov = scipy.sparse.load_npz(pseudo_cov_path).todense()
chol = scipy.sparse.load_npz(chol_path).todense()


cov_t = matrix_utils.get_cov_block(cov, "t,t")
cov_r = matrix_utils.get_cov_block(cov, "r,r")
cov_t2 = matrix_utils.get_cov_block(cov, "t2,t2")
cov_r2 = matrix_utils.get_cov_block(cov, "r2,r2")

eigs_t = np.linalg.eigvalsh(cov_t)
print(f"t,t max value: {np.max(cov_t)}")
print(f"t,t min eigenvalue: {np.min(eigs_t)}")

eigs_r = np.linalg.eigvalsh(cov_r)
print(f"r,r max value: {np.max(cov_r)}")
print(f"r,r min eigenvalue: {np.min(eigs_r)}")

eigs_t2 = np.linalg.eigvalsh(cov_t2)
print(f"t2,t2 max value: {np.max(cov_t2)}")
print(f"t2,t2 min eigenvalue: {np.min(eigs_t2)}")

eigs_r2 = np.linalg.eigvalsh(cov_r2)
print(f"r2,r2 max value: {np.max(cov_r2)}")
print(f"r2,r2 min eigenvalue: {np.min(eigs_r2)}")

eigs = np.linalg.eigvalsh(cov)
print(f"cov max value: {np.max(cov)}")
print(f"cov min eigenvalue: {np.min(eigs)}")


# eigs = np.linalg.eigvalsh(sigma)
# print(f"sigma max value: {np.max(sigma)}")
# print(f"sigma min eigenvalue: {np.min(sigma)}")

# eigs = np.linalg.eigvalsh(sigma_nop)
# print(f"sigma no Pmax value: {np.max(sigma_nop)}")
# print(f"sigma no P min eigenvalue: {np.min(sigma_nop)}")


def pair_test(i, j, u, v):

    first = (i, j)
    second = (u, v)

    ff = (*first, *first)
    ss = (*second, *second)
    fs = (*first, *second)
    sf = (*second, *first)

    ff_mat = matrix_utils.get_cov_sub_block(cov, "t,t", ff)
    fs_mat = matrix_utils.get_cov_sub_block(cov, "t,t", fs)
    ss_mat = matrix_utils.get_cov_sub_block(cov, "t,t", ss)
    sf_mat = matrix_utils.get_cov_sub_block(cov, "t,t", sf)

    ff_mat_hermitian = np.allclose(ff_mat, np.conj(ff_mat.T))
    ss_mat_hermitian = np.allclose(ss_mat, np.conj(ss_mat.T))
    fs_mat_hermitian = np.allclose(fs_mat, np.conj(sf_mat.T))
    print(f"ff is hermitian: {ff_mat_hermitian}")
    print(f"ss is hermitian: {ss_mat_hermitian}")
    print(f"sf/fs are hermitian pairs: {fs_mat_hermitian}")

    if np.allclose(sf_mat, 0.0):
        print("UNCORRELATED!!!!!!!!!")

    ff_min_eig = np.min(np.linalg.eigvals(ff_mat))
    ss_min_eig = np.min(np.linalg.eigvals(ss_mat))

    print(f"ff min eigval = {ff_min_eig}")
    print(f"ff max val: {np.max(ff_mat)}")
    print(f"ss min eigval = {ss_min_eig}")
    print(f"ss max val: {np.max(ss_mat)}")

    combined = np.block([[ff_mat, fs_mat], [sf_mat, ss_mat]])
    combined_min_eig = np.min(np.linalg.eigvals(combined))
    print("-------")
    print(f"combined min eigval = {combined_min_eig}")
    print(f"combined max val = {np.max(combined)}")

opts = ["r", "t", "t2", "r2"]
for b1 in opts:
    for b2 in opts:
        b_11 = f"{b1},{b1}"
        b_12 = f"{b1},{b2}"
        b_21 = f"{b2},{b1}"
        b_22 = f"{b2},{b2}"

        data = {}
        indices = my_grid.propagating_indices
        # indices = [9, -12, -7, 8, 0, -8, 7, 12, -9]
        for i in indices:
            for j in indices:
                for u in indices:
                    for v in indices:
                        first = (i, j)
                        second = (u, v)

                        ff = (*first, *first)
                        ss = (*second, *second)
                        fs = (*first, *second)
                        sf = (*second, *first)

                        fs_mat = matrix_utils.get_cov_sub_block(cov, b_12, fs)
                        is_uncorrelated = np.allclose(fs_mat, 0.0)
                        if is_uncorrelated:
                            continue

                        ff_mat = matrix_utils.get_cov_sub_block(cov, b_11, ff)
                        ss_mat = matrix_utils.get_cov_sub_block(cov, b_22, ss)
                        sf_mat = matrix_utils.get_cov_sub_block(cov, b_21, sf)
                        combined = np.block([[ff_mat, fs_mat], [sf_mat, ss_mat]])
                        combined_min_eig = np.min(np.linalg.eigvals(combined))
                        data[i, j, u, v] = combined_min_eig
        print(b_12)
        print(np.min(list(data.values())))

i, j, u, v = (9, 9, 9, -12)
first = (i, j)
second = (u, v)

ff = (*first, *first)
ss = (*second, *second)
fs = (*first, *second)
sf = (*second, *first)

fs_mat = matrix_utils.get_cov_sub_block(cov, "t,t", fs)
is_uncorrelated = np.allclose(fs_mat, 0.0)

ff_mat = matrix_utils.get_cov_sub_block(cov, "t,t", ff)
ss_mat = matrix_utils.get_cov_sub_block(cov, "t,t", ss)
sf_mat = matrix_utils.get_cov_sub_block(cov, "t,t", sf)
combined = np.block([[ff_mat, fs_mat], [sf_mat, ss_mat]])
combined_min_eig = np.min(np.linalg.eigvals(combined))


a, b = 0, 0
for a in range(4):
    for b in range(4):
        test_array = np.array(
            [[ff_mat[a, a], fs_mat[a, b]], [sf_mat[b, a], ss_mat[b, b]]]
        )
        print(test_array)
        print(np.min(np.linalg.eigvals(test_array)))


columns_to_keep = [0, 1, 2, 3, 4, 5]
mode_i_vertices = my_grid.by_index(i).vertices
mode_j_vertices = my_grid.by_index(j).vertices
mode_u_vertices = my_grid.by_index(i).vertices
mode_v_vertices = my_grid.by_index(j).vertices

# Get the integration domain
cartesian_product_ij = geometry_utils.iterated_cartesian_product(
    [mode_i_vertices, mode_j_vertices, mode_u_vertices, mode_v_vertices]
)
reduced_intersection_ij = geometry_utils.get_intersection_vertices(cartesian_product_ij)
reduced_hull_ij = scipy.spatial.ConvexHull(reduced_intersection_ij[:, columns_to_keep], qhull_options="QJ")
centroid_ij = np.mean(reduced_intersection_ij, axis=0)
ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = centroid_ij.T[columns_to_keep]

kv_x = -ki_x + kj_x + ku_x
kv_y = -ki_y + kj_y + ku_y

ki_z = np.sqrt(1.0 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1.0 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1.0 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1.0 - kv_x**2 - kv_y**2)
A_ijij_ij = isotropic_sphere.get_A(
    np.array([ki_x]),
    np.array([ki_y]),
    np.array([ki_z]),
    np.array([kj_x]),
    np.array([kj_y]),
    np.array([kj_z]),
    np.array([2.0]),
    np.array([1.2]),
)
A_ijij_uv = isotropic_sphere.get_A(
    np.array([ku_x]),
    np.array([ku_y]),
    np.array([ku_z]),
    np.array([kv_x]),
    np.array([kv_y]),
    np.array([kv_z]),
    np.array([2.0]),
    np.array([1.2]),
)
#----------------------------------------------------------------
mode_i_vertices = my_grid.by_index(u).vertices
mode_j_vertices = my_grid.by_index(v).vertices
mode_u_vertices = my_grid.by_index(u).vertices
mode_v_vertices = my_grid.by_index(v).vertices

# Get the integration domain
cartesian_product_uv = geometry_utils.iterated_cartesian_product(
    [mode_i_vertices, mode_j_vertices, mode_u_vertices, mode_v_vertices]
)
reduced_intersection_uv = geometry_utils.get_intersection_vertices(cartesian_product_uv)
reduced_hull_uv = scipy.spatial.ConvexHull(reduced_intersection_uv[:, columns_to_keep], qhull_options="QJ")
centroid_uv = np.mean(reduced_intersection_uv, axis=0)
ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = centroid_uv.T[columns_to_keep]

kv_x = -ki_x + kj_x + ku_x
kv_y = -ki_y + kj_y + ku_y

ki_z = np.sqrt(1.0 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1.0 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1.0 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1.0 - kv_x**2 - kv_y**2)
A_uvuv_ij = isotropic_sphere.get_A(
    np.array([ki_x]),
    np.array([ki_y]),
    np.array([ki_z]),
    np.array([kj_x]),
    np.array([kj_y]),
    np.array([kj_z]),
    np.array([2.0]),
    np.array([1.2]),
)
A_uvuv_uv = isotropic_sphere.get_A(
    np.array([ku_x]),
    np.array([ku_y]),
    np.array([ku_z]),
    np.array([kv_x]),
    np.array([kv_y]),
    np.array([kv_z]),
    np.array([2.0]),
    np.array([1.2]),
)

#----------------------------------------------------------------

mode_i_vertices = my_grid.by_index(i).vertices
mode_j_vertices = my_grid.by_index(j).vertices
mode_u_vertices = my_grid.by_index(u).vertices
mode_v_vertices = my_grid.by_index(v).vertices

# Get the integration domain
cartesian_product_ijuv = geometry_utils.iterated_cartesian_product(
    [mode_i_vertices, mode_j_vertices, mode_u_vertices, mode_v_vertices]
)
reduced_intersection_ijuv = geometry_utils.get_intersection_vertices(cartesian_product_ijuv)
reduced_hull_ijuv = scipy.spatial.ConvexHull(reduced_intersection_ijuv[:, columns_to_keep], qhull_options="QJ")
centroid_ijuv = np.mean(reduced_intersection_ijuv, axis=0)
ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = centroid_ijuv.T[columns_to_keep]

kv_x = -ki_x + kj_x + ku_x
kv_y = -ki_y + kj_y + ku_y

ki_z = np.sqrt(1.0 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1.0 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1.0 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1.0 - kv_x**2 - kv_y**2)
A_ijuv_ij = isotropic_sphere.get_A(
    np.array([ki_x]),
    np.array([ki_y]),
    np.array([ki_z]),
    np.array([kj_x]),
    np.array([kj_y]),
    np.array([kj_z]),
    np.array([2.0]),
    np.array([1.2]),
)
A_ijuv_uv = isotropic_sphere.get_A(
    np.array([ku_x]),
    np.array([ku_y]),
    np.array([ku_z]),
    np.array([kv_x]),
    np.array([kv_y]),
    np.array([kv_z]),
    np.array([2.0]),
    np.array([1.2]),
)
