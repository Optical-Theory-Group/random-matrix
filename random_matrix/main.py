import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import shapely

from random_matrix.amplitude_matrix import isotropic_sphere
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
from random_matrix.utils import (
    array_utils,
    function_utils,
    geometry_utils,
    integration_utils,
    special_functions,
)
from random_matrix.scattering_matrix import sampler


print("Preparing Grid")
my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.8, 0.8),
    r_lim=2.0,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
# my_grid.plot()
print(my_grid.num_propagating)
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

print("Finding indices")
indices = input_statistics_manager._get_indices()
print("Preparing tasks")
tasks = input_statistics_manager._get_integration_tasks(indices)
print("Performing integrals")

start = time.perf_counter()
results = tasks.execute_tasks()
end = time.perf_counter()
print(f"Time taken: {end- start}")


# File path of the existing file
file_path = "example_file.txt"

# Open the file in append mode ('a')
with open(file_path, 'a') as file:
    # Append each new string to the file
    file.write(f"{end-start}")  # Adding a newline to separate lines

# mean_results = results.by_statistic_type("mean")
# cov_results = results.by_statistic_type("covariance")
# pseudo_cov_results = results.by_statistic_type("pseudo_covariance")

# mean_S = input_statistics_manager._get_mean_S(mean_results)
# cov = input_statistics_manager._get_covariance_matrix(cov_results)
# pseudo_cov = input_statistics_manager._get_pseudo_covariance_matrix(
#     pseudo_cov_results
# )
# sigma = 0.5 * scipy.sparse.bmat(
#     [
#         [np.real(cov + pseudo_cov), np.imag(-cov + pseudo_cov)],
#         [np.imag(cov + pseudo_cov), np.real(cov + -pseudo_cov)],
#     ]
# )

# chol = input_statistics_manager._get_chol(sigma)

# # Find the (0, 0, 0, 0) task
# for task in tasks.tasks[4].sub_block_locations:
#     if task[1] == (8, 9, 8, 9):
#         print("Found")
#         s = task[0]
# domain = tasks.tasks[4].domain_stack[s]

# for i, result in enumerate(results.results[4].sub_block_locations):
#     if result == (8, 9, 8, 9):
#         integral = results.results[4].integral[i]
#         print(np.diag(integral.reshape(4,4)))
#         break


# print("Calculating statistics")
# mean_S, sigma, cov = input_statistics_manager.get_statistics()
# print("Done")
# size_of_sig, _ = np.shape(sigma)
# eigs = scipy.sparse.linalg.eigs(sigma)


# plt.figure()
# plt.spy(cov)

# plt.figure()
# plt.spy(sigma)

# chol = input_statistics_manager._get_chol(sigma)
# plt.figure()
# plt.spy(chol)

# # Pure stats
# size_of_cov, _ = np.shape(cov)
# cov_tt = cov[
#     int(size_of_cov * 1 / 4) : int(size_of_cov * 1 / 2),
#     int(size_of_cov * 1 / 4) : int(size_of_cov * 1 / 2),
# ]

# indices = [
#     4 * my_grid.max_index + 4 * my_grid.num_propagating * i
#     for i in range(my_grid.num_propagating)
# ]
# intensity_array = []
# thetas = []
# for num, i in enumerate(indices):
#     H = cov_tt[4 * i : 4 * i + 4, 4 * i : 4 * i + 4].todense()
#     if not np.shape(H) == (4, 4):
#         continue

#     intensity_array.append(H[0, 0] + H[2, 2])

#     mode_index = num - my_grid.max_index
#     mode_center = my_grid.by_index(mode_index).center
#     cos = np.sqrt(1.0 - np.linalg.norm(mode_center) ** 2)
#     theta = np.arccos(cos) * 180 / np.pi
#     thetas.append(theta)
# fig, axis = plt.subplots()
# axis.scatter(thetas, intensity_array)


# # Post matrix generation
# n_mats = 10**3
# S_array = sampler.S_sampler(mean_S, chol, n_mats)
# size_of_S, _ = np.shape(S_array[:, :, 0])

# intensity_array = np.zeros(my_grid.num_propagating * 2)

# for i in range(n_mats):
#     S = S_array[:, :, i]
#     t = S[int(size_of_S / 2) : size_of_S, 0 : int(size_of_S / 2)]
#     t = t - np.identity(int(size_of_S / 2))

#     r = S[0 : int(size_of_S / 2), 0 : int(size_of_S / 2)]
#     t_col = t[:, 2 * my_grid.max_index : 2 * my_grid.max_index + 2]
#     r_col = r[:, 2 * my_grid.max_index : 2 * my_grid.max_index + 2]

#     thetas = []
#     for j in range(my_grid.num_propagating):
#         mode_index = j - my_grid.max_index

#         # t
#         intensity = np.linalg.norm(t_col[2 * j : 2 * j + 2, 0]) ** 2
#         intensity_array[j] += intensity
#         mode_center = my_grid.by_index(mode_index).center
#         cos = np.sqrt(1.0 - np.linalg.norm(mode_center) ** 2)
#         theta = np.arccos(cos) * 180 / np.pi
#         thetas.append(theta)

#         # r
#         intensity = np.linalg.norm(r_col[2 * j : 2 * j + 2, 0]) ** 2
#         intensity_array[j + 1] += intensity
#         mode_center = my_grid.by_index(mode_index).center
#         cos = np.sqrt(1.0 - np.linalg.norm(mode_center) ** 2)
#         theta = 180 - np.arccos(cos) * 180 / np.pi
#         thetas.append(theta)
# intensity_array /= n_mats

# fig, axis = plt.subplots()
# axis.scatter(thetas, intensity_array)
