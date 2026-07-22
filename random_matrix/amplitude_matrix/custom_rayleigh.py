from pathlib import Path
import sys
import os

# Add parent directory to Python path
project_root = Path("/home/sdutta/code/random-matrix/")  # <-- adjust this
sys.path.insert(0, str(project_root))


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
from random_matrix.input_statistics import matrix_pool_manager
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
from random_matrix.input_statistics.matrix_pool_manager import MatrixPoolManager
from random_matrix.scattering_matrix import sampler
import cupy as cp
from collections import defaultdict
import matplotlib

import h5py
from matplotlib import colors
from random_matrix.amplitude_matrix import isotropic_sphere as rm_iso


# ----------------------------------------------------------------
# Calculates scattering cross section for isotropic sphere
# ----------------------------------------------------------------
def get_scattering_cs(x, rri):
    wavelength = 550e-9
    n = 100  # number of samples
    rri = rri
    m = np.ravel(np.reshape(rri * np.ones((n * (n + 1))), (n + 1, n)))
    k = (2 * np.pi) / wavelength
    size_param = x
    radius = size_param / k
    # sampling incident field
    theta_i = (0) * np.ones((n))
    phi_i = 0 * np.ones((n + 1))
    theta_grid_i, phi_grid_i = np.meshgrid(theta_i, phi_i)

    # Incident field
    ki_z = np.ravel(np.cos(theta_grid_i))
    ki_x = np.ravel(np.sin(theta_grid_i) * np.cos(phi_grid_i))
    ki_y = np.ravel(np.sin(theta_grid_i) * np.sin(phi_grid_i))

    # # sampling scattered field
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n + 1)
    theta_grid_s, phi_grid_s = np.meshgrid(theta, phi)

    # Scattered field
    ks_z = np.ravel(np.cos(theta_grid_s))
    ks_x = np.ravel(np.sin(theta_grid_s) * np.cos(phi_grid_s))
    ks_y = np.ravel(np.sin(theta_grid_s) * np.sin(phi_grid_s))

    # Incident polarization
    Ex = 1
    Ey = 0
    x = np.ravel(np.reshape((size_param * np.ones((n * (n + 1)))), (n + 1, n)))
    A = rm_iso.get_A(ki_x, ki_y, ki_z, ks_x, ks_y, ks_z, x, m)
    A = np.reshape(A, (n + 1, n, 4))
    S1 = A[:, :, 3]
    S2 = A[:, :, 0]
    S3 = A[:, :, 1]
    S4 = A[:, :, 2]
    d_theta = np.pi / (n - 1)
    d_phi = np.pi / (n)
    T = (np.abs(S2 * Ex + S3 * Ey) ** 2 + np.abs(S4 * Ex + S1 * Ey) ** 2) * np.sin(
        theta_grid_s
    )
    # T = (np.abs(S2) ** 2 + np.abs(S4) ** 2) * np.sin(theta_grid_s) / k**2
    inner_integral = np.trapezoid(T, phi, d_phi, axis=0)
    C_scaT = np.trapezoid(inner_integral, theta, d_theta)
    return C_scaT


def get_A_HG_scatterer(
    mu: np.ndarray,
    x: float,
    m: complex,
):
    """
    Synthetic scattering A-matrix using Henyey-Greenstein phase function,
    normalized to match Mie scattering cross section.

    Parameters
    ----------
    mu : cos(theta) array
    x : size parameter
    m : relative refractive index
    g : HG anisotropy parameter
    theta0 : angular shift

    Returns
    -------
    A : (..., 4) array [S2, S3, S4, S1]
    """

    theta = mu
    g = 0
    # --- HG phase function ---
    numerator = 1 - g**2
    denominator = (1 + g**2 - 2 * g * np.cos(theta)) ** (3 / 2)
    p_theta = (1 / (4 * np.pi)) * (numerator / denominator)

    sigma_s = get_scattering_cs(x, m)

    # --- Construct synthetic amplitude ---
    S = np.sqrt(sigma_s * p_theta).astype(np.complex128)

    # --- Build A matrix ---
    S1 = S
    S2 = S
    S3 = np.zeros_like(S)
    S4 = np.zeros_like(S)

    A = np.stack([S2, S3, S4, S1], axis=-1)

    return A


def get_A(
    ki_x: np.ndarray | cp.ndarray,
    ki_y: np.ndarray | cp.ndarray,
    ki_z: np.ndarray | cp.ndarray,
    kj_x: np.ndarray | cp.ndarray,
    kj_y: np.ndarray | cp.ndarray,
    kj_z: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:

    length = len(ki_x)
    # Get geometric transformation matrices for converting between
    # different basis vectors
    T_i, T_j = scattering_geometry.get_transformation_matrices(
        ki_x, ki_y, ki_z, kj_x, kj_y, kj_z
    )

    # Get A in the scattering plane
    mu = ki_x * kj_x + ki_y * kj_y + ki_z * kj_z
    A_scattering_plane = get_A_HG_scatterer(mu, x, m)
    output = (T_j @ A_scattering_plane.reshape(length, 2, 2) @ T_i).reshape(length, 4)
    return output


get_A.particle_type = "custom_rayleigh"


def get_A_product_conj(
    ki_x: np.ndarray | cp.ndarray,
    ki_y: np.ndarray | cp.ndarray,
    ki_z: np.ndarray | cp.ndarray,
    kj_x: np.ndarray | cp.ndarray,
    kj_y: np.ndarray | cp.ndarray,
    kj_z: np.ndarray | cp.ndarray,
    ku_x: np.ndarray | cp.ndarray,
    ku_y: np.ndarray | cp.ndarray,
    ku_z: np.ndarray | cp.ndarray,
    kv_x: np.ndarray | cp.ndarray,
    kv_y: np.ndarray | cp.ndarray,
    kv_z: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    xp = array_utils.get_module(ki_x)
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    product = A_ij[:, :, xp.newaxis] * xp.conj(A_uv[:, xp.newaxis, :])
    output = np.reshape(product, (len(ki_x), 16))
    return output


def get_A_product(
    ki_x: np.ndarray | cp.ndarray,
    ki_y: np.ndarray | cp.ndarray,
    ki_z: np.ndarray | cp.ndarray,
    kj_x: np.ndarray | cp.ndarray,
    kj_y: np.ndarray | cp.ndarray,
    kj_z: np.ndarray | cp.ndarray,
    ku_x: np.ndarray | cp.ndarray,
    ku_y: np.ndarray | cp.ndarray,
    ku_z: np.ndarray | cp.ndarray,
    kv_x: np.ndarray | cp.ndarray,
    kv_y: np.ndarray | cp.ndarray,
    kv_z: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    xp = array_utils.get_module(ki_x)
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    product = A_ij[:, :, xp.newaxis] * A_uv[:, xp.newaxis, :]
    output = np.reshape(product, (*ki_x.shape, 16))
    return output
