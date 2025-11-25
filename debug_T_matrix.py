import random_matrix.amplitude_matrix.isotropic_tmatrix as isotropic_tmatrix
from scipy.special import hankel1, spherical_jn, lpmv, h1vp, spherical_yn
import numpy as np
import math
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pickle
import random_matrix.amplitude_matrix.scattering_geometry as scattering_geometry
import plotly.graph_objects as go
from scipy.integrate import quad


def alph_mn_mpnp(m, n, mpr, npr):

    return (
        (-1) ** (m + mpr)
        * (1j) ** (npr - n - 1)
        * np.sqrt(((2 * n + 1) * (2 * npr + 1) / (n * (n + 1) * npr * (npr + 1))))
    )


def h_n(n, kr):
    return spherical_jn(n, kr) + 1j * spherical_yn(n, kr)


def dh_n(n, x):
    """Computes the derivative of the spherical Hankel function of the first kind."""
    hn = h_n(n, x)
    hn1 = h_n(n + 1, x)
    return (n / x) * hn - hn1


def b_n(n, m, x):  # here m is relative refractive index
    psi_n_x = x * spherical_jn(n, x)
    psi_n_mx = m * x * spherical_jn(n, m * x)
    psi_n_x_derivative = x * spherical_jn(n, x, derivative=True) + spherical_jn(n, x)
    psi_n_mx_derivative = m * x * spherical_jn(
        n, m * x, derivative=True
    ) + spherical_jn(n, m * x)
    xi_n_x = x * h_n(n, x)
    xi_n_x_derivative = x * dh_n(n, x) + h_n(n, x)

    num = m * psi_n_x * psi_n_mx_derivative - psi_n_mx * psi_n_x_derivative
    den = m * xi_n_x * psi_n_mx_derivative - psi_n_mx * xi_n_x_derivative

    return num / den


def a_n(n, m, x):
    psi_n_x = x * spherical_jn(n, x)
    psi_n_mx = m * x * spherical_jn(n, m * x)
    psi_n_x_derivative = x * spherical_jn(n, x, derivative=True) + spherical_jn(n, x)
    psi_n_mx_derivative = m * x * spherical_jn(
        n, m * x, derivative=True
    ) + spherical_jn(n, m * x)
    xi_n_x = x * h_n(n, x)
    xi_n_x_derivative = x * dh_n(n, x) + h_n(n, x)

    num = psi_n_x * psi_n_mx_derivative - m * psi_n_mx * psi_n_x_derivative
    den = xi_n_x * psi_n_mx_derivative - m * psi_n_mx * xi_n_x_derivative

    return num / den


def a_n_from_T(n, T_22):
    return (
        -1j
        * n
        * (n + 1)
        / (2 * n + 1)
        * alph_mn_mpnp(0, n, 0, n)
        * get_T_element(T_22, 0, n, 0, n)
    )


def b_n_from_T(n, T_11):
    return (
        -1j
        * n
        * (n + 1)
        / (2 * n + 1)
        * alph_mn_mpnp(0, n, 0, n)
        * get_T_element(T_11, 0, n, 0, n)
    )


wavelength1 = 400e-9
k1 = (2 * np.pi) / wavelength1
m = 1.2  # Relative refractive Index
k2 = m * k1
n_max = 5
size_param = 3
hull_uniform = isotropic_tmatrix.create_hull_uniform(size_param, k1)
T_matrix = isotropic_tmatrix.get_T(hull_uniform, k1, k2, n_max)

# sampling incident field
n = 100
theta_i = 0 * np.ones((100))
phi_i = 0 * np.ones((101))
theta_grid_i, phi_grid_i = np.meshgrid(theta_i, phi_i)
# Incident field
ki_z = np.ravel(np.cos(theta_grid_i))
ki_x = np.ravel(np.sin(theta_grid_i) * np.cos(phi_grid_i))
ki_y = np.ravel(np.sin(theta_grid_i) * np.sin(phi_grid_i))
# sampling scattered field
theta_s = np.linspace(0, np.pi, n)
phi_s = np.linspace(0, 2 * np.pi, n + 1)
# theta = 0 * np.ones((100))
# phi = 0* np.ones((101))
theta_grid_s, phi_grid_s = np.meshgrid(theta_s, phi_s)
# Scattered field
ks_z = np.ravel(np.cos(theta_grid_s))
ks_x = np.ravel(np.sin(theta_grid_s) * np.cos(phi_grid_s))
ks_y = np.ravel(np.sin(theta_grid_s) * np.sin(phi_grid_s))
d_theta = np.pi / (n - 1)
d_phi = 2 * np.pi / (n)


A = isotropic_tmatrix.scattering_amplitudes_from_T_v2(
    theta_grid_i, phi_grid_i, theta_grid_s, phi_grid_s, T_matrix, k1, n_max
)

output_file = "T_debug_datav2.pkl"
results = {
    "T": T_matrix,
    "A": A,
}

with open(output_file, "wb") as f:
    pickle.dump(results, f)


def get_T_element(T, m_row, n_row, m_col, n_col):
    """
    T is a square matrix where rows and columns are indexed as:
    For each n = n_min, n_min+1, ...:
        m runs from -n to +n

    This returns T[(m_row,n_row), (m_col,n_col)].
    """

    # Compute flat index for (m,n)
    def idx(m, n):
        # count elements in all previous n-levels
        prev_count = 0
        for k in range(1, n):
            prev_count += 2 * k + 1  # number of m-values at level k

        # position within the current n-level
        offset = m + n  # since m runs from -n..+n
        return prev_count + offset

    i = idx(m_row, n_row)
    j = idx(m_col, n_col)
    return T[i, j]


def get_T_sliced(T, n_max):
    n_limit = len(T) // 2

    T_11 = T[0:n_limit, 0:n_limit]
    T_12 = T[0:n_limit, n_limit : 2 * n_limit]
    T_21 = T[n_limit : 2 * n_limit, 0:n_limit]
    T_22 = T[n_limit : 2 * n_limit, n_limit : 2 * n_limit]
    size = 0
    for n in range(1, n_max + 1):
        size += 2 * n + 1
    T11_sliced = T_11[0:size, 0:size]
    T12_sliced = T_12[0:size, 0:size]
    T21_sliced = T_21[0:size, 0:size]
    T22_sliced = T_22[0:size, 0:size]
    T_sliced = np.block([[T11_sliced, T12_sliced], [T21_sliced, T22_sliced]])
    return T_sliced


# input_pol = 0 # 0 for x-polarized, pi/2 for y-polarized
# A = isotropic_tmatrix.scattering_amplitudes_from_T_v2(theta_grid_i, phi_grid_i, theta_grid, phi_grid, T_matrix, k1, 2)
# Ex = np.cos(input_pol)S
# Ey = np.sin(input_pol)

# S1 = np.reshape(A[:,0], (n + 1, n))
# S2 = np.reshape(A[:,1], (n + 1, n))
# S3 = np.reshape(A[:,2], (n + 1, n))
# S4 = np.reshape(A[:,3], (n + 1, n))
# Es_theta = S2 * Ex + S3 * Ey
# Es_phi = S4 * Ex + S1 * Ey
# Is = (np.abs(Es_theta) ** 2 + np.abs(Es_phi) ** 2) * np.sin(theta_grid) / k1**2
# inner_integral = np.trapezoid(Is, phi_s, d_phi, axis=0)
# Csca_ar= np.trapezoid(inner_integral, theta_s, d_theta)
