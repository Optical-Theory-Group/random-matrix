import time

import numba
import numpy as np
import cupy as cp
import scipy
from random_matrix.utils import array_utils
from random_matrix.amplitude_matrix import scattering_geometry


# @numba.njit(fastmath=False, parallel=True)
def get_A_scattering_plane(
    mu: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Work out the A matrix defined with respect to the scattering plane.

    x and m can be scalars, but mu must be an array"""
    xp = array_utils.get_module(mu)

    input_shape = xp.shape(mu)
    S_2 = xp.zeros(input_shape, dtype=xp.complex128)
    S_1 = xp.zeros(input_shape, dtype=xp.complex128)
    S_3 = xp.zeros(input_shape, dtype=xp.complex128)
    S_4 = xp.zeros(input_shape, dtype=xp.complex128)

    # Get stopping index for sum (Wiscombe approximation)
    num_stop = int(xp.max(xp.floor(x + 4.05 * x**0.33333 + 2.0) + 1))

    # Initialise variables
    # 1 and 2 subscripts are for recurrence relations, 2 is ahead of 1
    jx_0 = xp.sin(x) / x
    jx_1 = xp.sin(x) / x**2 - xp.cos(x) / x
    jmx_0 = xp.sin(m * x) / (m * x)
    jmx_1 = xp.sin(m * x) / (m * x) ** 2 - xp.cos(m * x) / (m * x)

    djx = ((x**2.0 - 2.0) * xp.sin(x) + 2.0 * x * xp.cos(x)) / x**3.0
    djmx = (
        ((m * x) ** 2.0 - 2.0) * xp.sin(m * x) + 2.0 * m * x * xp.cos(m * x)
    ) / (m * x) ** 3

    yx_0 = -xp.cos(x) / x
    yx_1 = -xp.cos(x) / x**2 - xp.sin(x) / x
    ymx_0 = -xp.cos(m * x) / (m * x)
    ymx_1 = -xp.cos(m * x) / (m * x) ** 2 - xp.sin(m * x) / (m * x)

    dyx = (2.0 * x * xp.sin(x) - (x**2 - 2.0) * xp.cos(x)) / x**3

    psix = x * jx_1
    psimx = m * x * jmx_1
    dpsix = x * djx + jx_1
    dpsimx = m * x * djmx + jmx_1
    phix = x * yx_1
    dphix = x * dyx + yx_1
    xix = psix + 1j * phix
    dxix = dpsix + 1j * dphix

    pi_0 = xp.zeros(input_shape, dtype=xp.float64)
    pi_1 = xp.ones(input_shape, dtype=xp.float64)

    tau = mu

    # Initialise S matrix terms. Note that n=1 corresponds to the _2 variables
    a = (m * psimx * dpsix - psix * dpsimx) / (m * psimx * dxix - xix * dpsimx)
    b = (psimx * dpsix - m * psix * dpsimx) / (psimx * dxix - m * xix * dpsimx)

    S_1 = 3.0 / 2.0 * (a * pi_1 + b * tau)
    S_2 = 3.0 / 2.0 * (a * tau + b * pi_1)

    for n in range(2, num_stop + 1):
        # Update all variables with recurrence relations
        # Bessel functions
        new_jx = (2.0 * n - 1.0) / x * jx_1 - jx_0
        jx_0 = jx_1
        jx_1 = new_jx
        new_jmx = (2.0 * n - 1.0) / (m * x) * jmx_1 - jmx_0
        jmx_0 = jmx_1
        jmx_1 = new_jmx
        new_yx = (2.0 * n - 1.0) / x * yx_1 - yx_0
        yx_0 = yx_1
        yx_1 = new_yx
        new_ymx = (2.0 * n - 1.0) / (m * x) * ymx_1 - ymx_0
        ymx_0 = ymx_1
        ymx_1 = new_ymx

        # Derivatives of bessel funtions
        new_djx = -(n + 1) / x * jx_1 + jx_0
        new_djmx = -(n + 1) / (m * x) * jmx_1 + jmx_0
        new_dyx = -(n + 1) / x * yx_1 + yx_0

        new_psix = x * new_jx
        new_psimx = m * x * new_jmx
        new_dpsix = x * new_djx + new_jx
        new_dpsimx = m * x * new_djmx + new_jmx
        new_phix = x * new_yx
        new_dphix = x * new_dyx + new_yx
        new_xix = new_psix + 1j * new_phix
        new_dxix = new_dpsix + 1j * new_dphix

        # Angular functions
        new_pi = (2.0 * n - 1.0) / (n - 1.0) * mu * pi_1 - n / (n - 1.0) * pi_0
        pi_0 = pi_1
        pi_1 = new_pi

        new_tau = n * mu * new_pi - (n + 1.0) * pi_0

        # Calculate new S terms
        a = (m * new_psimx * new_dpsix - new_psix * new_dpsimx) / (
            m * new_psimx * new_dxix - new_xix * new_dpsimx
        )
        b = (new_psimx * new_dpsix - m * new_psix * new_dpsimx) / (
            new_psimx * new_dxix - m * new_xix * new_dpsimx
        )

        S_1 = S_1 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_pi + b * new_tau
        )
        S_2 = S_2 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_tau + b * new_pi
        )

    A = np.stack([S_2, S_3, S_4, S_1], axis=-1)
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
    A_scattering_plane = get_A_scattering_plane(mu, x, m)
    output = (T_j @ A_scattering_plane.reshape(length, 2, 2) @ T_i).reshape(
        length, 4
    )
    return output


# @numba.njit(fastmath=True, parallel=True)
# def get_A(
#     ki_x: np.ndarray | cp.ndarray,
#     ki_y: np.ndarray | cp.ndarray,
#     ki_z: np.ndarray | cp.ndarray,
#     kj_x: np.ndarray | cp.ndarray,
#     kj_y: np.ndarray | cp.ndarray,
#     kj_z: np.ndarray | cp.ndarray,
#     x: np.ndarray | cp.ndarray,
#     m: np.ndarray | cp.ndarray,
# ) -> np.ndarray | cp.ndarray:
#     xs = x
#     ms = m
#     mus = ki_x * kj_x + ki_y * kj_y + ki_z * kj_z

#     # Array sizes for reshaping
#     num_one, num_two = np.shape(ki_x)
#     num_linear = num_one * num_two

#     theta_i = np.arccos(ki_z)
#     phi_i = np.arctan2(np.real(ki_y), np.real(ki_x))

#     # Check for (0,0,-1) vector, which is a special case
#     indices = np.where(
#         np.logical_and(
#             np.logical_and(np.isclose(ki_x, 0.0), np.isclose(ki_y, 0.0)),
#             np.isclose(ki_z, -1.0),
#         )
#     )
#     e_theta_i = np.array(
#         [
#             np.cos(theta_i) * np.cos(phi_i),
#             np.cos(theta_i) * np.sin(phi_i),
#             -np.sin(theta_i),
#         ]
#     )
#     e_theta_i = np.transpose(e_theta_i, (1, 2, 0))
#     e_theta_i[indices] = np.array([1.0, 0.0, 0.0])

#     e_phi_i = np.array(
#         [-np.sin(phi_i), np.cos(phi_i), np.zeros(np.shape(ki_x))]
#     )
#     e_phi_i = np.transpose(e_phi_i, (1, 2, 0))
#     e_phi_i[indices] = np.array([0, -1, 0])

#     # Same for the scattered wavevector
#     theta_s = np.arccos(kj_z)
#     phi_s = np.arctan2(np.real(kj_y), np.real(kj_x))

#     indices = np.where(
#         np.logical_and(
#             np.logical_and(np.isclose(kj_x, 0.0), np.isclose(kj_y, 0.0)),
#             np.isclose(kj_z, -1.0),
#         )
#     )
#     e_theta_s = np.array(
#         [
#             np.cos(theta_s) * np.cos(phi_s),
#             np.cos(theta_s) * np.sin(phi_s),
#             -np.sin(theta_s),
#         ]
#     )
#     e_theta_s = np.transpose(e_theta_s, (1, 2, 0))
#     e_theta_s[indices] = np.array([1.0, 0.0, 0.0])

#     e_phi_s = np.array(
#         [-np.sin(phi_s), np.cos(phi_s), np.zeros(np.shape(ki_x))]
#     )
#     e_phi_s = np.transpose(e_phi_s, (1, 2, 0))
#     e_phi_s[indices] = np.array([0, -1, 0])

#     # Get special case indices for par and per vectors
#     are_close_x = np.isclose(ki_x, kj_x)
#     are_close_y = np.isclose(ki_y, kj_y)
#     are_close_z = np.isclose(ki_z, kj_z)
#     parallel_indices = np.logical_and(
#         are_close_x, np.logical_and(are_close_y, are_close_z)
#     )
#     are_close_x = np.isclose(ki_x, -kj_x)
#     are_close_y = np.isclose(ki_y, -kj_y)
#     are_close_z = np.isclose(ki_z, -kj_z)
#     antiparallel_indices = np.logical_and(
#         are_close_x, np.logical_and(are_close_y, are_close_z)
#     )

#     kis = np.zeros((num_one, num_two, 3))
#     kis[:, :, 0] = ki_x
#     kis[:, :, 1] = ki_y
#     kis[:, :, 2] = ki_z
#     kis = np.reshape(kis, (num_linear, 3))

#     kjs = np.zeros((num_one, num_two, 3))
#     kjs[:, :, 0] = kj_x
#     kjs[:, :, 1] = kj_y
#     kjs[:, :, 2] = kj_z
#     kjs = np.reshape(kjs, (num_linear, 3))

#     e_per = np.cross(kis, kjs)
#     norms = np.linalg.norm(e_per, axis=-1)
#     e_per = e_per / norms[:, np.newaxis]

#     e_par_i = np.cross(e_per, kis)
#     norms = np.linalg.norm(e_par_i, axis=-1)
#     e_par_i = e_par_i / norms[:, np.newaxis]
#     e_par_i = np.reshape(e_par_i, (num_one, num_two, 3))

#     e_par_s = np.cross(e_per, kjs)
#     norms = np.linalg.norm(e_par_s, axis=-1)
#     e_par_s = e_par_s / norms[:, np.newaxis]
#     e_par_s = np.reshape(e_par_s, (num_one, num_two, 3))

#     e_per = np.reshape(e_per, (num_one, num_two, 3))
#     kis = np.reshape(kis, (num_one, num_two, 3))
#     kjs = np.reshape(kjs, (num_one, num_two, 3))

#     # Work out rotation angles between coordinate basis vectors and scattering
#     # plane vectors
#     # i case
#     cos_i = (
#         e_theta_i[:, :, 0] * e_par_i[:, :, 0]
#         + e_theta_i[:, :, 1] * e_par_i[:, :, 1]
#         + e_theta_i[:, :, 2] * e_par_i[:, :, 2]
#     )

#     cp = np.cross(
#         np.reshape(e_theta_i, (num_linear, 3)),
#         np.reshape(e_par_i, (num_linear, 3)),
#     ).reshape((num_one, num_two, 3))

#     norms = np.linalg.norm(cp, axis=-1)
#     cp = cp / norms[:, :, np.newaxis]

#     cos_i = np.where(
#         np.logical_or(np.isclose(cos_i, 1.0), cos_i > 1.0), 1.0, cos_i
#     )
#     cos_i = np.where(
#         np.logical_or(np.isclose(cos_i, -1.0), cos_i < -1.0), -1.0, cos_i
#     )
#     theta = np.arccos(cos_i)

#     # cp, ki
#     alignment = (
#         cp[:, :, 0] * kis[:, :, 0]
#         + cp[:, :, 1] * kis[:, :, 1]
#         + cp[:, :, 2] * kis[:, :, 2]
#     )

#     alpha_i = np.where(np.isclose(alignment, 1.0), theta, -theta)

#     # same for j
#     cos_j = (
#         e_theta_s[:, :, 0] * e_par_s[:, :, 0]
#         + e_theta_s[:, :, 1] * e_par_s[:, :, 1]
#         + e_theta_s[:, :, 2] * e_par_s[:, :, 2]
#     )

#     cp = np.cross(
#         np.reshape(e_par_s, (num_linear, 3)),
#         np.reshape(e_theta_s, (num_linear, 3)),
#     ).reshape((num_one, num_two, 3))
#     norms = np.linalg.norm(cp, axis=-1)
#     cp = cp / norms[:, :, np.newaxis]

#     cos_j = np.where(
#         np.logical_or(np.isclose(cos_j, 1.0), cos_j > 1.0), 1.0, cos_j
#     )
#     cos_j = np.where(
#         np.logical_or(np.isclose(cos_j, -1.0), cos_j < -1.0), -1.0, cos_j
#     )
#     theta = np.arccos(cos_j)

#     # cp, ki
#     alignment = (
#         cp[:, :, 0] * kjs[:, :, 0]
#         + cp[:, :, 1] * kjs[:, :, 1]
#         + cp[:, :, 2] * kjs[:, :, 2]
#     )

#     alpha_j = np.where(np.isclose(alignment, 1.0), theta, -theta)

#     # Special cases
#     e_par_i[parallel_indices] = e_theta_i[parallel_indices]
#     e_par_s[parallel_indices] = e_theta_s[parallel_indices]
#     e_per[parallel_indices] = e_phi_i[parallel_indices]
#     alpha_i[parallel_indices] = 0.0
#     alpha_j[parallel_indices] = 0.0

#     e_par_i[antiparallel_indices] = e_theta_i[antiparallel_indices]
#     e_par_s[antiparallel_indices] = -e_theta_s[antiparallel_indices]
#     e_per[antiparallel_indices] = e_phi_i[antiparallel_indices]
#     alpha_i[antiparallel_indices] = 0.0
#     alpha_j[antiparallel_indices] = np.pi

#     T_i = np.zeros((num_one, num_two, 2, 2))
#     T_i[:, :, 0, 0] = np.cos(alpha_i)
#     T_i[:, :, 0, 1] = np.sin(alpha_i)
#     T_i[:, :, 1, 0] = np.sin(alpha_i)
#     T_i[:, :, 1, 1] = -np.cos(alpha_i)

#     T_j = np.zeros((num_one, num_two, 2, 2))
#     T_j[:, :, 0, 0] = np.cos(alpha_j)
#     T_j[:, :, 0, 1] = -np.sin(alpha_j)
#     T_j[:, :, 1, 0] = -np.sin(alpha_j)
#     T_j[:, :, 1, 1] = -np.cos(alpha_j)

#     A_scattering_plane = np.transpose(
#         get_A_scattering_plane(mus, xs, ms).reshape(2, 2, num_one, num_two),
#         (2, 3, 0, 1),
#     )

#     final = (
#         (np.reshape(T_j, (num_linear, 2, 2)))
#         @ (np.reshape(A_scattering_plane, (num_linear, 2, 2)))
#         @ (np.reshape(T_i, (num_linear, 2, 2)))
#     )
#     final = -np.transpose(np.reshape(final, (num_one, num_two, 4)), (2, 0, 1))

#     return final


get_A.particle_type = "sphere"


# # @numba.njit(fastmath=True, parallel=True)
# def get_A_product_conj(
#     ki_x: np.ndarray | cp.ndarray,
#     ki_y: np.ndarray | cp.ndarray,
#     ki_z: np.ndarray | cp.ndarray,
#     kj_x: np.ndarray | cp.ndarray,
#     kj_y: np.ndarray | cp.ndarray,
#     kj_z: np.ndarray | cp.ndarray,
#     ku_x: np.ndarray | cp.ndarray,
#     ku_y: np.ndarray | cp.ndarray,
#     ku_z: np.ndarray | cp.ndarray,
#     kv_x: np.ndarray | cp.ndarray,
#     kv_y: np.ndarray | cp.ndarray,
#     kv_z: np.ndarray | cp.ndarray,
#     x: np.ndarray | cp.ndarray,
#     m: np.ndarray | cp.ndarray,
# ) -> np.ndarray | cp.ndarray:
#     xp = array_utils.get_module(ki_x)
#     A_ij = get_A_new(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
#     A_uv_new = get_A_new(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
#     return (
#         A_ij[:, xp.newaxis, :, :] * xp.conj(A_uv[xp.newaxis, :, :, :])
#     ).reshape(16, *A_ij.shape[1:])


# @numba.njit(fastmath=True, parallel=True)
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


# @numba.njit(fastmath=True, parallel=True)
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


if __name__ == "__main__":
    mu = np.array([0.8])
    x = 1
    m = 1.3
    # A = get_A_scattering_plane(mu,x,m)
    # print(A)

    ki_x = np.array([1.0])
    ki_y = np.array([0.0])
    ki_z = np.array([0.0])
    kj_x = np.array([0.0])
    kj_y = np.array([1.0])
    kj_z = np.array([0.0])
    A = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    print(A)

# # Test
# num_one = 1000
# num_two = 8
# x = 1.0 * np.ones((num_one, num_two))
# m = 1.2 * np.ones((num_one, num_two))

# ks = np.random.randn(num_one, num_two, 3)
# norms = np.linalg.norm(ks, axis=2)
# ks = ks / norms[:, :, np.newaxis]
# ki_x = ks[:, :, 0]
# ki_y = ks[:, :, 1]
# ki_z = ks[:, :, 2]

# ks = np.random.randn(num_one, num_two, 3)
# norms = np.linalg.norm(ks, axis=2)
# ks = ks / norms[:, :, np.newaxis]
# kj_x = ks[:, :, 0]
# kj_y = ks[:, :, 1]
# kj_z = ks[:, :, 2]

# # ks = np.random.randn(num_one, num_two, 3)
# # norms = np.linalg.norm(ks, axis=2)
# # ks = ks / norms[:, :, np.newaxis]
# # ku_x = ks[:, :, 0]
# # ku_y = ks[:, :, 1]
# # ku_z = ks[:, :, 2]

# # ks = np.random.randn(num_one, num_two, 3)
# # norms = np.linalg.norm(ks, axis=2)
# # ks = ks / norms[:, :, np.newaxis]
# # kv_x = ks[:, :, 0]
# # kv_y = ks[:, :, 1]
# # kv_z = ks[:, :, 2]


# A = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
# A = get_A_product(
#     ki_x,
#     ki_y,
#     ki_z,
#     kj_x,
#     kj_y,
#     kj_z,
#     ku_x,
#     ku_y,
#     ku_z,
#     kv_x,
#     kv_y,
#     kv_z,
#     x,
#     m,
# )

# x = np.array([[1]])
# m = np.array([[1.2]])
# ki_x = np.array([[0]])
# ki_y = np.array([[0]])
# ki_z = np.array([[1]])

# kj_x = np.array([[0]])
# kj_y = np.array([[-0.8575]])
# kj_z = np.array([[np.sqrt(1-0.8575**2)]])
# A = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
