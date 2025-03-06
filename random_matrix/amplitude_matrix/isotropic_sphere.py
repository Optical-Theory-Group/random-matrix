import time

import numba
import numpy as np
import scipy

from random_matrix.utils.types import Numeric


# @numba.njit(fastmath=False, parallel=True)
def get_A_from_mus(mus: Numeric, xs: Numeric, ms: Numeric) -> Numeric:
    input_shape = np.shape(mus)

    S_2 = np.zeros(input_shape, dtype=np.complex128)
    S_1 = np.zeros(input_shape, dtype=np.complex128)
    S_3 = np.zeros(input_shape, dtype=np.complex128)
    S_4 = np.zeros(input_shape, dtype=np.complex128)

    # Get stopping index for sum
    if isinstance(xs, np.float64):
        num_stop = int(np.floor(xs + 4.05 * xs**0.33333 + 2.0) + 1)
    else:
        num_stop = int(np.max(np.floor(xs + 4.05 * xs**0.33333 + 2.0) + 1))

    # Initialise variables
    # 1 and 2 subscripts are for recurrence relations, 2 is ahead of 1
    jx_0 = np.sin(xs) / xs
    jx_1 = np.sin(xs) / xs**2 - np.cos(xs) / xs
    jmx_0 = np.sin(ms * xs) / (ms * xs)
    jmx_1 = np.sin(ms * xs) / (ms * xs) ** 2 - np.cos(ms * xs) / (ms * xs)

    djx = ((xs**2.0 - 2.0) * np.sin(xs) + 2.0 * xs * np.cos(xs)) / xs**3.0
    djmx = (
        ((ms * xs) ** 2.0 - 2.0) * np.sin(ms * xs) + 2.0 * ms * xs * np.cos(ms * xs)
    ) / (ms * xs) ** 3

    yx_0 = -np.cos(xs) / xs
    yx_1 = -np.cos(xs) / xs**2 - np.sin(xs) / xs
    ymx_0 = -np.cos(ms * xs) / (ms * xs)
    ymx_1 = -np.cos(ms * xs) / (ms * xs) ** 2 - np.sin(ms * xs) / (ms * xs)

    dyx = (2.0 * xs * np.sin(xs) - (xs**2 - 2.0) * np.cos(xs)) / xs**3

    psix = xs * jx_1
    psimx = ms * xs * jmx_1
    dpsix = xs * djx + jx_1
    dpsimx = ms * xs * djmx + jmx_1
    phix = xs * yx_1
    dphix = xs * dyx + yx_1
    xix = psix + 1j * phix
    dxix = dpsix + 1j * dphix

    pi_0 = np.zeros(input_shape, dtype=np.float64)
    pi_1 = np.ones(input_shape, dtype=np.float64)

    tau = mus

    # Initialise S matrix terms. Note that n=1 corresponds to the _2 variables
    a = (ms * psimx * dpsix - psix * dpsimx) / (ms * psimx * dxix - xix * dpsimx)
    b = (psimx * dpsix - ms * psix * dpsimx) / (psimx * dxix - ms * xix * dpsimx)

    S_1 = 3.0 / 2.0 * (a * pi_1 + b * tau)
    S_2 = 3.0 / 2.0 * (a * tau + b * pi_1)

    for n in range(2, num_stop + 1):
        # Update all variables with recurrence relations
        # Bessel functions
        new_jx = (2.0 * n - 1.0) / xs * jx_1 - jx_0
        jx_0 = jx_1
        jx_1 = new_jx
        new_jmx = (2.0 * n - 1.0) / (ms * xs) * jmx_1 - jmx_0
        jmx_0 = jmx_1
        jmx_1 = new_jmx
        new_yx = (2.0 * n - 1.0) / xs * yx_1 - yx_0
        yx_0 = yx_1
        yx_1 = new_yx
        new_ymx = (2.0 * n - 1.0) / (ms * xs) * ymx_1 - ymx_0
        ymx_0 = ymx_1
        ymx_1 = new_ymx

        # Derivatives of bessel funtions
        new_djx = -(n + 1) / xs * jx_1 + jx_0
        new_djmx = -(n + 1) / (ms * xs) * jmx_1 + jmx_0
        new_dyx = -(n + 1) / xs * yx_1 + yx_0

        new_psix = xs * new_jx
        new_psimx = ms * xs * new_jmx
        new_dpsix = xs * new_djx + new_jx
        new_dpsimx = ms * xs * new_djmx + new_jmx
        new_phix = xs * new_yx
        new_dphix = xs * new_dyx + new_yx
        new_xix = new_psix + 1j * new_phix
        new_dxix = new_dpsix + 1j * new_dphix

        # Angular functions
        new_pi = (2.0 * n - 1.0) / (n - 1.0) * mus * pi_1 - n / (n - 1.0) * pi_0
        pi_0 = pi_1
        pi_1 = new_pi

        new_tau = n * mus * new_pi - (n + 1.0) * pi_0

        # Calculate new S terms
        a = (ms * new_psimx * new_dpsix - new_psix * new_dpsimx) / (
            ms * new_psimx * new_dxix - new_xix * new_dpsimx
        )
        b = (new_psimx * new_dpsix - ms * new_psix * new_dpsimx) / (
            new_psimx * new_dxix - ms * new_xix * new_dpsimx
        )

        S_1 = S_1 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (a * new_pi + b * new_tau)
        S_2 = S_2 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (a * new_tau + b * new_pi)

    combined_array = np.concatenate(
        (
            S_2[np.newaxis, :],
            S_3[np.newaxis, :],
            S_4[np.newaxis, :],
            S_1[np.newaxis, :],
        )
    )

    return combined_array


# @numba.njit(fastmath=True, parallel=True)
def get_A(
    ki_x: Numeric,
    ki_y: Numeric,
    ki_z: Numeric,
    kj_x: Numeric,
    kj_y: Numeric,
    kj_z: Numeric,
    x: Numeric,
    m: Numeric,
) -> Numeric:
    xs = x
    ms = m
    mus = ki_x * kj_x + ki_y * kj_y + ki_z * kj_z

    # Array sizes for reshaping
    num_one, num_two = np.shape(ki_x)
    num_linear = num_one * num_two

    theta_i = np.arccos(ki_z)
    phi_i = np.arctan2(np.real(ki_y), np.real(ki_x))

    # Check for (0,0,-1) vector, which is a special case
    indices = np.where(
        np.logical_and(
            np.logical_and(np.isclose(ki_x, 0.0), np.isclose(ki_y, 0.0)),
            np.isclose(ki_z, -1.0),
        )
    )
    e_theta_i = np.array(
        [
            np.cos(theta_i) * np.cos(phi_i),
            np.cos(theta_i) * np.sin(phi_i),
            -np.sin(theta_i),
        ]
    )
    e_theta_i = np.transpose(e_theta_i, (1, 2, 0))
    e_theta_i[indices] = np.array([1.0, 0.0, 0.0])

    e_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i), np.zeros(np.shape(ki_x))])
    e_phi_i = np.transpose(e_phi_i, (1, 2, 0))
    e_phi_i[indices] = np.array([0, -1, 0])

    # Same for the scattered wavevector
    theta_s = np.arccos(kj_z)
    phi_s = np.arctan2(np.real(kj_y), np.real(kj_x))

    indices = np.where(
        np.logical_and(
            np.logical_and(np.isclose(kj_x, 0.0), np.isclose(kj_y, 0.0)),
            np.isclose(kj_z, -1.0),
        )
    )
    e_theta_s = np.array(
        [
            np.cos(theta_s) * np.cos(phi_s),
            np.cos(theta_s) * np.sin(phi_s),
            -np.sin(theta_s),
        ]
    )
    e_theta_s = np.transpose(e_theta_s, (1, 2, 0))
    e_theta_s[indices] = np.array([1.0, 0.0, 0.0])

    e_phi_s = np.array([-np.sin(phi_s), np.cos(phi_s), np.zeros(np.shape(ki_x))])
    e_phi_s = np.transpose(e_phi_s, (1, 2, 0))
    e_phi_s[indices] = np.array([0, -1, 0])

    # Get special case indices for par and per vectors
    are_close_x = np.isclose(ki_x, kj_x)
    are_close_y = np.isclose(ki_y, kj_y)
    are_close_z = np.isclose(ki_z, kj_z)
    parallel_indices = np.logical_and(
        are_close_x, np.logical_and(are_close_y, are_close_z)
    )
    are_close_x = np.isclose(ki_x, -kj_x)
    are_close_y = np.isclose(ki_y, -kj_y)
    are_close_z = np.isclose(ki_z, -kj_z)
    antiparallel_indices = np.logical_and(
        are_close_x, np.logical_and(are_close_y, are_close_z)
    )

    kis = np.zeros((num_one, num_two, 3))
    kis[:, :, 0] = ki_x
    kis[:, :, 1] = ki_y
    kis[:, :, 2] = ki_z
    kis = np.reshape(kis, (num_linear, 3))

    kjs = np.zeros((num_one, num_two, 3))
    kjs[:, :, 0] = kj_x
    kjs[:, :, 1] = kj_y
    kjs[:, :, 2] = kj_z
    kjs = np.reshape(kjs, (num_linear, 3))

    e_per = np.cross(kis, kjs)
    norms = np.linalg.norm(e_per, axis=-1)
    e_per = e_per / norms[:, np.newaxis]

    e_par_i = np.cross(e_per, kis)
    norms = np.linalg.norm(e_par_i, axis=-1)
    e_par_i = e_par_i / norms[:, np.newaxis]
    e_par_i = np.reshape(e_par_i, (num_one, num_two, 3))

    e_par_s = np.cross(e_per, kjs)
    norms = np.linalg.norm(e_par_s, axis=-1)
    e_par_s = e_par_s / norms[:, np.newaxis]
    e_par_s = np.reshape(e_par_s, (num_one, num_two, 3))

    e_per = np.reshape(e_per, (num_one, num_two, 3))
    kis = np.reshape(kis, (num_one, num_two, 3))
    kjs = np.reshape(kjs, (num_one, num_two, 3))

    # Work out rotation angles between coordinate basis vectors and scattering
    # plane vectors
    # i case
    cos_i = (
        e_theta_i[:, :, 0] * e_par_i[:, :, 0]
        + e_theta_i[:, :, 1] * e_par_i[:, :, 1]
        + e_theta_i[:, :, 2] * e_par_i[:, :, 2]
    )

    cp = np.cross(
        np.reshape(e_theta_i, (num_linear, 3)),
        np.reshape(e_par_i, (num_linear, 3)),
    ).reshape((num_one, num_two, 3))

    norms = np.linalg.norm(cp, axis=-1)
    cp = cp / norms[:, :, np.newaxis]

    cos_i = np.where(np.logical_or(np.isclose(cos_i, 1.0), cos_i > 1.0), 1.0, cos_i)
    cos_i = np.where(np.logical_or(np.isclose(cos_i, -1.0), cos_i < -1.0), -1.0, cos_i)
    theta = np.arccos(cos_i)

    # cp, ki
    alignment = (
        cp[:, :, 0] * kis[:, :, 0]
        + cp[:, :, 1] * kis[:, :, 1]
        + cp[:, :, 2] * kis[:, :, 2]
    )

    alpha_i = np.where(np.isclose(alignment, 1.0), theta, -theta)

    # same for j
    cos_j = (
        e_theta_s[:, :, 0] * e_par_s[:, :, 0]
        + e_theta_s[:, :, 1] * e_par_s[:, :, 1]
        + e_theta_s[:, :, 2] * e_par_s[:, :, 2]
    )

    cp = np.cross(
        np.reshape(e_par_s, (num_linear, 3)),
        np.reshape(e_theta_s, (num_linear, 3)),
    ).reshape((num_one, num_two, 3))
    norms = np.linalg.norm(cp, axis=-1)
    cp = cp / norms[:, :, np.newaxis]

    cos_j = np.where(np.logical_or(np.isclose(cos_j, 1.0), cos_j > 1.0), 1.0, cos_j)
    cos_j = np.where(np.logical_or(np.isclose(cos_j, -1.0), cos_j < -1.0), -1.0, cos_j)
    theta = np.arccos(cos_j)

    # cp, ki
    alignment = (
        cp[:, :, 0] * kjs[:, :, 0]
        + cp[:, :, 1] * kjs[:, :, 1]
        + cp[:, :, 2] * kjs[:, :, 2]
    )

    alpha_j = np.where(np.isclose(alignment, 1.0), theta, -theta)

    # Special cases
    e_par_i[parallel_indices] = e_theta_i[parallel_indices]
    e_par_s[parallel_indices] = e_theta_s[parallel_indices]
    e_per[parallel_indices] = e_phi_i[parallel_indices]
    alpha_i[parallel_indices] = 0.0
    alpha_j[parallel_indices] = 0.0

    e_par_i[antiparallel_indices] = e_theta_i[antiparallel_indices]
    e_par_s[antiparallel_indices] = -e_theta_s[antiparallel_indices]
    e_per[antiparallel_indices] = e_phi_i[antiparallel_indices]
    alpha_i[antiparallel_indices] = 0.0
    alpha_j[antiparallel_indices] = np.pi

    T_i = np.zeros((num_one, num_two, 2, 2))
    T_i[:, :, 0, 0] = np.cos(alpha_i)
    T_i[:, :, 0, 1] = np.sin(alpha_i)
    T_i[:, :, 1, 0] = np.sin(alpha_i)
    T_i[:, :, 1, 1] = -np.cos(alpha_i)

    T_j = np.zeros((num_one, num_two, 2, 2))
    T_j[:, :, 0, 0] = np.cos(alpha_j)
    T_j[:, :, 0, 1] = -np.sin(alpha_j)
    T_j[:, :, 1, 0] = -np.sin(alpha_j)
    T_j[:, :, 1, 1] = -np.cos(alpha_j)

    A_scattering_plane = np.transpose(
        get_A_from_mus(mus, xs, ms).reshape(2, 2, num_one, num_two),
        (2, 3, 0, 1),
    )

    final = (
        (np.reshape(T_j, (num_linear, 2, 2)))
        @ (np.reshape(A_scattering_plane, (num_linear, 2, 2)))
        @ (np.reshape(T_i, (num_linear, 2, 2)))
    )
    final = -np.transpose(np.reshape(final, (num_one, num_two, 4)), (2, 0, 1))

    return final


get_A.particle_type = "sphere"


# @numba.njit(fastmath=True, parallel=True)
def get_A_product_conj(
    ki_x: Numeric,
    ki_y: Numeric,
    ki_z: Numeric,
    kj_x: Numeric,
    kj_y: Numeric,
    kj_z: Numeric,
    ku_x: Numeric,
    ku_y: Numeric,
    ku_z: Numeric,
    kv_x: Numeric,
    kv_y: Numeric,
    kv_z: Numeric,
    x: Numeric,
    m: Numeric,
) -> Numeric:
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    shape = np.shape(A_ij[0])

    output = np.zeros((16, shape[0], shape[1]), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            output[4 * i + j] = A_ij[i] * np.conj(A_uv[j])

    return output


# @numba.njit(fastmath=True, parallel=True)
def get_A_product(
    ki_x: Numeric,
    ki_y: Numeric,
    ki_z: Numeric,
    kj_x: Numeric,
    kj_y: Numeric,
    kj_z: Numeric,
    ku_x: Numeric,
    ku_y: Numeric,
    ku_z: Numeric,
    kv_x: Numeric,
    kv_y: Numeric,
    kv_z: Numeric,
    x: Numeric,
    m: Numeric,
) -> Numeric:
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    output = np.zeros((16, *np.shape(A_ij[0])), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            output[4 * i + j] = A_ij[i] * A_uv[j]
    return output


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
