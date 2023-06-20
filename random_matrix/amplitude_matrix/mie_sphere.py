import numpy as np
import numba
import scipy
import time
from random_matrix.utils.types import FloatLike


#@numba.njit(fastmath=False, parallel=True)
def get_A_from_mus(mus: FloatLike, xs: FloatLike, ms: FloatLike) -> FloatLike:
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
        ((ms * xs) ** 2.0 - 2.0) * np.sin(ms * xs)
        + 2.0 * ms * xs * np.cos(ms * xs)
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
    a = (ms * psimx * dpsix - psix * dpsimx) / (
        ms * psimx * dxix - xix * dpsimx
    )
    b = (psimx * dpsix - ms * psix * dpsimx) / (
        psimx * dxix - ms * xix * dpsimx
    )

    S_2 = 3.0 / 2.0 * (a * pi_1 + b * tau)
    S_1 = 3.0 / 2.0 * (a * tau + b * pi_1)

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
        new_pi = (2.0 * n - 1.0) / (n - 1.0) * mus * pi_1 - n / (
            n - 1.0
        ) * pi_0
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

        S_2 = S_2 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_pi + b * new_tau
        )
        S_1 = S_1 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_tau + b * new_pi
        )

    # Prepare output
    # top_row = np.concatenate((S_2[np.newaxis, :], S_3[np.newaxis, :]))
    # bottom_row = np.concatenate((S_4[np.newaxis, :], S_1[np.newaxis, :]))
    # combined_array = np.concatenate(
    #     (top_row[np.newaxis, :], bottom_row[np.newaxis, :])
    # )

    combined_array = np.concatenate(
        (
            S_2[np.newaxis, :],
            S_3[np.newaxis, :],
            S_4[np.newaxis, :],
            S_1[np.newaxis, :],
        )
    )

    return combined_array


#@numba.njit(fastmath=True, parallel=True)
def get_A(
    ki_x: FloatLike,
    ki_y: FloatLike,
    ki_z: FloatLike,
    kj_x: FloatLike,
    kj_y: FloatLike,
    kj_z: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    xs = x
    ms = m
    mus = ki_x * kj_x + ki_y * kj_y + ki_z * kj_z

    return get_A_from_mus(mus, xs, ms)


get_A.particle_type = "sphere"


#@numba.njit(fastmath=True, parallel=True)
def get_A_product_conj(
    ki_x: FloatLike,
    ki_y: FloatLike,
    ki_z: FloatLike,
    kj_x: FloatLike,
    kj_y: FloatLike,
    kj_z: FloatLike,
    ku_x: FloatLike,
    ku_y: FloatLike,
    ku_z: FloatLike,
    kv_x: FloatLike,
    kv_y: FloatLike,
    kv_z: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    shape = np.shape(A_ij[0])

    output = np.zeros((16, shape[0], shape[1]), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            output[4 * i + j] = A_ij[i] * np.conj(A_uv[j])

    return output


#@numba.njit(fastmath=True, parallel=True)
def get_A_product(
    ki_x: FloatLike,
    ki_y: FloatLike,
    ki_z: FloatLike,
    kj_x: FloatLike,
    kj_y: FloatLike,
    kj_z: FloatLike,
    ku_x: FloatLike,
    ku_y: FloatLike,
    ku_z: FloatLike,
    kv_x: FloatLike,
    kv_y: FloatLike,
    kv_z: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m)
    output = np.zeros((16, *np.shape(A_ij[0])), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            output[4 * i + j] = A_ij[i] * A_uv[j]
    return output
