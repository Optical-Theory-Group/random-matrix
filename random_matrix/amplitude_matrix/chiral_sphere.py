import time

# import numba
import numpy as np
import scipy
import cupy as cp
from random_matrix.amplitude_matrix import scattering_geometry
from random_matrix.utils import array_utils


# @numba.njit(fastmath=False, parallel=True)
def get_A_scattering_plane(
    mu: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
    brg: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:

    input_shape = np.shape(mu)
    mrs = m + brg
    mls = m - brg
    S_2 = np.zeros(input_shape, dtype=np.complex128)
    S_1 = np.zeros(input_shape, dtype=np.complex128)
    S_3 = np.zeros(input_shape, dtype=np.complex128)
    S_4 = np.zeros(input_shape, dtype=np.complex128)

    # Get stopping index for sum
    if isinstance(x, np.float64):
        num_stop = int(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1)
    else:
        num_stop = int(np.max(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1))

    # Initialise variables
    # 1 and 2 subscripts are for recurrence relations, 2 is ahead of 1
    jx_0 = np.sin(x) / x
    jx_1 = np.sin(x) / x**2 - np.cos(x) / x
    jmrx_0 = np.sin(mrs * x) / (mrs * x)
    jmrx_1 = np.sin(mrs * x) / (mrs * x) ** 2 - np.cos(mrs * x) / (mrs * x)
    jmlx_0 = np.sin(mls * x) / (mls * x)
    jmlx_1 = np.sin(mls * x) / (mls * x) ** 2 - np.cos(mls * x) / (mls * x)

    djx = ((x**2.0 - 2.0) * np.sin(x) + 2.0 * x * np.cos(x)) / x**3.0
    djmrx = (
        ((mrs * x) ** 2.0 - 2.0) * np.sin(mrs * x)
        + 2.0 * mrs * x * np.cos(mrs * x)
    ) / (mrs * x) ** 3
    djmlx = (
        ((mls * x) ** 2.0 - 2.0) * np.sin(mls * x)
        + 2.0 * mls * x * np.cos(mls * x)
    ) / (mls * x) ** 3

    yx_0 = -np.cos(x) / x
    yx_1 = -np.cos(x) / x**2 - np.sin(x) / x
    ymrx_0 = -np.cos(mrs * x) / (mrs * x)
    ymrx_1 = -np.cos(mrs * x) / (mrs * x) ** 2 - np.sin(mrs * x) / (mrs * x)
    ymlx_0 = -np.cos(mls * x) / (mls * x)
    ymlx_1 = -np.cos(mls * x) / (mls * x) ** 2 - np.sin(mls * x) / (mls * x)

    dyx = (2.0 * x * np.sin(x) - (x**2 - 2.0) * np.cos(x)) / x**3

    psix = x * jx_1
    psimrx = mrs * x * jmrx_1
    psimlx = mls * x * jmlx_1
    dpsix = x * djx + jx_1
    dpsimrx = mrs * x * djmrx + jmrx_1
    dpsimlx = mls * x * djmlx + jmlx_1
    phix = x * yx_1
    dphix = x * dyx + yx_1
    xix = psix + 1j * phix
    dxix = dpsix + 1j * dphix

    WR = m * psimrx * dxix - xix * dpsimrx
    WL = m * psimlx * dxix - xix * dpsimlx

    VR = psimrx * dxix - m * xix * dpsimrx
    VL = psimlx * dxix - m * xix * dpsimlx

    AR = m * psimrx * dpsix - psix * dpsimrx
    AL = m * psimlx * dpsix - psix * dpsimlx

    BR = psimrx * dpsix - m * psix * dpsimrx
    BL = psimlx * dpsix - m * psix * dpsimlx

    pi_0 = np.zeros(input_shape, dtype=np.float64)
    pi_1 = np.ones(input_shape, dtype=np.float64)

    tau = mu

    # Initialise S matrix terms. Note that n=1 corresponds to the _2 variables
    a = (VR * AL + VL * AR) / (WL * VR + VL * WR)
    b = (WL * BR + WR * BL) / (WL * VR + VL * WR)
    c = 1j * (WR * AL - WL * AR) / (WL * VR + VL * WR)
    d = -c

    S_1 = 3.0 / 2.0 * (a * pi_1 + b * tau)
    S_2 = 3.0 / 2.0 * (a * tau + b * pi_1)
    S_3 = 3.0 / 2.0 * (c * (pi_1 + tau))
    S_4 = -S_3

    for n in range(2, num_stop + 1):
        # Update all variables with recurrence relations
        # Bessel functions
        new_jx = (2.0 * n - 1.0) / x * jx_1 - jx_0
        jx_0 = jx_1
        jx_1 = new_jx
        new_jmrx = (2.0 * n - 1.0) / (mrs * x) * jmrx_1 - jmrx_0
        jmrx_0 = jmrx_1
        jmrx_1 = new_jmrx
        new_jmlx = (2.0 * n - 1.0) / (mls * x) * jmlx_1 - jmlx_0
        jmlx_0 = jmlx_1
        jmlx_1 = new_jmlx
        new_yx = (2.0 * n - 1.0) / x * yx_1 - yx_0
        yx_0 = yx_1
        yx_1 = new_yx
        new_ymrx = (2.0 * n - 1.0) / (mrs * x) * ymrx_1 - ymrx_0
        ymrx_0 = ymrx_1
        ymrx_1 = new_ymrx
        new_ymlx = (2.0 * n - 1.0) / (mls * x) * ymlx_1 - ymlx_0
        ymlx_0 = ymlx_1
        ymlx_1 = new_ymlx

        # Derivatives of bessel funtions
        new_djx = -(n + 1) / x * jx_1 + jx_0
        new_djmrx = -(n + 1) / (mrs * x) * jmrx_1 + jmrx_0
        new_djmlx = -(n + 1) / (mls * x) * jmlx_1 + jmlx_0
        new_dyx = -(n + 1) / x * yx_1 + yx_0

        new_psix = x * new_jx
        new_psimrx = mrs * x * new_jmrx
        new_psimlx = mls * x * new_jmlx
        new_dpsix = x * new_djx + new_jx
        new_dpsimrx = mrs * x * new_djmrx + new_jmrx
        new_dpsimlx = mls * x * new_djmlx + new_jmlx
        new_phix = x * new_yx
        new_dphix = x * new_dyx + new_yx
        new_xix = new_psix + 1j * new_phix
        new_dxix = new_dpsix + 1j * new_dphix

        # Angular functions
        new_pi = (2.0 * n - 1.0) / (n - 1.0) * mu * pi_1 - n / (n - 1.0) * pi_0
        pi_0 = pi_1
        pi_1 = new_pi

        new_tau = n * mu * new_pi - (n + 1.0) * pi_0
        WR = m * new_psimrx * new_dxix - new_xix * new_dpsimrx
        WL = m * new_psimlx * new_dxix - new_xix * new_dpsimlx

        VR = new_psimrx * new_dxix - m * new_xix * new_dpsimrx
        VL = new_psimlx * new_dxix - m * new_xix * new_dpsimlx

        AR = m * new_psimrx * new_dpsix - new_psix * new_dpsimrx
        AL = m * new_psimlx * new_dpsix - new_psix * new_dpsimlx

        BR = new_psimrx * new_dpsix - m * new_psix * new_dpsimrx
        BL = new_psimlx * new_dpsix - m * new_psix * new_dpsimlx

        # Calculate new S terms
        a = (VR * AL + VL * AR) / (WL * VR + VL * WR)
        b = (WL * BR + WR * BL) / (WL * VR + VL * WR)
        c = 1j * (WR * AL - WL * AR) / (WL * VR + VL * WR)
        d = -c

        S_1 = S_1 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_pi + b * new_tau
        )
        S_2 = S_2 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            a * new_tau + b * new_pi
        )
        S_3 = S_3 + (2.0 * n + 1.0) / (n * (n + 1.0)) * (
            c * (new_tau + b * new_pi)
        )
        S_4 = -S_3

    A = np.stack([S_2, S_3, S_4, S_1], axis=-1)
    return A


# @numba.njit(fastmath=True, parallel=True)
def get_A(
    ki_x: np.ndarray | cp.ndarray,
    ki_y: np.ndarray | cp.ndarray,
    ki_z: np.ndarray | cp.ndarray,
    kj_x: np.ndarray | cp.ndarray,
    kj_y: np.ndarray | cp.ndarray,
    kj_z: np.ndarray | cp.ndarray,
    x: np.ndarray | cp.ndarray,
    m: np.ndarray | cp.ndarray,
    brg: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:

    length = len(ki_x)
    # Get geometric transformation matrices for converting between
    # different basis vectors
    T_i, T_j = scattering_geometry.get_transformation_matrices(
        ki_x, ki_y, ki_z, kj_x, kj_y, kj_z
    )

    # Get A in the scattering plane
    mu = ki_x * kj_x + ki_y * kj_y + ki_z * kj_z
    A_scattering_plane = get_A_scattering_plane(mu, x, m, brg)
    output = (T_j @ A_scattering_plane.reshape(length, 2, 2) @ T_i).reshape(
        length, 4
    )
    return output


get_A.particle_type = "sphere"


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
    brg: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    xp = array_utils.get_module(ki_x)
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m, brg)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m, brg)
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
    brg: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    xp = array_utils.get_module(ki_x)
    A_ij = get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m, brg)
    A_uv = get_A(ku_x, ku_y, ku_z, kv_x, kv_y, kv_z, x, m, brg)
    product = A_ij[:, :, xp.newaxis] * A_uv[:, xp.newaxis, :]
    output = np.reshape(product, (*ki_x.shape, 16))
    return output
