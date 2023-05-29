import numpy as np
import numba
import scipy
import time
from random_matrix.utils.types import FloatLike

@numba.njit
def get_A_from_mus(inputs: FloatLike) -> FloatLike:
    """Inputs:

    inputs[0] = mu
    input[1] = x
    input[2] = m
    """

    mus = inputs[0]
    xs = inputs[1]
    ms = inputs[2]

    num_inputs = len(mus)

    S_2 = np.zeros(num_inputs, dtype=np.complex128)
    S_1 = np.zeros(num_inputs, dtype=np.complex128)
    S_3 = np.zeros(num_inputs, dtype=np.complex128)
    S_4 = np.zeros(num_inputs, dtype=np.complex128)

    # Get stopping index for sum
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

    pi_0 = np.zeros(num_inputs, dtype=np.float64)
    pi_1 = np.ones(num_inputs, dtype=np.float64)

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
    output = np.vstack((S_2, S_3, S_4, S_1))

    return output


@numba.njit
def get_A(inputs: FloatLike) -> FloatLike:
    """Inputs:

    inputs[0] = ki_x
    inputs[1] = ki_y
    inputs[2] = ki_z
    inputs[3] = kj_x
    inputs[4] = kj_y
    inputs[5] = kj_z
    inputs[6] = x
    inputs[7] = m
    """

    kis = inputs[0:3]
    kjs = inputs[3:6]
    xs = inputs[6]
    ms = inputs[7]

    mus = np.sum(np.multiply(kis, kjs), axis=0)
    new_inputs = np.vstack((mus, xs, ms))

    return get_A_from_mus(new_inputs)
