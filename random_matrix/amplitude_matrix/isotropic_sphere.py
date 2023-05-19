from random_matrix.utils.types import FloatLike
import miepython
import numpy as np


def get_A_scattering_plane(
    k_inc: FloatLike,
    k_sca: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    if np.ndim(k_inc) == 1 and np.ndim(k_sca) == 1:
        mu = np.dot(k_inc, k_sca)
        return _get_A_scattering_plane_from_mu(x, m, mu)

    mu_array = np.array([0.0, 1.0])
    return _get_A_scattering_plane_from_mu(x, m, mu_array)


get_A_scattering_plane.particle_type = "isotropic sphere"  # type: ignore


def _get_A_scattering_plane_from_mu(
    x: FloatLike, m: FloatLike, mu: FloatLike
) -> FloatLike:
    S1, S2 = miepython.mie_S1_S2(m, x, mu)

    if isinstance(mu, np.ndarray):
        S3 = np.zeros(len(mu))
        S4 = np.zeros(len(mu))
        output = np.array([*S2, *S3, *S4, *S1])
    else:
        S3 = 0.0 + 0.0 * 1j
        S4 = 0.0 + 0.0 * 1j
        output = np.array([S2, S3, S4, S1])

    return output
