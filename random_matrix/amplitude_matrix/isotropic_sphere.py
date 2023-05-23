from random_matrix.utils.types import FloatLike
from miepython import miepython_nojit
import numpy as np


def get_A_scattering_plane(
    k_inc: FloatLike,
    k_sca: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    """Compute A with respect to the scattering plane"""
    mu = np.dot(k_inc, k_sca)
    return _get_A_scattering_plane_from_mu(x, m, mu)


get_A_scattering_plane.particle_type = "isotropic sphere"  # type: ignore


def _get_A_scattering_plane_from_mu(
    x: FloatLike, m: FloatLike, mu: FloatLike
) -> FloatLike:
    S1, S2 = miepython_nojit.mie_S1_S2(m, x, [mu], norm="bohren")
    S3 = np.complex128(0)
    S4 = np.complex128(0)
    output = np.array([*S2, S3, S4, *S1])
    return output
