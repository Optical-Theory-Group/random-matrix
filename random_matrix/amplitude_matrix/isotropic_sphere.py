import numpy as np
import scipy
import time

from random_matrix.utils import memoize
from random_matrix.utils.types import FloatLike


def get_angle_plane(v1: FloatLike, v2: FloatLike, n: FloatLike) -> FloatLike:
    """
    Calculate rotation angle used in A matrix calculation
    See Appendix D for more inforamtion
    """

    # Special cases
    cosine = np.dot(v1, v2)

    if cosine >= 1 or np.isclose(cosine, 1):
        return 0
    elif cosine <= -1 or np.isclose(cosine, -1):
        return np.pi

    theta = np.arccos(cosine)

    # check if n is in the same direction as v1xv2
    k = np.cross(v1, v2)
    k = k / np.linalg.norm(k)

    alignment = np.dot(k, n)
    if np.isclose(alignment, 1):
        alpha = theta
    else:
        alpha = -theta

    return alpha


def get_A(
    k_inc: FloatLike, k_sca: FloatLike, x: FloatLike, m: FloatLike
) -> FloatLike:
    """
    Calculate far-field single scattering matrix (A) using Mie theory
    k_inc = incident wavevector
    k_sca = outgoing wavevector
    params must contain:
        'x': size parameter
        'm': relative refractive index
        'type': scattering type (rayleigh, mie or chiral)
            if 'type' is 'chiral' then params must also contain 'mL' and 'mR',
            the relative refractive indices for left and right handed circularly polarized light

    N_lim is the number of terms used in calculating the scattering coefficients. Increase for greater accuracy.
    """

    N_lim = 10
    k_inc = k_inc / np.linalg.norm(k_inc)
    k_sca = k_sca / np.linalg.norm(k_sca)

    for i, component in enumerate(k_inc):
        if np.isclose(component, 0):
            k_inc[i] = 0
    for i, component in enumerate(k_sca):
        if np.isclose(component, 0):
            k_sca[i] = 0

    theta_i = np.arccos(k_inc[2])
    phi_i = np.arctan2(k_inc[1], k_inc[0])
    theta_s = np.arccos(k_sca[2])
    phi_s = np.arctan2(k_sca[1], k_sca[0])
    cos_theta_scattering = np.dot(k_inc, k_sca)

    if np.allclose(k_inc, np.array([0, 0, -1])):
        e_theta_i = np.array([1, 0, 0])
        e_phi_i = np.array([0, -1, 0])
    else:
        e_theta_i = np.array(
            [
                np.cos(theta_i) * np.cos(phi_i),
                np.cos(theta_i) * np.sin(phi_i),
                -np.sin(theta_i),
            ]
        )
        e_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i), 0])

    if np.allclose(k_sca, np.array([0, 0, -1])):
        e_theta_s = np.array([1, 0, 0])
        # e_phi_s = np.array([0,-1,0])
    else:
        e_theta_s = np.array(
            [
                np.cos(theta_s) * np.cos(phi_s),
                np.cos(theta_s) * np.sin(phi_s),
                -np.sin(theta_s),
            ]
        )
        # e_phi_s = np.array([-np.sin(phi_s), np.cos(phi_s), 0])

    if np.allclose(k_inc, k_sca):
        e_par_i = e_theta_i
        e_par_s = e_theta_s
        e_per = e_phi_i
        alpha1 = 0
        alpha2 = 0
    elif np.allclose(k_inc, -k_sca):
        e_par_i = e_theta_i
        e_par_s = -e_theta_s
        e_per = e_phi_i
        alpha1 = 0
        alpha2 = np.pi

    else:
        e_per = np.cross(k_inc, k_sca)
        e_per = e_per / np.linalg.norm(e_per)

        e_par_i = np.cross(e_per, k_inc)
        e_par_i = e_par_i / np.linalg.norm(e_par_i)

        e_par_s = np.cross(e_per, k_sca)
        e_par_s = e_par_s / np.linalg.norm(e_par_s)

        alpha1 = get_angle_plane(e_theta_i, e_par_i, k_inc)
        alpha2 = get_angle_plane(e_par_s, e_theta_s, k_sca)

    T1 = np.array(
        [[np.cos(alpha1), np.sin(alpha1)], [-np.sin(alpha1), np.cos(alpha1)]]
    )
    T2 = np.array([[1, 0], [0, -1]])
    T4 = np.array([[1, 0], [0, -1]])
    T5 = np.array(
        [[np.cos(alpha2), np.sin(alpha2)], [-np.sin(alpha2), np.cos(alpha2)]]
    )

    mu = np.dot(k_inc, k_sca)

    T3 = get_T3(mu, x, m)

    T = -T5 @ T4 @ T3 @ T2 @ T1

    return T.flatten()


get_A.particle_type = "isotropic_sphere"


def get_T3(mu, x, m):
    N_lim = 10
    psi_x = scipy.special.riccati_jn(N_lim, x)
    psi_mx = scipy.special.riccati_jn(N_lim, m * x)
    phi_x = scipy.special.riccati_yn(N_lim, x)
    xi_x = [psi + 1j * phi for psi, phi in zip(psi_x, phi_x)]
    S1 = 0
    S2 = 0

    for n in range(1, N_lim):
        a = (m * psi_mx[0][n] * psi_x[1][n] - psi_x[0][n] * psi_mx[1][n]) / (
            m * psi_mx[0][n] * xi_x[1][n] - xi_x[0][n] * psi_mx[1][n]
        )
        b = (psi_mx[0][n] * psi_x[1][n] - m * psi_x[0][n] * psi_mx[1][n]) / (
            psi_mx[0][n] * xi_x[1][n] - m * xi_x[0][n] * psi_mx[1][n]
        )
        S1 = S1 + (2 * n + 1) / (n * (n + 1)) * (
            a * pi(n, mu) + b * tau(n, mu)
        )
        S2 = S2 + (2 * n + 1) / (n * (n + 1)) * (
            a * tau(n, mu) + b * pi(n, mu)
        )

    T3 = np.array([[S2, 0], [0, S1]])
    return T3


def pi(n: FloatLike, mu: FloatLike) -> FloatLike:
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return (2 * n - 1) / (n - 1) * mu * pi(n - 1, mu) - n / (n - 1) * pi(
            n - 2, mu
        )


def tau(n: FloatLike, mu: FloatLike) -> FloatLike:
    return n * mu * pi(n, mu) - (n + 1) * pi(n - 1, mu)


def get_A_product_conj(
    k_i: FloatLike,
    k_j: FloatLike,
    k_u: FloatLike,
    k_v: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    A_ij = get_A(k_i, k_j, x, m)
    A_uv = get_A(k_u, k_v, x, m)
    prod = np.outer(A_ij, np.conj(A_uv))
    return np.ravel(prod)


def get_A_product(
    k_i: FloatLike,
    k_j: FloatLike,
    k_u: FloatLike,
    k_v: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    A_ij = get_A(k_i, k_j, x, m)
    A_uv = get_A(k_u, k_v, x, m)
    prod = np.outer(A_ij, A_uv)
    return np.ravel(prod)

start = time.perf_counter()
for _ in range(10**4):
    get_T3(1.0,4.0,1.3)
end = time.perf_counter()
print(end-start)