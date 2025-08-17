import time

import numba
import numpy as np
import cupy as cp
import scipy
from random_matrix.utils import array_utils


def get_e_theta_phi(
    k_x: np.ndarray | cp.ndarray,
    k_y: np.ndarray | cp.ndarray,
    k_z: np.ndarray | cp.ndarray,
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
    """Get the theta and phi spherical polar basis vectors associated with
    a plane wave with given wavevector. The wavevector is assumed to be
    normalized."""
    xp = array_utils.get_module(k_x)
    mod_kappa = xp.sqrt(k_x**2 + k_y**2)

    # Find indices where k is parallel to z
    bad_indices = xp.where(xp.isclose(mod_kappa, 0.0))
    num_bad_indices = len(bad_indices[0])

    # Do the cross product and normalize (e_phi = z cross k)
    e_phi = xp.empty((*k_x.shape, 3))
    e_phi[..., 0] = -k_y
    e_phi[..., 1] = k_x
    e_phi[..., 2] = 0.0
    e_phi /= mod_kappa[..., xp.newaxis]
    e_phi[bad_indices] = xp.column_stack(
        (
            xp.zeros(num_bad_indices),
            k_z[bad_indices],
            xp.zeros(num_bad_indices),
        )
    )

    # e_theta = e_phi cross k
    e_theta = xp.empty((*k_x.shape, 3))
    e_theta[..., 0] = k_x * k_z
    e_theta[..., 1] = k_y * k_z
    e_theta[..., 2] = -(mod_kappa**2)
    e_theta /= xp.sqrt(
        e_theta[..., 0] ** 2 + e_theta[..., 1] ** 2 + e_theta[..., 2] ** 2
    )[..., xp.newaxis]
    e_theta[bad_indices] = xp.column_stack(
        (
            xp.ones(num_bad_indices),
            xp.zeros(num_bad_indices),
            xp.zeros(num_bad_indices),
        )
    )
    return e_theta, e_phi


def get_transformation_matrices(
    ki_x: np.ndarray | cp.ndarray,
    ki_y: np.ndarray | cp.ndarray,
    ki_z: np.ndarray | cp.ndarray,
    kj_x: np.ndarray | cp.ndarray,
    kj_y: np.ndarray | cp.ndarray,
    kj_z: np.ndarray | cp.ndarray,
    sign_convention: str = "BH",
) -> np.ndarray | cp.ndarray:
    """Get the basis vectors for the scattering plane formed from ki and kj."""
    xp = array_utils.get_module(ki_x)

    num_linear = len(ki_x)

    e_theta_i, e_phi_i = get_e_theta_phi(ki_x, ki_y, ki_z)
    e_theta_j, e_phi_j = get_e_theta_phi(kj_x, kj_y, kj_z)

    e_per = xp.cross(
        xp.stack([ki_x, ki_y, ki_z], axis=-1),
        xp.stack([kj_x, kj_y, kj_z], axis=-1),
    )
    e_per_norms = xp.sqrt(
        e_per[:, 0] ** 2 + e_per[:, 1] ** 2 + e_per[:, 2] ** 2
    )

    # Clean up cases where ki and kj are parallel
    all_indices = xp.arange(len(e_per_norms))
    zero_indices = xp.where(xp.isclose(e_per_norms, 0.0))
    non_zero_indices = xp.setdiff1d(all_indices, zero_indices)

    # Normalize non-zero cases
    e_per[non_zero_indices] /= e_per_norms[non_zero_indices][..., xp.newaxis]

    # Fix zero cases
    e_per[zero_indices] = e_phi_i[zero_indices]
    new_e_per_norms = xp.sqrt(
        e_per[zero_indices, 0] ** 2
        + e_per[zero_indices, 1] ** 2
        + e_per[zero_indices, 2] ** 2
    )[0]
    e_per[zero_indices] /= new_e_per_norms[..., xp.newaxis]

    # Get the parallel basis vectors
    e_par_i = xp.cross(
        e_per,
        xp.stack([ki_x, ki_y, ki_z], axis=-1),
    )
    e_par_j = xp.cross(
        e_per,
        xp.stack([kj_x, kj_y, kj_z], axis=-1),
    )

    # Work out rotation angles
    cos_alpha_i = (
        e_par_i[:, 0] * e_theta_i[:, 0]
        + e_par_i[:, 1] * e_theta_i[:, 1]
        + e_par_i[:, 2] * e_theta_i[:, 2]
    )

    cross_i = xp.cross(e_theta_i, e_par_i)
    cross_i_norms = xp.sqrt(
        cross_i[:, 0] ** 2 + cross_i[:, 1] ** 2 + cross_i[:, 2] ** 2
    )
    signs = xp.sign(
        cross_i[..., 0] * ki_x
        + cross_i[..., 1] * ki_y
        + cross_i[..., 2] * ki_z
    )
    sin_alpha_i = cross_i_norms * signs

    # Same for j
    cos_alpha_j = (
        e_par_j[:, 0] * e_theta_j[:, 0]
        + e_par_j[:, 1] * e_theta_j[:, 1]
        + e_par_j[:, 2] * e_theta_j[:, 2]
    )

    cross_j = xp.cross(e_theta_j, e_par_j)
    cross_j_norms = xp.sqrt(
        cross_j[:, 0] ** 2 + cross_j[:, 1] ** 2 + cross_j[:, 2] ** 2
    )
    signs = xp.sign(
        cross_j[..., 0] * kj_x
        + cross_j[..., 1] * kj_y
        + cross_j[..., 2] * kj_z
    )
    sin_alpha_j = cross_j_norms * signs

    # Build transformation matrices
    # Not
    sign_convention = "BH"
    sign_factor = 1.0
    if sign_convention == "BH":
        sign_factor = -1.0

    T_i = xp.zeros((num_linear, 2, 2))
    T_j = xp.zeros((num_linear, 2, 2))

    T_i[:, 0, 0] = cos_alpha_i
    T_i[:, 0, 1] = sin_alpha_i
    T_i[:, 1, 0] = -sign_factor * sin_alpha_i
    T_i[:, 1, 1] = sign_factor * cos_alpha_i

    T_j[:, 0, 0] = cos_alpha_j
    T_j[:, 0, 1] = -sign_factor * sin_alpha_j
    T_j[:, 1, 0] = sin_alpha_j
    T_j[:, 1, 1] = sign_factor * cos_alpha_j

    return T_i, T_j


def get_two_to_three_matrices(
    k_x: np.ndarray | cp.ndarray,
    k_y: np.ndarray | cp.ndarray,
    k_z: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    e_theta, e_phi = get_e_theta_phi(k_x, k_y, k_z)
    P = np.stack([e_theta, e_phi], axis=1)
    return P
