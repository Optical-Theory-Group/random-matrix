import os

# # Change to a new directory
# new_path = r"/home/sdutta/code/random-matrix"
# os.chdir(new_path)
# # print(os.getcwd())
import matplotlib.pyplot as plt
from math import factorial
import scipy
import quadpy
from scipy.special import spherical_yn, spherical_jn, lpmv
from scipy.integrate import quad, dblquad
from scipy.spatial import ConvexHull
import numpy as np
import cupy as cp
from random_matrix.utils import integration_utils as iu
import math
from tqdm import tqdm
from functools import lru_cache
import cProfile
from typing import Callable, Any
from numpy.polynomial.legendre import leggauss


"""
All functions are defined according to the equations of Appendix C of Mischenko
and Travis 2002. If you have a list of points you can create the convex hull of these 
points and pass it to the get_T function to calculate the T matrix. You will aslo need 
to pass the k1 and k2 values along with n_max value.
The T matrix will have a size of (2*modes X 2*modes).
"""


def pi_tau_mn(m, n, theta):
    if isinstance(theta, (list, np.ndarray)):
        theta[theta == 0] = np.sqrt(
            2 * np.finfo(np.float64).eps
        )  # remove issues with divide by zero
        theta[theta == np.pi] = np.pi + np.sqrt(
            2 * np.finfo(np.float64).eps
        )  # remove issues with divide by zero

    elif isinstance(theta, int):
        if theta == 0:
            theta = np.sqrt(2 * np.finfo(np.float64).eps)
        elif theta == np.pi:
            theta = np.pi + np.sqrt(2 * np.finfo(np.float64).eps)

    Ln = lpmv(m, n, np.cos(theta))
    Lnp1 = lpmv(m, n + 1, np.cos(theta))

    norm = np.sqrt(factorial(n - m) / factorial(n + m))
    pi_out = norm * m * Ln / (np.sin(theta))
    tau_out = norm * (
        -(1 + n) * np.cos(theta) * Ln / (np.sin(theta))
        + (1 - m + n) * Lnp1 / np.sin(theta)
    )

    return (pi_out, tau_out)


"""
This function calculates the constant used to convert the Cmn(theta)
function to Cmn(theta,phi) function.

"""


@lru_cache(maxsize=None)  # Memoize for all (m, n) pairs
def const_mn(m, n):

    return ((-1) ** (m)) * np.sqrt(factorial(n + m) / factorial(n - m))


"""
This function calcultes the gamma_mn constant used in the calculation of the
RgN and RgM functions.

"""


@lru_cache(maxsize=None)  # Memoize for all (m, n) pairs
def gamma_mn(m, n):
    if n == 0:  # To avoid division by zero
        return 0
    return np.sqrt(
        ((2 * n + 1) * factorial(n - m)) / (4 * np.pi * n * (n + 1) * factorial(n + m))
    )


# Coordinate transformation from cartesian to spherical
def cart2sph(x, y, z):
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    t = np.pi / 2 - np.arctan2(z, np.sqrt(xy))
    p = np.arctan2(y, x)
    if np.all(p < 0):
        p = p + 2 * np.pi

    return (r, t, p)


def sph2cart_comp_mat(theta, phi):
    R = np.array(
        [
            [
                np.cos(phi) * np.sin(theta),
                np.cos(phi) * np.cos(theta),
                -np.sin(phi),
            ],
            [
                np.sin(phi) * np.sin(theta),
                np.sin(phi) * np.cos(theta),
                np.cos(phi),
            ],
            [np.cos(theta), -np.sin(theta), 0 * theta],
        ]
    )

    return R


# Converts a vector from spherical to cartesian coordinates
def sph2cart_comp_vec(U, theta, phi):
    R = sph2cart_comp_mat(theta, phi)
    Uout = np.einsum("mn...,n...->m...", R, U)

    return Uout


def B_mn(m, n, theta, phi):
    c = const_mn(m, n) * np.exp(1j * m * phi)

    pitaumn = pi_tau_mn(m, n, theta)
    theta_comp = c * pitaumn[1]
    phi_comp = 1j * c * pitaumn[0]
    return (theta_comp, phi_comp)


def C_mn(m, n, theta, phi):
    c = const_mn(m, n) * np.exp(1j * m * phi)

    pitaumn = pi_tau_mn(m, n, theta)
    theta_comp = 1j * c * pitaumn[0]
    phi_comp = -c * pitaumn[1]
    return (theta_comp, phi_comp)


def d_mn(m, n, theta):
    x = np.cos(theta)
    lp = lpmv(m, n, x)
    return lp * (np.sqrt(factorial(n - m) / factorial(n + m)))


# Function returns the spherical Bessel functions
def j_n(n, kr):

    return spherical_jn(n, kr)


# Function reutrns Hankel function of 1st kind
def h_n(n, kr):
    return spherical_jn(n, kr) + 1j * spherical_yn(n, kr)


def dh_n(n, x):
    """Computes the derivative of the spherical Hankel function of the first kind."""
    hn = h_n(n, x)
    hn1 = h_n(n + 1, x)
    return (n / x) * hn - hn1


def P_mn(m, n, theta, phi):
    c = const_mn(m, n) * np.exp(1j * m * phi)
    return c * d_mn(m, n, theta)


def RgM_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)  # Cached value
    Cmn = C_mn(m, n, theta, phi)
    theta_comp = gamma * j_n(n, kr) * Cmn[0]
    phi_comp = gamma * j_n(n, kr) * Cmn[1]

    return np.array([theta_comp, phi_comp], dtype=np.complex128)


def RgN_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)
    r_comp = gamma * n * (n + 1) / kr * spherical_jn(n, kr) * P_mn(m, n, theta, phi)
    theta_comp = (
        gamma
        * (spherical_jn(n, kr) / kr + spherical_jn(n, kr, derivative=True))
        * B_mn(m, n, theta, phi)[0]
    )
    phi_comp = (
        gamma
        * (spherical_jn(n, kr) / kr + spherical_jn(n, kr, derivative=True))
        * B_mn(m, n, theta, phi)[1]
    )

    return np.array([r_comp, theta_comp, phi_comp])


def N_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)
    Bmn = B_mn(m, n, theta, phi)
    r_comp = gamma * (n * (n + 1) / kr) * h_n(n, kr) * P_mn(m, n, theta, phi)
    theta_comp = gamma * (h_n(n, kr) / kr + dh_n(n, kr)) * Bmn[0]
    phi_comp = gamma * (h_n(n, kr) / kr + dh_n(n, kr)) * Bmn[1]

    return np.array([r_comp, theta_comp, phi_comp], dtype=np.complex128)


def M_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)  # Cached value
    Cmn = C_mn(m, n, theta, phi)
    theta_comp = gamma * h_n(n, kr) * Cmn[0]
    phi_comp = gamma * h_n(n, kr) * Cmn[1]
    return np.array([theta_comp, phi_comp], dtype=np.complex128)


"""
This function creates the Convex Hull from a set of points in (x,y,z)
It uses uniform sampling in theta and phi (Not the best way of sampling in theta)

"""


def create_hull_uniform(size_param, k):
    row = 50
    col = 51
    num_linear = row * col
    theta = np.linspace(0, np.pi, col)
    phi = np.linspace(0, 2 * np.pi, row)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    r = size_param / k  # particle size is 600nm
    x = np.reshape((r * np.sin(theta_grid) * np.cos(phi_grid)), (1, num_linear))
    y = np.reshape((r * np.sin(theta_grid) * np.sin(phi_grid)), (1, num_linear))
    z = np.reshape((r * np.cos(theta_grid)), (1, num_linear))
    points = np.transpose(np.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


def create_hull_random():
    num_points = 10000
    points = np.random.rand(num_points, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * 600e-9
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


"""
Theta sampling done using Gauss-Chebyshev Quadrature.
"""


def create_hull_inv_transform():
    row = 50
    col = 51
    num_linear = row * col
    u = np.linspace(0, 1, col)
    theta = np.arccos(1 - 2 * u)
    phi = np.linspace(0, 2 * np.pi, row)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    r = 600e-9  # particle size is 600nm
    x = np.reshape((r * np.sin(theta_grid) * np.cos(phi_grid)), (1, num_linear))
    y = np.reshape((r * np.sin(theta_grid) * np.sin(phi_grid)), (1, num_linear))
    z = np.reshape((r * np.cos(theta_grid)), (1, num_linear))
    points = np.transpose(np.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


def create_hull_GC():
    row = 100
    col = 101
    num_linear = row * col

    theta = np.pi * (np.arange(1, col + 1) - 0.5) / col
    phi = np.linspace(0, 2 * np.pi, row)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    r = 600e-9  # particle size is 600nm
    x = np.reshape((r * np.sin(theta_grid) * np.cos(phi_grid)), (1, num_linear))
    y = np.reshape((r * np.sin(theta_grid) * np.sin(phi_grid)), (1, num_linear))
    z = np.reshape((r * np.cos(theta_grid)), (1, num_linear))
    points = np.transpose(np.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


def ellipse_hull(lx, ly, lz):
    """
    Generate a rotated ellipsoid, compute its convex hull,
    and plot both the 3D shape and its 2D XY projection.

    Parameters
    ----------
    a, b, c : float
        Semi-axes of the ellipsoid along x, y, z.
    R : np.ndarray
        3x3 rotation matrix defining the ellipsoid orientation.
    n_points : int
        Number of samples .
    """
    n_samples = 100
    # ---- 1. Generate ellipsoid surface points
    u = np.linspace(0, 2 * np.pi, n_samples)
    v = np.linspace(0, np.pi, n_samples)
    u, v = np.meshgrid(u, v)

    x = lx * np.cos(u) * np.sin(v)
    y = ly * np.sin(u) * np.sin(v)
    z = lz * np.cos(v)

    # Stack and rotate
    points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    rotated_points = points  # shape (3, N)

    # ---- 2. 3D Convex hull
    hull_3d = ConvexHull(rotated_points.T)

    return hull_3d


def hull_surface_integral_vector(
    function: Callable,
    hull: scipy.spatial.ConvexHull,
    scheme: Any | None = None,
    use_gpu: bool = False,
) -> np.ndarray | cp.ndarray:
    """Compute the integral of a function over the surface of a convex hull. It
    is assumed that the function returns a vector output, i.e. maps R^3 -> R^3.

    function: The function to be integrated. This function must be vectorized
    to accept arguments of shape

    N x n

    where n-1 is the number of dimensions of the simplical facets of the
    surface (alternatively, n is the number of dimensions of the ambient space
    in which the surface lies. For example, for the surface of a sphere, n=3)

    hull: The convex hull object that defines the surface.
    scheme: Integration scheme to be used.
    use_gpu: If true, use cupy instead of numpy
    """
    # Pick appropriate array module
    xp = cp if use_gpu else np

    # Pick integration scheme
    # For future developers: one can use Cayley-Menger determinants for d > 2
    num_dimensions = hull.points.shape[1] - 1
    if num_dimensions > 2:
        raise NotImplementedError(
            "Integration of surfaces in d > 2 dimensions" "not currently supported."
        )

    if scheme is None:
        # The second parameter to quadpy's scheme method here is about
        # accuracy of the scheme, not the number of spatial dimensions.
        scheme = quadpy.tn.grundmann_moeller(num_dimensions, 3)

    barycentric_weights = xp.asarray(scheme.points)
    weights = xp.asarray(scheme.weights)
    points = xp.asarray(hull.points)
    vertices = xp.asarray(hull.vertices)
    centroid = xp.mean(points[vertices], axis=0)
    simplices = xp.asarray(hull.simplices)

    # Generate integration points
    num_simplices = len(hull.simplices)
    num_weights = len(weights)
    simplical_points = points[simplices].transpose(0, 2, 1)
    integration_points = simplical_points @ barycentric_weights
    reshaped_points = integration_points.transpose(0, 2, 1).reshape(
        -1, num_dimensions + 1
    )

    # Find simplex volumes using cross product
    v1s = simplical_points[:, :, 1] - simplical_points[:, :, 0]
    v2s = simplical_points[:, :, 2] - simplical_points[:, :, 0]
    cross_products = xp.cross(v1s, v2s)
    unit_normals = cross_products / xp.linalg.norm(
        cross_products, axis=1, keepdims=True
    )
    radial_points = simplical_points[:, :, 0] - centroid
    dot_products = xp.sum(radial_points * unit_normals, axis=1)
    orientation_signs = xp.sign(dot_products)
    areas = 0.5 * xp.sqrt(xp.sum(cross_products**2, axis=1))

    # Output
    function_output = function(reshaped_points)
    integrand = xp.sum(
        function_output.reshape(num_simplices, num_weights, 3)
        * unit_normals[:, None, :],
        axis=2,
    ).reshape(num_simplices * num_weights)
    weighted_output = (
        integrand
        * xp.tile(weights, num_simplices)
        * xp.repeat(areas, num_weights)
        * xp.repeat(orientation_signs, num_weights)
    )
    integral = xp.sum(weighted_output)
    return integral


def compute_Js(hull, k1, k2, n_max):
    """
    This function calculates the J and RgJ matrices for the given modes. The function
    takes inputs k1,k2 and the convex hull of the particle surface sample points.
    """

    k1 = k1
    k2 = k2
    n_max = n_max
    weights = 1  # np.pi / 101

    hull = hull
    modes = sum(2 * n + 1 for n in range(1, n_max + 1))
    print(f"Number of modes: {modes} ")
    # modes for exterior region (m,n)
    # For a given n m runs from -n to +n
    modes_nm = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]
    # modes for the interior region (m',n')
    modes_nmp = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]

    j11 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    j12 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    j21 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    j22 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    Rgj11 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    Rgj12 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    Rgj21 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)
    Rgj22 = np.zeros((len(modes_nmp), len(modes_nm)), dtype=np.complex128)

    for i in range(0, modes):  # for each mode in the interior region
        for j in tqdm(range(0, modes)):  # for each mode in the exterior region

            def integrandj11(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                # RgM = np.hstack((np.zeros(6),RgM))
                M = M_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM[0] * M[1] - RgM[1] * M[0]
                theta_comp = 0 * M[1]
                phi_comp = 0 * RgM[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j11[j][i] = hull_surface_integral_vector(integrandj11, hull, use_gpu=False)
            j11[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj12(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                N = N_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM[0] * N[2] - RgM[1] * N[1]
                theta_comp = RgM[1] * N[0]
                phi_comp = -RgM[0] * N[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j12[j][i] = hull_surface_integral_vector(integrandj12, hull, use_gpu=False)
            j12[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj21(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                M = M_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN[1] * M[1] - RgN[2] * M[0]
                theta_comp = -RgN[0] * M[1]
                phi_comp = RgN[0] * M[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j21[j][i] = hull_surface_integral_vector(integrandj21, hull, use_gpu=False)
            j21[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj22(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)

                N = N_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN[1] * N[2] - RgN[2] * N[1]
                theta_comp = -RgN[0] * N[2] + RgN[2] * N[0]
                phi_comp = RgN[0] * N[1] - RgN[1] * N[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j22[j][i] = hull_surface_integral_vector(integrandj22, hull, use_gpu=False)
            j22[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandRgj11(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM1 = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                # RgM = np.hstack((np.zeros(6),RgM))
                RgM2 = RgM_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM1[0] * RgM2[1] - RgM1[1] * RgM2[0]
                theta_comp = 0 * RgM2[1]
                phi_comp = 0 * RgM1[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj11[j][i] = hull_surface_integral_vector(
                integrandRgj11, hull, use_gpu=False
            )
            Rgj11[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandRgj12(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                RgN = RgN_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM[0] * RgN[2] - RgM[1] * RgN[1]
                theta_comp = RgM[1] * RgN[0]
                phi_comp = -RgM[0] * RgN[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj12[j][i] = hull_surface_integral_vector(
                integrandRgj12, hull, use_gpu=False
            )
            Rgj12[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandRgj21(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                RgM = RgM_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN[1] * RgM[1] - RgN[2] * RgM[0]
                theta_comp = -RgN[0] * RgM[1]
                phi_comp = RgN[0] * RgM[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj21[j][i] = hull_surface_integral_vector(
                integrandRgj21, hull, use_gpu=False
            )
            Rgj21[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandRgj22(input):
                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN1 = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)

                RgN2 = RgN_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN1[1] * RgN2[2] - RgN1[2] * RgN2[1]
                theta_comp = -RgN1[0] * RgN2[2] + RgN1[2] * RgN2[0]
                phi_comp = RgN1[0] * RgN2[1] - RgN1[1] * RgN2[0]
                cross_prod = np.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj22[j][i] = hull_surface_integral_vector(
                integrandRgj22, hull, use_gpu=False
            )
            Rgj22[j][i] *= (-1) ** (modes_nm[i][1])

    return j11, j12, j21, j22, Rgj11, Rgj12, Rgj21, Rgj22


def get_T(hull, k1, k2, n_max):
    """This function calculates the T matrix for the given modes. The function
    takes inputs k1,k2 and the convex hull of the particle surface sample points.
    The rows are (n',m') and the columns are (n,m) for the T matrix.
    The T matrix will have a size of (2*modes X 2*modes).
    """
    j11, j12, j21, j22, Rgj11, Rgj12, Rgj21, Rgj22 = compute_Js(hull, k1, k2, n_max)

    # Define the Q matrices
    Q_11 = -1j * k1 * k2 * j21 - 1j * k1**2 * j12
    Q_12 = -1j * k1 * k2 * j11 - 1j * k1**2 * j22
    Q_21 = -1j * k1 * k2 * j22 - 1j * k1**2 * j11
    Q_22 = -1j * k1 * k2 * j12 - 1j * k1**2 * j21

    # Define the RgQ matrices
    RgQ_11 = -1j * k1 * k2 * Rgj21 - 1j * k1**2 * Rgj12
    RgQ_12 = -1j * k1 * k2 * Rgj11 - 1j * k1**2 * Rgj22
    RgQ_21 = -1j * k1 * k2 * Rgj22 - 1j * k1**2 * Rgj11
    RgQ_22 = -1j * k1 * k2 * Rgj12 - 1j * k1**2 * Rgj21

    # Arrange the Q matrices in a 2x2 larger matrix
    Q_matrix = np.block([[Q_11, Q_12], [Q_21, Q_22]])
    RgQ_matrix = np.block([[RgQ_11, RgQ_12], [RgQ_21, RgQ_22]])
    T = -RgQ_matrix @ np.linalg.inv(Q_matrix)

    return T


# def scattering_amplitudes_from_T(
#     theta_inc: float,
#     phi_inc: float,
#     theta_sca: float,
#     phi_sca: float,
#     T: np.ndarray,
#     k1: float,
#     n_max: int,

# ) -> tuple:
#     """
#     Compute S11, S12, S21, S22 from full T matrix using the formula shown.
#     - T is the 2M x 2M matrix assembled as [[T11, T12],[T21, T22]] where
#       M = sum_{n=1..n_max} (2n+1).
#     - Ordering of modes: for n in 1..n_max, m in -n..n (same order as rest of code).
#     - Assumes alpha_{mmnn'} = alpha (scalar) for all indices (default 2).
#     Returns complex tuple (S11, S12, S21, S22).
#     """
#     # build mode list in same ordering used elsewhere
#     modes = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]
#     M = len(modes)
#     if T.shape[0] != 2 * M or T.shape[1] != 2 * M:
#         raise ValueError(f"T must be shape {(2*M, 2*M)}, got {T.shape}")

#     # partition T
#     T11 = T[:M, :M]
#     T12 = T[:M, M:]
#     T21 = T[M:, :M]
#     T22 = T[M:, M:]

#     # build the pi/tau vectors
#     pi_sca = np.empty(M, dtype=np.complex128)
#     tau_sca = np.empty(M, dtype=np.complex128)
#     pi_inc = np.empty(M, dtype=np.complex128)  # will include exp(-i m' phi_inc)
#     tau_inc = np.empty(M, dtype=np.complex128)  # will include exp(-i m' phi_inc)

#     for idx, (n, m) in enumerate(modes):
#         p, t = pi_tau_mn(m, n, theta_sca)
#         pi_sca[idx] = p * np.exp(1j * m * phi_sca)
#         tau_sca[idx] = t * np.exp(1j * m * phi_sca)

#         p_i, t_i = pi_tau_mn(m, n, theta_inc)
#         # incident factor needs exp(- i m' phi_inc) according to formula
#         phase_inc = np.exp(-1j * m * phi_inc)
#         pi_inc[idx] = p_i * phase_inc
#         tau_inc[idx] = t_i * phase_inc

#     # helper for quadratic form a^T B c  (no conj on a)
#     def quad(a, B, c):
#         return float(0) + np.dot(a, B.dot(c))

#     # S11: prefactor 1/k1, left = [pi_sca; tau_sca], right = [pi_inc; tau_inc]
#     S11_terms = (
#         quad(pi_sca, T11, pi_inc)
#         + quad(tau_sca, T21, pi_inc)
#         + quad(pi_sca, T12, tau_inc)
#         + quad(tau_sca, T22, tau_inc)
#     )
#     S11 = (alpha / k1) * S11_terms

#     # S12: prefactor 1/(i k1), left same as S11, right swapped [tau_inc; pi_inc]
#     S12_terms = (
#         quad(pi_sca, T11, tau_inc)
#         + quad(tau_sca, T21, tau_inc)
#         + quad(pi_sca, T12, pi_inc)
#         + quad(tau_sca, T22, pi_inc)
#     )
#     S12 = (alpha / (1j * k1)) * S12_terms

#     # S21: prefactor i/k1, left swapped [tau_sca; pi_sca], right = [pi_inc; tau_inc]
#     S21_terms = (
#         quad(tau_sca, T11, pi_inc)
#         + quad(pi_sca, T21, pi_inc)
#         + quad(tau_sca, T12, tau_inc)
#         + quad(pi_sca, T22, tau_inc)
#     )
#     S21 = (alpha * 1j / k1) * S21_terms

#     # S22: prefactor 1/k1, left swapped [tau_sca; pi_sca], right swapped [tau_inc; pi_inc]
#     S22_terms = (
#         quad(tau_sca, T11, tau_inc)
#         + quad(pi_sca, T21, tau_inc)
#         + quad(tau_sca, T12, pi_inc)
#         + quad(pi_sca, T22, pi_inc)
#     )
#     S22 = (alpha / k1) * S22_terms

#     return S11, S12, S21, S22


def scattering_amplitudes_from_T(
    theta_inc: float,
    phi_inc: float,
    theta_sca: float,
    phi_sca: float,
    T: np.ndarray,
    k1: float,
    n_max: int,
) -> tuple:
    """
    Compute S11, S12, S21, S22 from full T matrix.

    Rows of T correspond to modes (n', m') and columns correspond to modes (n, m).
        The per-mode alpha_{n' m', n m} is computed as:

    alpha = i^{n' - n - 1} (-1)^{m + m'} sqrt[ ((2n+1)(2n'+1)) / (n(n+1) n'(n'+1)) ].

    The scalar 'alpha' argument is kept for compatibility but not used for the
        per-mode scaling.
    """
    # build mode list (same ordering used elsewhere): (n, m) for n=1..n_max, m=-n..n
    modes = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]
    M = len(modes)
    if T.shape[0] != 2 * M or T.shape[1] != 2 * M:
        raise ValueError(f"T must be shape {(2*M, 2*M)}, got {T.shape}")

    # partition T into MxM blocks; rows are (n',m') and cols are (n,m)
    T11 = T[:M, :M]
    T12 = T[:M, M:]
    T21 = T[M:, :M]
    T22 = T[M:, M:]

    # build the pi/tau vectors
    pi_sca = np.empty(M, dtype=np.complex128)
    tau_sca = np.empty(M, dtype=np.complex128)
    pi_inc = np.empty(M, dtype=np.complex128)  # includes exp(-i m phi_inc)
    tau_inc = np.empty(M, dtype=np.complex128)  # includes exp(-i m phi_inc)

    for idx, (n, m) in enumerate(modes):
        p, t = pi_tau_mn(m, n, theta_sca)
        pi_sca[idx] = p * np.exp(1j * m * phi_sca)
        tau_sca[idx] = t * np.exp(1j * m * phi_sca)

        p_i, t_i = pi_tau_mn(m, n, theta_inc)
        phase_inc = np.exp(-1j * m * phi_inc)
        pi_inc[idx] = p_i * phase_inc
        tau_inc[idx] = t_i * phase_inc

    # helper for quadratic form a^T B c
    def quad(a, B, c):
        return np.dot(a, B.dot(c))

    # ---------------------------------------------------------------------
    # Build per-mode alpha matrix with rows = (n',m') and cols = (n,m)
    # formula:
    #   alpha_{n' m', n m} = i^{n' - n - 1} (-1)^{m + m'} sqrt(((2n+1)(2n'+1)) / (n(n+1)n'(n'+1)))
    # ---------------------------------------------------------------------
    n_row = np.array([mode[0] for mode in modes], dtype=int)[
        :, None
    ]  # shape (M,1) -> n'
    n_col = np.array([mode[0] for mode in modes], dtype=int)[
        None, :
    ]  # shape (1,M) -> n
    m_row = np.array([mode[1] for mode in modes], dtype=int)[:, None]  # m'
    m_col = np.array([mode[1] for mode in modes], dtype=int)[None, :]  # m

    # complex phase i^{n' - n - 1}
    phase_factor = (1j) ** (n_row - n_col - 1)
    # sign factor (-1)^{m + m'}
    sign_factor = (-1.0) ** (m_row + m_col)

    coeff = ((2 * n_col + 1) * (2 * n_row + 1)) / (
        (n_col * (n_col + 1) * n_row * (n_row + 1))
    )
    # ensure complex dtype for sqrt of negative/complex phases
    alpha_mat = phase_factor * sign_factor * np.sqrt(coeff.astype(np.complex128))

    T11_a = alpha_mat * T11
    T12_a = alpha_mat * T12
    T21_a = alpha_mat * T21
    T22_a = alpha_mat * T22

    # Build scattering amplitudes via quadratic forms (alpha already applied)
    S11_terms = (
        quad(pi_sca, T11_a, pi_inc)
        + quad(tau_sca, T21_a, pi_inc)
        + quad(pi_sca, T12_a, tau_inc)
        + quad(tau_sca, T22_a, tau_inc)
    )
    S11 = (1.0 / k1) * S11_terms * (-1j * k1)

    S12_terms = (
        quad(pi_sca, T11_a, tau_inc)
        + quad(tau_sca, T21_a, tau_inc)
        + quad(pi_sca, T12_a, pi_inc)
        + quad(tau_sca, T22_a, pi_inc)
    )
    S12 = (1.0 / (1j * k1)) * S12_terms * (-1j * k1)

    S21_terms = (
        quad(tau_sca, T11_a, pi_inc)
        + quad(pi_sca, T21_a, pi_inc)
        + quad(tau_sca, T12_a, tau_inc)
        + quad(pi_sca, T22_a, tau_inc)
    )
    S21 = (1j / k1) * S21_terms * (-1j * k1)

    S22_terms = (
        quad(tau_sca, T11_a, tau_inc)
        + quad(pi_sca, T21_a, tau_inc)
        + quad(tau_sca, T12_a, pi_inc)
        + quad(pi_sca, T22_a, pi_inc)
    )
    S22 = (1.0 / k1) * S22_terms * (-1j * k1)
    return np.vstack((S11, S12, S21, S22)).T.astype(np.complex128)


def scattering_amplitudes_from_T_v2(
    theta_inc: float | np.ndarray,
    phi_inc: float | np.ndarray,
    theta_sca: float | np.ndarray,
    phi_sca: float | np.ndarray,
    T: np.ndarray,
    k1: float,
    n_max: int,
) -> np.ndarray:
    """
    Compute S11, S12, S21, S22 from full T matrix.

    Accepts either scalar angles (floats) or meshgrids/arrays of identical shape
    for the incident and scattered directions. If arrays are passed they must
    have the same shape; the output will be an array of shape (N, 4) where
    N = product(shape) and rows correspond to corresponding pairs of
    (theta_inc[i],phi_inc[i]) and (theta_sca[i],phi_sca[i]) flattened row-major.
    """
    # build mode list (same ordering used elsewhere): (n, m) for n=1..n_max, m=-n..n
    modes = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]
    M = len(modes)
    if T.shape[0] != 2 * M or T.shape[1] != 2 * M:
        raise ValueError(f"T must be shape {(2*M, 2*M)}, got {T.shape}")

    # partition T into MxM blocks; rows are (n',m') and cols are (n,m)
    T11 = T[:M, :M]
    T12 = T[:M, M:]
    T21 = T[M:, :M]
    T22 = T[M:, M:]

    # Normalize inputs to 1D arrays of matching length N
    def to_1d_arr(x):
        if np.isscalar(x):
            return np.atleast_1d(np.array([x], dtype=float))
        a = np.asarray(x)
        return a.ravel()

    theta_inc_1 = to_1d_arr(theta_inc)
    phi_inc_1 = to_1d_arr(phi_inc)
    theta_sca_1 = to_1d_arr(theta_sca)
    phi_sca_1 = to_1d_arr(phi_sca)

    N = theta_inc_1.size  # number of angle pairs

    # build the pi/tau arrays: shape (M, N)
    pi_sca = np.empty((M, N), dtype=np.complex128)
    tau_sca = np.empty((M, N), dtype=np.complex128)
    pi_inc = np.empty((M, N), dtype=np.complex128)  # includes exp(-i m phi_inc)
    tau_inc = np.empty((M, N), dtype=np.complex128)  # includes exp(-i m phi_inc)

    for idx, (n, m) in enumerate(modes):
        p_s, t_s = pi_tau_mn(m, n, theta_sca_1)
        pi_sca[idx, :] = p_s * np.exp(1j * m * phi_sca_1)
        tau_sca[idx, :] = t_s * np.exp(1j * m * phi_sca_1)

        p_i, t_i = pi_tau_mn(m, n, theta_inc_1)
        phase_inc = np.exp(-1j * m * phi_inc_1)
        pi_inc[idx, :] = p_i * phase_inc
        tau_inc[idx, :] = t_i * phase_inc

    # helper to compute per-pair quadratic forms producing vectors length N
    def quad_vec(a_mat, B_mat, c_mat):
        # a_mat, c_mat shape (M,N); B_mat shape (M,M)
        # compute B_mat @ c_mat -> (M,N) then dot with a_mat over axis=0
        return np.sum(a_mat * (B_mat @ c_mat), axis=0)

    # ---------------------------------------------------------------------
    # Build per-mode alpha matrix with rows = (n',m') and cols = (n,m)
    # formula:
    #   alpha_{n' m', n m} = i^{n' - n - 1} (-1)^{m + m'} sqrt(((2n+1)(2n'+1)) / (n(n+1)n'(n'+1)))
    # ---------------------------------------------------------------------
    n_row = np.array([mode[0] for mode in modes], dtype=int)[
        :, None
    ]  # shape (M,1) -> n'
    n_col = np.array([mode[0] for mode in modes], dtype=int)[
        None, :
    ]  # shape (1,M) -> n
    m_row = np.array([mode[1] for mode in modes], dtype=int)[:, None]  # m'
    m_col = np.array([mode[1] for mode in modes], dtype=int)[None, :]  # m

    phase_factor = (1j) ** (n_row - n_col - 1)
    sign_factor = (-1.0) ** (m_row + m_col)
    coeff = ((2 * n_col + 1) * (2 * n_row + 1)) / (
        (n_col * (n_col + 1) * n_row * (n_row + 1))
    )
    alpha_mat = phase_factor * sign_factor * np.sqrt(coeff.astype(np.complex128))

    # Apply alpha elementwise to each T sub-block (rows = n',m' ; cols = n,m)
    T11_a = alpha_mat * T11
    T12_a = alpha_mat * T12
    T21_a = alpha_mat * T21
    T22_a = alpha_mat * T22

    # Build scattering amplitudes via quadratic forms (alpha already applied)
    S11_terms = (
        quad_vec(pi_sca, T11_a, pi_inc)
        + quad_vec(tau_sca, T21_a, pi_inc)
        + quad_vec(pi_sca, T12_a, tau_inc)
        + quad_vec(tau_sca, T22_a, tau_inc)
    )
    # algebra reduces prefactors: S11 = -1j * S11_terms
    S11_vec = -1j * S11_terms

    S12_terms = (
        quad_vec(pi_sca, T11_a, tau_inc)
        + quad_vec(tau_sca, T21_a, tau_inc)
        + quad_vec(pi_sca, T12_a, pi_inc)
        + quad_vec(tau_sca, T22_a, pi_inc)
    )
    S12_vec = -S12_terms  # prefactors cancel

    S21_terms = (
        quad_vec(tau_sca, T11_a, pi_inc)
        + quad_vec(pi_sca, T21_a, pi_inc)
        + quad_vec(tau_sca, T12_a, tau_inc)
        + quad_vec(pi_sca, T22_a, tau_inc)
    )
    S21_vec = -S21_terms  # prefactors cancel

    S22_terms = (
        quad_vec(tau_sca, T11_a, tau_inc)
        + quad_vec(pi_sca, T21_a, tau_inc)
        + quad_vec(tau_sca, T12_a, pi_inc)
        + quad_vec(pi_sca, T22_a, pi_inc)
    )
    S22_vec = -1j * S22_terms

    # Stack into (N,4) array: columns [S11, S12, S21, S22]
    S_all = np.vstack((S11_vec, S12_vec, S21_vec, S22_vec)).T.astype(np.complex128)
    return S_all
