import os

# Change to a new directory
new_path = r"/home/sdutta/code/random-matrix"
os.chdir(new_path)
# print(os.getcwd())
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
    if isinstance(theta, (list, cp.ndarray)):
        theta[theta == 0] = cp.sqrt(
            2 * cp.finfo(cp.float64).eps
        )  # remove issues with divide by zero
        theta[theta == cp.pi] = cp.pi + cp.sqrt(
            2 * cp.finfo(cp.float64).eps
        )  # remove issues with divide by zero

    elif isinstance(theta, int):
        if theta == 0:
            theta = cp.sqrt(2 * cp.finfo(cp.float64).eps)
        elif theta == cp.pi:
            theta = cp.pi + cp.sqrt(2 * cp.finfo(cp.float64).eps)

    Ln = lpmv(m, n, cp.cos(theta))
    Lnp1 = lpmv(m, n + 1, cp.cos(theta))

    norm = cp.sqrt(factorial(n - m) / factorial(n + m))
    pi_out = norm * m * Ln / (cp.sin(theta))
    tau_out = norm * (
        -(1 + n) * cp.cos(theta) * Ln / (cp.sin(theta))
        + (1 - m + n) * Lnp1 / cp.sin(theta)
    )

    return (pi_out, tau_out)


"""
This function calculates the constant used to convert the Cmn(theta)
function to Cmn(theta,phi) function.

"""


@lru_cache(maxsize=None)  # Memoize for all (m, n) pairs
def const_mn(m, n):

    return ((-1) ** (m)) * cp.sqrt(factorial(n + m) / factorial(n - m))


"""
This function calcultes the gamma_mn constant used in the calculation of the
RgN and RgM functions.

"""


@lru_cache(maxsize=None)  # Memoize for all (m, n) pairs
def gamma_mn(m, n):
    if n == 0:  # To avoid division by zero
        return 0
    return cp.sqrt(
        ((2 * n + 1) * factorial(n - m)) / (4 * cp.pi * n * (n + 1) * factorial(n + m))
    )


# Coordinate transformation from cartesian to spherical
def cart2sph(x, y, z):
    xy = x**2 + y**2
    r = cp.sqrt(xy + z**2)
    t = cp.pi / 2 - cp.arctan2(z, cp.sqrt(xy))
    p = cp.arctan2(y, x)
    if cp.all(p < 0):
        p = p + 2 * cp.pi

    return (r, t, p)


def sph2cart_comp_mat(theta, phi):
    R = cp.array(
        [
            [
                cp.cos(phi) * cp.sin(theta),
                cp.cos(phi) * cp.cos(theta),
                -cp.sin(phi),
            ],
            [
                cp.sin(phi) * cp.sin(theta),
                cp.sin(phi) * cp.cos(theta),
                cp.cos(phi),
            ],
            [cp.cos(theta), -cp.sin(theta), 0 * theta],
        ]
    )

    return R


# Converts a vector from spherical to cartesian coordinates
def sph2cart_comp_vec(U, theta, phi):
    R = sph2cart_comp_mat(theta, phi)
    Uout = cp.einsum("mn...,n...->m...", R, U)

    return Uout


def B_mn(m, n, theta, phi):
    c = const_mn(m, n) * cp.exp(1j * m * phi)

    pitaumn = pi_tau_mn(m, n, theta)
    theta_comp = c * pitaumn[1]
    phi_comp = 1j * c * pitaumn[0]
    return (theta_comp, phi_comp)


def C_mn(m, n, theta, phi):
    c = const_mn(m, n) * cp.exp(1j * m * phi)

    pitaumn = pi_tau_mn(m, n, theta)
    theta_comp = 1j * c * pitaumn[0]
    phi_comp = -c * pitaumn[1]
    return (theta_comp, phi_comp)


def d_mn(m, n, theta):
    x = cp.cos(theta)
    lp = lpmv(m, n, x)
    return lp * (cp.sqrt(factorial(n - m) / factorial(n + m)))


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
    c = const_mn(m, n) * cp.exp(1j * m * phi)
    return c * d_mn(m, n, theta)


def RgM_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)  # Cached value
    Cmn = C_mn(m, n, theta, phi)
    theta_comp = gamma * j_n(n, kr) * Cmn[0]
    phi_comp = gamma * j_n(n, kr) * Cmn[1]

    return cp.array([theta_comp, phi_comp], dtype=cp.complex128)


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

    return cp.array([r_comp, theta_comp, phi_comp])


def N_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)
    Bmn = B_mn(m, n, theta, phi)
    r_comp = gamma * (n * (n + 1) / kr) * h_n(n, kr) * P_mn(m, n, theta, phi)
    theta_comp = gamma * (h_n(n, kr) / kr + dh_n(n, kr)) * Bmn[0]
    phi_comp = gamma * (h_n(n, kr) / kr + dh_n(n, kr)) * Bmn[1]

    return cp.array([r_comp, theta_comp, phi_comp], dtype=cp.complex128)


def M_mn(m, n, kr, theta, phi):
    gamma = gamma_mn(m, n)  # Cached value
    Cmn = C_mn(m, n, theta, phi)
    theta_comp = gamma * h_n(n, kr) * Cmn[0]
    phi_comp = gamma * h_n(n, kr) * Cmn[1]
    return cp.array([theta_comp, phi_comp], dtype=cp.complex128)


"""
This function creates the Convex Hull from a set of points in (x,y,z)
It uses uniform sampling in theta and phi (Not the best way of sampling in theta)

"""


def create_hull_uniform():
    row = 50
    col = 51
    num_linear = row * col
    theta = np.linspace(0, np.pi, col)
    phi = np.linspace(0, 2 * cp.pi, row)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    r = 600e-9  # particle size is 600nm
    x = np.reshape((r * np.sin(theta_grid) * np.cos(phi_grid)), (1, num_linear))
    y = np.reshape((r * np.sin(theta_grid) * np.sin(phi_grid)), (1, num_linear))
    z = np.reshape((r * np.cos(theta_grid)), (1, num_linear))
    points = np.transpose(np.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


def create_hull_random():
    num_points = 10000
    points = cp.random.rand(num_points, 3)
    points = points / cp.linalg.norm(points, axis=1, keepdims=True) * 600e-9
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
    u = cp.linspace(0, 1, col)
    theta = cp.arccos(1 - 2 * u)
    phi = cp.linspace(0, 2 * cp.pi, row)
    theta_grid, phi_grid = cp.meshgrid(theta, phi)
    r = 600e-9  # particle size is 600nm
    x = cp.reshape((r * cp.sin(theta_grid) * cp.cos(phi_grid)), (1, num_linear))
    y = cp.reshape((r * cp.sin(theta_grid) * cp.sin(phi_grid)), (1, num_linear))
    z = cp.reshape((r * cp.cos(theta_grid)), (1, num_linear))
    points = cp.transpose(cp.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


def create_hull_GC():
    row = 100
    col = 101
    num_linear = row * col

    theta = cp.pi * (cp.arange(1, col + 1) - 0.5) / col
    phi = cp.linspace(0, 2 * cp.pi, row)
    theta_grid, phi_grid = cp.meshgrid(theta, phi)
    r = 600e-9  # particle size is 600nm
    x = cp.reshape((r * cp.sin(theta_grid) * cp.cos(phi_grid)), (1, num_linear))
    y = cp.reshape((r * cp.sin(theta_grid) * cp.sin(phi_grid)), (1, num_linear))
    z = cp.reshape((r * cp.cos(theta_grid)), (1, num_linear))
    points = cp.transpose(cp.vstack((x, y, z)))
    hull = ConvexHull(points)
    print("Number of simplices:", hull.simplices.shape[0])
    return hull


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
    weights = 1  # cp.pi / 101

    hull = hull
    modes = sum(2 * n + 1 for n in range(1, n_max + 1))
    print(f"Number of modes: {modes} ")
    # modes for exterior region (m,n)
    # For a given n m runs from -n to +n
    modes_nm = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]
    # modes for the interior region (m',n')
    modes_nmp = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1)]

    j11 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    j12 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    j21 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    j22 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    Rgj11 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    Rgj12 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    Rgj21 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)
    Rgj22 = cp.zeros((len(modes_nmp), len(modes_nm)), dtype=cp.complex128)

    for i in range(0, modes):  # for each mode in the interior region
        for j in tqdm(range(0, modes)):  # for each mode in the exterior region

            def integrandj11(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                # RgM = cp.hstack((cp.zeros(6),RgM))
                M = M_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM[0] * M[1] - RgM[1] * M[0]
                theta_comp = 0 * M[1]
                phi_comp = 0 * RgM[0]
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j11[j][i] = hull_surface_integral_vector(integrandj11, hull, use_gpu=True)
            j11[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj12(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                N = N_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM[0] * N[2] - RgM[1] * N[1]
                theta_comp = RgM[1] * N[0]
                phi_comp = -RgM[0] * N[0]
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j12[j][i] = hull_surface_integral_vector(integrandj12, hull, use_gpu=True)
            j12[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj21(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                M = M_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN[1] * M[1] - RgN[2] * M[0]
                theta_comp = -RgN[0] * M[1]
                phi_comp = RgN[0] * M[0]
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j21[j][i] = hull_surface_integral_vector(integrandj21, hull, use_gpu=True)
            j21[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandj22(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgN = RgN_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)

                N = N_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgN[1] * N[2] - RgN[2] * N[1]
                theta_comp = -RgN[0] * N[2] + RgN[2] * N[0]
                phi_comp = RgN[0] * N[1] - RgN[1] * N[0]
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            j22[j][i] = hull_surface_integral_vector(integrandj22, hull, use_gpu=True)
            j22[j][i] *= (-1) ** (modes_nm[i][1])

            def integrandRgj11(input):

                x, y, z = input[:, 0], input[:, 1], input[:, 2]
                r, theta, phi = cart2sph(x, y, z)

                RgM1 = RgM_mn(modes_nmp[j][1], modes_nmp[j][0], k2 * r, theta, phi)
                # RgM = cp.hstack((cp.zeros(6),RgM))
                RgM2 = RgM_mn(-modes_nm[i][1], modes_nm[i][0], k1 * r, theta, phi)

                r_comp = RgM1[0] * RgM2[1] - RgM1[1] * RgM2[0]
                theta_comp = 0 * RgM2[1]
                phi_comp = 0 * RgM1[0]
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj11[j][i] = hull_surface_integral_vector(
                integrandRgj11, hull, use_gpu=True
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
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj12[j][i] = hull_surface_integral_vector(
                integrandRgj12, hull, use_gpu=True
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
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj21[j][i] = hull_surface_integral_vector(
                integrandRgj21, hull, use_gpu=True
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
                cross_prod = cp.array([r_comp, theta_comp, phi_comp])
                cp_cart = sph2cart_comp_vec(cross_prod, theta, phi)  # do not normalise

                return weights * cp_cart.transpose((1, 0))

            Rgj22[j][i] = hull_surface_integral_vector(
                integrandRgj22, hull, use_gpu=True
            )
            Rgj22[j][i] *= (-1) ** (modes_nm[i][1])

    return j11, j12, j21, j22, Rgj11, Rgj12, Rgj21, Rgj22


def get_T(hull, k1, k2, n_max):
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
    Q_matrix = cp.block([[Q_11, Q_12], [Q_21, Q_22]])
    RgQ_matrix = cp.block([[RgQ_11, RgQ_12], [RgQ_21, RgQ_22]])
    T = -RgQ_matrix @ cp.linalg.inv(Q_matrix)

    return T


"""
hull, points = create_hull()
wavelength1 = 532e-9
k1 = (2 * cp.pi) / wavelength1
wavelength2 = 266e-9
k2 = 1.5 * k1
n_max = 2
modes = sum(2 * n + 1 for n in range(1, n_max + 1))
print(f"Total number of modes is {modes}")
T = get_T(hull, points, k1, k2, n_max)
plt.matshow(cp.abs(T), cmap="viridis")
plt.title("T Matrix")
plt.colorbar()
plt.show()
"""
