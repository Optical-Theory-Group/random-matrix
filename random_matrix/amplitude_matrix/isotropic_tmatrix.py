import os

# Change to a new directory
new_path = r"/home/sdutta/code/random-matrix"
os.chdir(new_path)

import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.special import hankel1, spherical_jn, lpmv
from scipy.integrate import quad, dblquad
from scipy.spatial import ConvexHull
from random_matrix.utils import integration_utils as iu
import math

count = 0

"""
class ArbParticle:
    def __init__(
        self, count
    ):  # other object variables that maybe needed can be added after the comma
        self.count = 0  # placeholder
"""


# @staticmethod
# This function defines the Wigner-D functions for a given m,n, and theta from the Associalted LP functions
def d_mn(m, n, theta):
    x = np.cos(theta)
    lp = lpmv(m, n, x)
    return np.sqrt(factorial(n - m) / factorial(n + m)) * lp


def j_n(n, kr):
    return spherical_jn(n, kr)


def h_n(n, kr):
    return hankel1(n, kr)


def M_mn(m, n, kr, theta, phi):
    gamma_mn = np.sqrt(
        ((2 * n + 1) * factorial(n - m)) / (4 * np.pi * n * (n + 1) * factorial(n + m))
    )
    res = gamma_mn * h_n(n, kr) * C_mn(m, n, theta, phi)
    return res


def RgM_mn(m, n, kr, theta, phi):
    gamma_mn = np.sqrt(
        ((2 * n + 1) * factorial(n - m)) / (4 * np.pi * n * (n + 1) * factorial(n + m))
    )
    res = gamma_mn * j_n(n, kr) * C_mn(m, n, theta, phi)
    return res


def N_mn():
    pass


def RgN_mn():
    pass


# @staticmethod
# This function generates the pi and tau functions for the angular parts of the VSHs
def pi_tau_mn(m, n, theta):
    """
    theta[theta == 0] = np.sqrt(
        2 * np.finfo(np.float64).eps
    )  # remove issues with divide by zero
    theta[theta == np.pi] = np.pi + np.sqrt(
        2 * np.finfo(np.float64).eps
    )  # remove issues with divide by zero
    """
    Ln = lpmv(np.abs(m), n, np.cos(theta))
    Lnp1 = lpmv(np.abs(m), n + 1, np.cos(theta))

    pi_out = (m * d_mn(m, n, theta)) / (np.sin(theta))

    tau_out = (np.sqrt(factorial(n - m) / factorial(n + m))) * (
        -(1 + n) * (np.cos(theta) / np.sin(theta)) * Ln
        + (1 - np.abs(m) + n) * Lnp1 / (np.sin(theta))
    )

    return (pi_out, tau_out)


# @staticmethod
def B_mn(m, n, theta, phi):
    c = (
        (-1) ** (m)
        * np.sqrt(factorial(n + m) / factorial(n - m))
        * np.exp(1j * m * phi)
    )
    theta_comp = c * pi_tau_mn(m, n, theta)[0]
    phi_comp = 1j * c * pi_tau_mn(m, n, theta)[1]
    return (theta_comp, phi_comp)


# @staticmethod
def C_mn(m, n, theta, phi):
    c = (
        (-1) ** (m)
        * np.sqrt(factorial(n + m) / factorial(n - m))
        * np.exp(1j * m * phi)
    )
    theta_comp = 1j * c * pi_tau_mn(m, n, theta)[1]
    phi_comp = -c * pi_tau_mn(m, n, theta)[0]

    # print(res
    return (theta_comp, phi_comp)


# @staticmethod
def orthonormality_check(n1, m1, n2, m2):

    def integrand(x):
        # Weight function: (1 - x^2)^(m/2) and the product of the two polynomials
        p1 = lpmv(l1, m1, x)
        p2 = lpmv(l2, m2, x)
        return p1 * p2 * (1 - x**2) ** (m1 / 2)

    def integrand2(theta, phi):
        B1 = B_mn(m1, n1, theta, phi)
        B2 = B_mn(m2, n2, theta, phi)
        return (B1[0] * np.conj(B2[0]) + B1[1] * np.conj(B2[1])) * np.sin(theta)

    def integrand3(theta, phi):
        C1 = C_mn(m1, n1, theta, phi)
        C2 = C_mn(m2, n2, theta, phi)
        return (C1[0] * np.conj(C2[0]) + C1[1] * np.conj(C2[1])) * np.sin(theta)

    # Compute the integral over [-1, 1]
    result, _ = dblquad(integrand2, 0, 2 * np.pi, 0, np.pi)
    return result


def samples(n):
    # Returns the coordinates of the vertices of the nth simplex
    # Define theta and _phi_grid values
    row = 50
    col = 51
    num_linear = row * col
    theta = np.linspace(0, np.pi, col)
    phi = np.linspace(0, 2 * np.pi, row)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    r = 1
    # Calculate x, y, z coordinates for the sphere
    x = np.reshape((r * np.sin(theta_grid) * np.cos(phi_grid)), (1, num_linear))
    y = np.reshape((r * np.sin(theta_grid) * np.sin(phi_grid)), (1, num_linear))
    z = np.reshape((r * np.cos(theta_grid)), (1, num_linear))
    points = np.transpose(np.vstack((x, y, z)))
    # Compute the convex hull
    hull = ConvexHull(points)
    return len(hull.simplices), points[hull.simplices[n]]


def GramMatrix(n):
    # Here n is the simplex number. Function returns the det of the Gram matrix for the nth simplex
    points = samples(n)[1]
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    J = np.array([[np.dot(v1, v1), np.dot(v1, v2)], [np.dot(v2, v1), np.dot(v2, v2)]])
    G = np.sqrt(np.linalg.det(J))
    return G


def function(a, b):
    global count
    points = samples(count)[1]
    vertex = points[0]
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    x = vertex[0] + a * v1[0] + b * v2[0]
    y = vertex[1] + a * v1[1] + b * v2[1]
    z = vertex[2] + a * v1[2] + b * v2[2]
    r, theta, phi = cart2sph(x, y, z)
    B1 = B_mn(1, 1, theta, phi)
    B2 = B_mn(1, 1, theta, phi)

    return GramMatrix(count) * (B1[0] * np.conj(B2[0]) + B1[1] * np.conj(B2[1]))
    # return GramMatrix(count) * theta


def cart2sph(x, y, z):
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    t = np.pi / 2 - np.arctan2(z, np.sqrt(xy))
    p = np.arctan2(y, x)
    if np.all(p < 0):
        p = p + 2 * np.pi

    return (r, t, p)


def orthotest_simplex():
    global count
    count = 0
    res = 0
    num = samples(0)[0]
    for i in range(0, num):
        res = res + iu.basic_triangle_integral(function, [[0, 0], [1, 0], [0, 1]])
        count += 1
    return res


"""


def integrand(x, y):
    return x / x


norms = []
centroids = []
test_points = []

for simplex in hull.simplices:
    test_point = points[simplex[1]]
    test_points.append(test_point)
    v1 = points[simplex[1]] - points[simplex[0]]
    v2 = points[simplex[2]] - points[simplex[0]]
    centroid = (
        [np.sum(points[simplex, 0]) / 3],
        [np.sum(points[simplex, 1]) / 3],
        [np.sum(points[simplex, 2]) / 3],
    )
    centroids.append(centroid)
    norm = np.cross(v1, v2)
    norms.append(norm)
    J = np.array([[np.dot(v1, v1), np.dot(v1, v2)], [np.dot(v2, v1), np.dot(v2, v2)]])
    G = np.sqrt(np.linalg.det(J))
    area = area + G * iu.basic_triangle_integral(integrand, [[0, 0], [0, 1], [1, 0]])

print(area)


for i, n in enumerate(norms):
    if np.allclose(-norms[0], n):
        print(i)
        break


norms = np.array(norms)
centroids = np.reshape(np.array(centroids), [32, 3])
test_points = np.array(test_points)
# test_points = np.reshape(np.array(test_points), [162, 3, 1])
test1 = np.sum(norms * test_points, axis=1)
indices1 = np.where(test1 < 0)
test2 = np.sum(norms * centroids, axis=1)
indices2 = np.where(test2 < 0)


# Plotting the points and the convex hull
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the points
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="b", label="Points")
ax.scatter(norms[:, 0], norms[:, 1], norms[:, 1], "r-")
# ax.scatter(
# test_points[:, 0], test_points[:, 1], test_points[:, 2], color="g", label="TP"
# )

# Plot the convex hull
for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], "k-")

# Labeling
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Convex Hull in 3D")
plt.legend()
plt.show()


# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, color="b", edgecolor="k", alpha=0.7)

# Set plot labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Sphere Plot")

plt.show()


# Create the plot
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis")])
fig.update_layout(title="Interactive 3D Sphere Plot", autosize=True)
fig.show()



s = np.zeros((row, col, 3))
s[:, :, 0] = x
s[:, :, 1] = y
s[:, :, 2] = z

# Calculating dr_dtheta
drx_dt = np.gradient(x, dt, axis=1, edge_order=1)
dry_dt = np.gradient(y, dt, axis=1, edge_order=1)
drz_dt = np.gradient(z, dt, axis=1, edge_order=1)

V1 = np.zeros((row, col, 3))
V1[:, :, 0] = drx_dt
V1[:, :, 1] = dry_dt
V1[:, :, 2] = drz_dt
V1 = np.reshape(V1, (num_linear, 3))

# Calculating dr_dphi
drx_dp = np.gradient(x, dp, axis=0, edge_order=1)
dry_dp = np.gradient(y, dp, axis=0, edge_order=1)
drz_dp = np.gradient(z, dp, axis=0, edge_order=1)

V2 = np.zeros((row, col, 3))
V2[:, :, 0] = drx_dp
V2[:, :, 1] = dry_dp
V2[:, :, 2] = drz_dp
V2 = np.reshape(V2, (num_linear, 3))

# Calculating surface normals
surf_norm1 = np.cross(V1, V2)
norms = np.linalg.norm(surf_norm1, axis=-1)

# Check for 0 norm
indices = np.where(np.isclose(norms, 0.0))
norms[indices] = 1.0

surf_norm = surf_norm1 / norms[:, np.newaxis]
surf_norm = np.reshape(surf_norm, (row, col, 3))

"""
