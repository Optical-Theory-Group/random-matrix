import numpy as np
import matplotlib.pyplot as plt
import random_matrix.amplitude_matrix.isotropic_sphere as rm

wavelength = 400e-9
n = 100  # number of samples
rri = 1.2
m = np.reshape(rri * np.ones((n * (n + 1))), (n + 1, n))
k = (2 * np.pi) / wavelength

# sampling incident field
theta_i = np.pi / 4 * np.ones((100))
phi_i = np.pi / 8 * np.ones((101))
theta_grid_i, phi_grid_i = np.meshgrid(theta_i, phi_i)

# Scattered field
ki_z = np.cos(theta_grid_i)
ki_x = np.sin(theta_grid_i) * np.cos(phi_grid_i)
ki_y = np.sin(theta_grid_i) * np.sin(phi_grid_i)

# sampling scattered field
theta = np.linspace(0, np.pi, n)
phi = np.linspace(0, 2 * np.pi, n + 1)
theta_grid, phi_grid = np.meshgrid(theta, phi)
# Scattered field
ks_z = np.cos(theta_grid)
ks_x = np.sin(theta_grid) * np.cos(phi_grid)
ks_y = np.sin(theta_grid) * np.sin(phi_grid)


size_param = np.linspace(0.01, 50, 1000)
C_scaT = np.zeros((1000))
radius1 = size_param / k

d_theta = np.pi / (n - 1)
d_phi = np.pi / (n)


for i in range(0, 1000):
    x = np.reshape((size_param[i] * np.ones((n * (n + 1)))), (n + 1, n))
    A = rm.get_A(ki_x, ki_y, ki_z, ks_x, ks_y, ks_z, x, m)
    S1 = A[3, :, :]
    S2 = A[0, :, :]
    S3 = A[1, :, :]
    S4 = A[2, :, :]
    T = (np.abs(S2) ** 2 + np.abs(S4) ** 2) * np.sin(theta_grid) / k**2
    inner_integral = np.trapezoid(T, phi, d_phi, axis=0)
    C_scaT[i] = np.trapezoid(inner_integral, theta, d_theta) / (np.pi * radius1[i] ** 2)

fig, ax = plt.subplots()
plt.plot(size_param, C_scaT, "b-", label=f"m = {rri}")

ax.set_title("$Q_{sca}$ for isotropic sphere \u03BB=400nm")
ax.set_xlabel("x size parameter")
ax.set_ylabel("$Q_{sca}$")
plt.legend()
fig.savefig("/home/sdutta/code/random-matrix/examples/iso_LCP.png")
