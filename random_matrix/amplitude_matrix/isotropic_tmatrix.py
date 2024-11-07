import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# Define theta and phi values
theta = np.linspace(0, np.pi, 50)
dt = np.pi / 50
phi = np.linspace(0, 2 * np.pi, 51)
dp = (2 * np.pi) / 51
theta_grid, phi_grid = np.meshgrid(theta, phi)

r = 1

# Calculate x, y, z coordinates for the sphere
x = r * np.sin(theta_grid) * np.cos(phi_grid)
y = r * np.sin(theta_grid) * np.sin(phi_grid)
z = r * np.cos(theta_grid)

"""

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
"""

s = np.zeros((51, 50, 3))
s[:, :, 0] = x
s[:, :, 1] = y
s[:, :, 2] = z
dr_dt = np.gradient(s, dt, axis=1, edge_order=1)
# drx_dt = np.gradient(x, dt, axis=1, edge_order=1)
# dry_dt = np.gradient(y, dt, axis=1, edge_order=1)
# drz_dt = np.gradient(z, dt, axis=1, edge_order=1)
