import numpy as np
from skspatial.objects import Circle, Line

from random_matrix.mode import Mode
from random_matrix.mode_grid import ModeGrid
from random_matrix.utils.geometry_utils import (
    cartesian_to_polar,
    is_rectangle,
    polar_to_cartesian,
    rotate_points,
)
from random_matrix.utils.plotting_utils import draw_convex_polygon

np.random.seed(128)

#####################
# Mode module tests #
#####################

# Random individual mode
r_vals = np.random.random(2)
t_vals = np.random.random(2) * 2 * np.pi
r_grid, t_grid = np.meshgrid(r_vals, t_vals)
points_polar = np.column_stack((r_grid.ravel(), t_grid.ravel()))
points_cartesian = polar_to_cartesian(points_polar)
my_mode = Mode(mode_boundary=points_cartesian, is_polar=True)
my_mode.plot()

# Polar grid of modes
num_r = 8
num_t = 8
r_vals = np.linspace(0.0, 1.0, num_r + 1)
t_vals = np.linspace(0.0, 2 * np.pi, num_t + 1)
grid_data = {"r_vals": r_vals, "t_vals": t_vals, "grid_type": "polar"}
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)

# Rotated polar grid (0.1 radians)
grid_data["t_offset"] = 0.1
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)

# Weird non reciprocal case
r_vals = np.array([0.2, 0.6, 0.7])
t_vals = np.array([0.5, 1.0, 1.4, 2.0, 3.0, 4.5, 5.0, 6.1])
grid_data = {"r_vals": r_vals, "t_vals": t_vals, "grid_type": "polar"}
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)

# Individual quadrilateral mode
points = np.array([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
mode = Mode(mode_boundary=points)
mode.plot(show_triangulation=False)
mode.plot(show_triangulation=True)
print(mode)

# Convexh hull of a lot of points
points = 0.2 * np.random.randn(10**5, 2)
mode = Mode(mode_boundary=points)
mode.plot(show_triangulation=False)
mode.plot(show_triangulation=True)

# Cartesian grid
grid_data = {
    "dx": 0.2,
    "dy": 0.2,
    "grid_type": "cartesian",
    "t_offset": 0.0,
    "grid_wave_type": "propagating",
}
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)

# Cartesian grid with larger limit and just evanescent modes
grid_data = {
    "dx": 0.2,
    "dy": 0.2,
    "x_lim": 1.5,
    "y_lim": 1.5,
    "grid_type": "cartesian",
    "t_offset": 0.0,
    "grid_wave_type": "evanescent",
}
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)

# Cartesian grid with larger limit and all modes
grid_data = {
    "dx": 0.25,
    "dy": 0.25,
    "x_lim": 1.5,
    "y_lim": 1.5,
    "grid_type": "cartesian",
    "t_offset": 0.0,
    "grid_wave_type": "all",
}
modes = ModeGrid(grid_data=grid_data)
modes.plot(show_indices=True)
