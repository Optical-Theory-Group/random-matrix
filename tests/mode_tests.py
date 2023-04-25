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

np.random.seed(1)

#####################
# Mode module tests #
#####################

# Cartesian test from dx, dy
grid_data = {
    "t_offset": 0.0,
    "grid_type": "cartesian",
    "grid_wave_type": "propagating",
}
dx = 0.2
dy = 0.2
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)
print("Standard grid with propagating modes")
print(my_grid)
print("------------")

# Cartesian test from dx, dy
# Only evanescent modes and rotated
grid_data = {
    "t_offset": 0.1,
    "grid_type": "cartesian",
    "grid_wave_type": "evanescent",
}
dx = 0.2
dy = 0.2
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)
print("Standard grid with evanescent modes, rotated")
print(my_grid)
print("------------")

# Cartesian test from dx, dy
# Both types of modes and rotated
grid_data = {
    "t_offset": 1.1,
    "grid_type": "cartesian",
    "grid_wave_type": "all",
}
dx = 0.3
dy = 0.3
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)
print("Standard grid with all modes, rotated")
print(my_grid)
print("------------")


# Slightly shifted cartesian grid
# Not recirpcoal
grid_data = {
    "t_offset": 0.0,
    "grid_type": "cartesian",
    "grid_wave_type": "propagating",
}

x_vals = np.linspace(-1.0, 1.1, 10)
y_vals = np.linspace(-1.0, 1.0, 10)
my_grid = ModeGrid.from_xy_vals(
    x_vals=x_vals, y_vals=y_vals, grid_data=grid_data
)
my_grid.plot(show_indices=True)
print("Slightly shifted non-reciprocal grid")
print(my_grid)
print("------------")

# Wild Cartesian grid from random x and y values
grid_data = {
    "t_offset": 0.0,
    "grid_type": "cartesian",
    "grid_wave_type": "all",
}
x_vals = np.random.randn(7)
y_vals = np.random.randn(7)
my_grid = ModeGrid.from_xy_vals(
    x_vals=x_vals, y_vals=y_vals, grid_data=grid_data
)
my_grid.plot(show_indices=True)
print("Wild grid with random x and y values")
print(my_grid)
