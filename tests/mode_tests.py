import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # add parent directory to sys.path

from random_matrix.mode import Mode
from random_matrix.mode_grid import ModeGrid
from random_matrix.utils.geometry_utils import polar_to_cartesian, cartesian_to_polar, is_rectangle, rotate_points
from random_matrix.utils.plotting_utils import draw_convex_polygon

np.random.seed(128)

#####################
# Mode module tests #
#####################

###############
# Random individual mode
r_vals = np.random.random(2)
t_vals = np.random.random(2)*2*np.pi
r_grid, t_grid = np.meshgrid(r_vals, t_vals)
points_polar = np.column_stack((r_grid.ravel(), t_grid.ravel()))
points_cartesian = polar_to_cartesian(points_polar)
my_mode = Mode(mode_boundary=points_cartesian, mode_shape_type="polar")
my_mode.plot()

############
# Polar grid of modes
# 
num_r = 8
num_t = 8
r_values = np.linspace(0.0, 1.0, num_r+1)
t_values = np.linspace(0.0, 2*np.pi, num_t+1)
mode_boundary_data = {"r_vals": r_values, "t_vals": t_values, "mode_shape_type": "polar"}
modes = ModeGrid(mode_boundary_data=mode_boundary_data)
modes.plot()

# Rotate grid by 0.1 radians
mode_boundary_data["t_offset"] = 0.1
new_modes = ModeGrid(mode_boundary_data=mode_boundary_data)
new_modes.plot()

#######
# Cartesian mode
#

points = np.array([[0.3,0.0],[0.6,0.0],[0.3,0.5],[0.6,0.5]])
my_mode = Mode(index=0, mode_boundary=points,mode_shape_type="cartesian")
my_mode.plot()
print(my_mode.weight)

rotated_points = rotate_points(points, np.array([0.0,0.0]), rotation_angle=1.0)
my_mode = Mode(index=0, mode_boundary=rotated_points,mode_shape_type="cartesian")
my_mode.plot()
print(my_mode.weight)