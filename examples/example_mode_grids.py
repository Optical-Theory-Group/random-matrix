"""Example grids to demonstrate the mode_grid_generator module"""

import os

import numpy as np

from random_matrix.modes import mode_grid_factory

np.random.seed(25)

# -----------------------------------------------------------------------------
# Polar grids
# -----------------------------------------------------------------------------

# Standard polar grid rotated
mode_grid_factory.from_dr_dt(
    dr=0.1,
    dt=2 * np.pi / 12,
    r_lim=1.0,
    rotation_angle=0.0,
).plot()

# Irregular polar grid from random r and t values
r_vals = np.random.uniform(0.0, 2.0, 10)
t_vals = np.random.uniform(0.0, 2 * np.pi, 10)
mode_grid_factory.from_rt_vals(
    r_vals=r_vals, t_vals=t_vals, include_central_mode=False
).plot()

# -----------------------------------------------------------------------------
# Tilings
# -----------------------------------------------------------------------------

# Hexagons
mode_grid_factory.from_tiling(
    tiling_type="hexagons",
    side_length=0.2,
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=2.0,
    translation_vector=np.array([0.1, 0.0]),
).plot()

# Triangles
mode_grid_factory.from_tiling(
    tiling_type="triangles",
    side_length=0.35,
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=0.7,
    translation_vector=np.array([0.0, 0.0]),
).plot()

# Rectangles
mode_grid_factory.from_tiling(
    tiling_type="rectangles",
    side_length=(0.2, 0.2),
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
).plot()

# -----------------------------------------------------------------------------
# Random grids
# -----------------------------------------------------------------------------

# Delaunay triangles
mode_grid_factory.from_random(
    num_points=500,
    r_lim=2.0,
    random_type="delaunay",
    grid_wave_type="all",
).plot()

# Voronoi regions
mode_grid_factory.from_random(
    num_points=500,
    r_lim=2.0,
    random_type="voronoi",
    grid_wave_type="all",
).plot()
