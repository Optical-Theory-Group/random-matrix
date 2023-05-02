import numpy as np

from random_matrix import mode_grid_generator

np.random.seed(0)

# -----------------------------------------------------------------------------
# Tilings
# -----------------------------------------------------------------------------

my_grid = mode_grid_generator.from_tiling(
    tiling_type="hexagons",
    side_length=0.25,
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot()


my_grid = mode_grid_generator.from_tiling(
    tiling_type="triangles",
    side_length=0.35,
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot()


my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=0.25,
    r_lim=2.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot()


# -----------------------------------------------------------------------------
# Random grids
# -----------------------------------------------------------------------------

my_grid = mode_grid_generator.from_random(
    num_points=500,
    r_lim=2.0,
    random_type="delaunay",
    grid_wave_type="all",
)
my_grid.plot()


my_grid = mode_grid_generator.from_random(
    num_points=500,
    r_lim=2.0,
    random_type="voronoi",
    grid_wave_type="all",
)
my_grid.plot()
