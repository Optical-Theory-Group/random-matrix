"""Example grids to demonstrate the mode_grid_generator module"""

from pathlib import Path
import sys
import os
import warnings

# warnings.filterwarnings("error")
# Add parent directory to Python path
project_root = Path("/home/nbyrnes/code/random-matrix/")  # <-- adjust this
sys.path.insert(0, str(project_root))

import numpy as np

from random_matrix.modes import mode_grid_factory

np.random.seed(25)

# -----------------------------------------------------------------------------
# Polar grids
# -----------------------------------------------------------------------------

# Irregular polar grid from random r and t values
dr = 0.1
dt_divisor = 20
mg = mode_grid_factory.from_dr_dt(
    dr, 2 * np.pi / dt_divisor, include_central_mode=False, r_lim=1.0
)
fig, _ = mg.plot()
fig.savefig("/home/nbyrnes/code/random-matrix/rmt_analysis/grid_polar.svg",format="svg")
# -----------------------------------------------------------------------------
# Tilings
# -----------------------------------------------------------------------------

# Hexagons
mg = mode_grid_factory.from_tiling(
    tiling_type="hexagons",
    side_length=0.1,
    r_lim=1.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
fig, _ = mg.plot()
fig.savefig("/home/nbyrnes/code/random-matrix/rmt_analysis/grid_hexagon.svg",format="svg")

# Triangles
# mg = mode_grid_factory.from_tiling(
#     tiling_type="triangles",
#     side_length=0.35,
#     r_lim=1.0,
#     grid_wave_type="all",
#     rotation_angle=0.7,
#     translation_vector=np.array([0.0, 0.0]),
# )
# fig = mg.plot()

# Rectangles
mg = mode_grid_factory.from_tiling(
    tiling_type="rectangles",
    side_length=(0.15, 0.15),
    r_lim=1.0,
    grid_wave_type="all",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.0]),
)
fig, _ = mg.plot()
fig.savefig("/home/nbyrnes/code/random-matrix/rmt_analysis/grid_square.svg",format="svg")


# -----------------------------------------------------------------------------
# Random grids
# -----------------------------------------------------------------------------

# Delaunay triangles
# mg = mode_grid_factory.from_random(
#     num_points=100,
#     r_lim=1.0,
#     random_type="delaunay",
#     grid_wave_type="all",
# )
# fig = mg.plot()

# Voronoi regions
mg = mode_grid_factory.from_random(
    num_points=200,
    r_lim=1.0,
    random_type="voronoi",
    grid_wave_type="all",
)
fig, _ = mg.plot()
fig.savefig("/home/nbyrnes/code/random-matrix/rmt_analysis/grid_voronoi.svg",format="svg")

