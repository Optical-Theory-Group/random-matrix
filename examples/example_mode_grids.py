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
mode_grid_factory.from_dr_dt(
    dr=0.1,
    dt=2 * np.pi / 12,
    r_lim=1.0,
    include_central_mode=False,
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
