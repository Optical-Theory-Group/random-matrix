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
# my_grid = mode_grid_factory.from_tiling("rectangles", [0.1, 0.1])
# theta = np.linspace(np.pi/4, 5*np.pi/4, 10_000)

# rs = [0.1, np.sqrt(2)*0.1, 0.2, np.linalg.norm(my_grid.by_index(138).center)]
# y1s = [np.sqrt(r**2-(r*t1)**2) for r in rs]
# y2s = [-r*np.sqrt(1.0-t2**2) for r in rs]


# fig, ax = my_grid.plot(show_indices=False, figsize=(12, 12))
# lw=0.9
# for r in rs:
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     ax.plot(x, y, color="black", lw=lw)    # ax.plot(t2, y2, color="black",lw=lw)

# marker="x"
# color="red"
# ax.scatter(*my_grid.by_index(0).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(158).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(159).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(137).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(138).center,color=color,marker=marker)

# ax.scatter(*my_grid.by_index(-156).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(-157).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(136).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(157).center,color=color,marker=marker)

# ax.scatter(*my_grid.by_index(-177).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(-178).center,color=color,marker=marker)
# ax.scatter(*my_grid.by_index(160).center,color=color,marker=marker)
# fig.savefig("/home/nbyrnes/code/random-matrix/examples/square_grid.svg", format="svg")


my_grid = mode_grid_factory.from_tiling("hexagons", 0.07)
theta = np.linspace(np.pi/3, 2 * np.pi/6 * 3, 10_000)

rs = [
    np.linalg.norm(my_grid.by_index(118).center),
    np.linalg.norm(my_grid.by_index(120).center),
    np.linalg.norm(my_grid.by_index(117).center),
    np.linalg.norm(my_grid.by_index(84).center),

]

fig, ax = my_grid.plot(show_indices=False, figsize=(12, 12))
lw = 0.9
for r in rs:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color="black", lw=lw)  # ax.plot(t2, y2, color="black",lw=lw)

marker="x"
color="red"
ax.scatter(*my_grid.by_index(0).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(118).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(-141).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(117).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(120).center,color=color,marker=marker)

ax.scatter(*my_grid.by_index(-140).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(84).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(119).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(122).center,color=color,marker=marker)

ax.scatter(*my_grid.by_index(-112).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(115).center,color=color,marker=marker)
ax.scatter(*my_grid.by_index(-114).center,color=color,marker=marker)
fig.savefig("/home/nbyrnes/code/random-matrix/examples/hexagon_grid.svg", format="svg")


# my_grid = mode_grid_factory.from_tiling("rectangles", [0.1, 0.1])
# t = np.linspace(-1, 0, 10**4)
# r1 = 0.1
# r2 = np.sqrt(2)*0.1
# r3 = 0.2
# r4 = np.linalg.norm(my_grid.by_index(138).center)


# y1 = np.sqrt(r1**2 - t**2)
# y2 = np.sqrt(r2**2-t**2)
# y3 = np.sqrt(r3**2-t**2)
# y4 = np.sqrt(r4**2-t**2)


# fig, ax = my_grid.plot(show_indices=False, figsize=(12, 12))
# lw=0.9
# ax.plot(t, y1, color="black",lw=lw)
# ax.plot(t, y2, color="black",lw=lw)
# ax.plot(t, y3, color="black",lw=lw)
# ax.plot(t, y4, color="black",lw=lw)
# marker="x"
# ax.scatter(*my_grid.by_index(0).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(158).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(159).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(137).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(138).center,color="black",marker=marker)

# ax.scatter(*my_grid.by_index(-177).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(-178).center,color="black",marker=marker)
# ax.scatter(*my_grid.by_index(160).center,color="black",marker=marker)
# fig.savefig("/home/nbyrnes/code/random-matrix/examples/square_grid.svg", format="svg")

# Standard polar grid rotated
# g = mode_grid_factory.from_dr_dt(
#     dr=0.1,
#     dt=2 * np.pi / 20,
#     r_lim=1.0,
#     rotation_angle=0.0,
#     include_central_mode=True,
# )
# g.plot(show_indices=True, close=False)

assert False

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
