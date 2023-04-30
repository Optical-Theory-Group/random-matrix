import numpy as np


from random_matrix.grid_generator import GridGenerator


#####################
# Mode module tests #
#####################

my_grid = GridGenerator.from_tiling(
    tiling_type="hexagons",
    side_length=0.1,
    r_lim=1.5,
    rotation_angle=0.1,
    translation_vector=np.array([0.0, 0.0]),
    grid_wave_type="all",
)
my_grid.plot()
print(my_grid)
s = 0
for mode in my_grid.modes_propagating.values():
    s += mode.weight
for mode in my_grid.modes_evanescent.values():
    s += mode.weight
print(s)
print(np.pi * 1.5**2)


my_grid = GridGenerator.from_random(
    num_points=1000, r_lim=1.5, random_type="delaunay", grid_wave_type="all"
)
my_grid.plot()
print(my_grid)
s = 0
for mode in my_grid.modes_propagating.values():
    s += mode.weight
for mode in my_grid.modes_evanescent.values():
    s += mode.weight
print(s)
print(np.pi * 1.5**2)
