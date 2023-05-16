import numpy as np

from random_matrix.modes import mode_grid, mode_grid_generator
from random_matrix.statistics import scattering_statistics

my_grid = mode_grid_generator.from_tiling(
    tiling_type="rectangles",
    side_length=(0.7, 0.7),
    r_lim=2.0,
    grid_wave_type="propagating",
    rotation_angle=0.0,
    translation_vector=np.array([0.0, 0.00]),
)
my_grid.plot(show_indices=True)
my_stats = scattering_statistics.InputStatisticsManager(None, None, my_grid)
print(len(my_stats.independent_element_indices["pp"]["t"]))

