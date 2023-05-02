import numpy as np

from random_matrix import mode_grid_generator


# -----------------------------------------------------------------------------
# Tilings
# -----------------------------------------------------------------------------


def check_weights(grid, my_list):
    s = 0
    for mode in grid.modes.values():
        s += mode.weight
    truth = grid.r_lim**2 * np.pi
    if np.isclose(s, truth):
        print("Correct")
    else:
        print("Uh oh")
        print(s)
        print(truth)
        my_list.append(grid)
        grid.plot()
        
dodgy_grids = []

for _ in range(1):
    # my_grid = mode_grid_generator.from_tiling(
    #     tiling_type="hexagons",
    #     side_length=0.25,
    #     r_lim=2.0,
    #     grid_wave_type="all",
    #     rotation_angle=2.0,
    #     translation_vector=np.array([0.0, 0.00]),
    # )
    # # my_grid.plot(show_indices=True)
    # check_weights(my_grid)

    # my_grid = mode_grid_generator.from_tiling(
    #     tiling_type="triangles",
    #     side_length=0.35,
    #     r_lim=2.0,
    #     grid_wave_type="all",
    #     rotation_angle=0.0,
    #     translation_vector=np.array([0.0, 0.0]),
    # )
    # # my_grid.plot(show_indices=True)
    # check_weights(my_grid)

    # my_grid = mode_grid_generator.from_tiling(
    #     tiling_type="rectangles",
    #     side_length=0.25,
    #     r_lim=2.0,
    #     grid_wave_type="all",
    #     rotation_angle=0.0,
    #     translation_vector=np.array([0.0, 0.0]),
    # )
    # # my_grid.plot(show_indices=True)
    # check_weights(my_grid)

    # -----------------------------------------------------------------------------
    # Random grids
    # -----------------------------------------------------------------------------

    # my_grid = mode_grid_generator.from_random(
    #     num_points=500,
    #     r_lim=2.0,
    #     random_type="delaunay",
    #     grid_wave_type="all",
    # )
    # # my_grid.plot()
    # check_weights(my_grid, dodgy_grids)

    my_grid = mode_grid_generator.from_random(
        num_points=500,
        r_lim=2.0,
        random_type="voronoi",
        grid_wave_type="all",
    )
    # my_grid.plot()
    check_weights(my_grid, dodgy_grids)
