import numpy as np


from random_matrix.grid_generator import GridGenerator


#####################
# Mode module tests #
#####################

""" # Cartesian test from dx, dy
grid_data = {
    "t_offset": 0.0,
    "is_polar_grid": False,
    "grid_wave_type": "propagating",
}
dx = 0.25
dy = 0.25
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)

# Cartesian test from dx, dy
# Only evanescent modes and rotated
grid_data = {
    "t_offset": 0.1,
    "is_polar_grid": False,
    "grid_wave_type": "evanescent",
}
dx = 0.25
dy = 0.25
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)

# Cartesian test from dx, dy
# Both types of modes and rotated
grid_data = {
    "t_offset": 1.1,
    "is_polar_grid": False,
    "grid_wave_type": "all",
}
dx = 0.3
dy = 0.3
my_grid = ModeGrid.from_dx_dy(dx=dx, dy=dy, grid_data=grid_data)
my_grid.plot(show_indices=True)


# Slightly shifted cartesian grid
# Not recirpcoal
grid_data = {
    "t_offset": 0.0,
    "is_polar_grid": False,
    "grid_wave_type": "propagating",
}

x_vals = np.linspace(-1.0, 1.1, 10)
y_vals = np.linspace(-1.0, 1.0, 10)
my_grid = ModeGrid.from_xy_vals(
    x_vals=x_vals, y_vals=y_vals, grid_data=grid_data
)
my_grid.plot(show_indices=True)

# Wild Cartesian grid from random x and y values
grid_data = {
    "t_offset": 0.0,
    "is_polar_grid": False,
    "grid_wave_type": "all",
}
x_vals = np.random.randn(7)
y_vals = np.random.randn(7)
my_grid = ModeGrid.from_xy_vals(
    x_vals=x_vals, y_vals=y_vals, grid_data=grid_data
)
my_grid.plot(show_indices=True)

# Polar from dr dt
grid_data = {
    "t_offset": 0.0,
    "is_polar_grid": True,
    "grid_wave_type": "propagating",
    "include_central_mode": True,
}
dr = 0.2
dt = 2 * np.pi / 6
my_grid = ModeGrid.from_dr_dt(dr=dr, dt=dt, r_lim=2.0, grid_data=grid_data)
my_grid.plot(show_indices=True)

# Polar from dr dt, not including middle and roatated
grid_data = {
    "t_offset": 1.5,
    "is_polar_grid": True,
    "grid_wave_type": "all",
    "include_central_mode": True,
}
dr = 0.25
dt = 2 * np.pi / 8
my_grid = ModeGrid.from_dr_dt(dr=dr, dt=dt, r_lim=2.0, grid_data=grid_data)
my_grid.plot(show_indices=True)

# Polar from dr dt, not including middle and roatated
grid_data = {
    "t_offset": 2.5,
    "is_polar_grid": True,
    "grid_wave_type": "all",
    "include_central_mode": False,
}
dr = 0.25
dt = 2 * np.pi / 8
my_grid = ModeGrid.from_dr_dt(dr=dr, dt=dt, r_lim=1.25, grid_data=grid_data)
my_grid.plot(show_indices=True)

# Polar from dr dt, not including middle and roatated
grid_data = {
    "t_offset": 0.0,
    "is_polar_grid": True,
    "grid_wave_type": "all",
    "include_central_mode": False,
}
r_vals = np.random.uniform(0, 3.0, 5)
t_vals = np.random.uniform(0, 2 * np.pi, 5)
my_grid = ModeGrid.from_rt_vals(
    r_vals=r_vals, t_vals=t_vals, grid_data=grid_data
)
my_grid.plot(show_indices=True)
 """


""" grid_params = {"is_polar_grid": False, "grid_wave_type": "all"}
my_grid = ModeGridGenerator.from_tiling(
    tiling_type="triangles",
    side_length=0.3,
    r_lim=2.0,
    grid_params=grid_params,
)
my_grid.plot(show_indices=False)
 """

grid_params = {
    "is_polar_grid": True,
    "grid_wave_type": "all",
}

for _ in range(10):
    my_grid = GridGenerator.from_random(num_points=500, r_lim=2.0)
# my_grid.plot(show_indices=True)

# s = 0
# for mode in my_grid.modes_propagating.values():
#    s += mode.weight
# for mode in my_grid.modes_evanescent.values():
#    s += mode.weight
#
# print(np.isclose(s, np.pi * r_lim**2))
