import numpy as np
from random_matrix import integrator, mode_grid_generator
import quadpy

def f(x):
    return 1.0


mode_grid = mode_grid_generator.from_tiling("rectangles", 0.4)
mode = mode_grid.by_index(0)
triangle = mode.triangulation[0]
scheme = quadpy.t2.get_good_scheme(10)

integral = 0.0
for triangle in mode.triangulation:
    print(triangle)
    a = scheme.integrate(f, triangle)








#integral = integrator.integrate(f, mode)

#print(integral)