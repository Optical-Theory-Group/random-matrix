import sys

sys.path.append("..")  # add parent directory to sys.path

import matplotlib.pyplot as plt
import numpy as np
from random_matrix.utils.geometry_utils import is_rectangle, rotate_points
from random_matrix.utils.plotting_utils import draw_line
from scipy.spatial import ConvexHull

np.random.seed(128)

######
# Not a rectangle
#
points = np.random.randn(4, 2)
print(is_rectangle((points)))

#####
# Obvious rectangle
#
points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [1.0, 5.0]])
print(is_rectangle(points))

#####
# Rotated rectangle
#
rotated_points = rotate_points(points, np.random.randn(2), np.random.randn())
print(is_rectangle(rotated_points))

####
# Get cyclic points
#
points = np.random.randn(5, 2)
