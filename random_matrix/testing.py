from modes import Mode
import numpy as np
from utils import polar_to_cartesian, cartesian_to_polar

np.random.seed(0)

#####################
# Mode module tests #
#####################

###############
# Polar mode

# Central mode
my_points = np.array([[0.0,0.0],[0.0,0.2]])
my_mode = Mode(index=0, mode_boundary=my_points, mode_shape_type="polar")
my_mode.plot()

# Non-central mode
my_points = np.array([[0.2,0.0],[0.0,0.2],[0.3,0.0],[0.0,0.3]])
my_mode = Mode(mode_boundary=my_points, mode_shape_type="polar")
my_mode.plot()

r1, r2 = np.random.random(2)
t1, t2 = np.random.random(2)*2*np.pi
p1 = (r1, t1)
p2 = (r1, t2)
p3 = (r2, t1)
p4 = (r2, t2)
points = np.array([p1, p2, p3, p4])
points_cartesian = polar_to_cartesian(points)
points_polar = points
my_mode = Mode(mode_boundary=points_cartesian, mode_shape_type="polar")
my_mode.plot()