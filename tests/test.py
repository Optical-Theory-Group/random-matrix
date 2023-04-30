import matplotlib.pyplot as plt
import numpy as np
from random_matrix.utils import array_utils, geometry_utils, plotting_utils
import skspatial.objects
from shapely.geometry import Polygon, Point, MultiPolygon
import scipy.spatial

# create a rectangle
x0 = 0.5
y0 = geometry_utils.get_circle_coordinate(x0)
rectangle_points = [(x0, 0), (x0, y0), (-x0, y0), (-x0, 0)]
rectangle = np.array(rectangle_points)

triangle = np.array([[-1, -0.9], [1, -0.9], [-1, -1.1],[1,-1.1]])

intersection_points = geometry_utils.get_polygon_circle_intersection_points(
    triangle
)

augmented_points = np.vstack((triangle, intersection_points))
augmented_points = array_utils.remove_duplicate_points(augmented_points)


fig, ax = plt.subplots()
ax.set_aspect("equal")
plotting_utils.draw_convex_polygon(ax, triangle)
plotting_utils.draw_circle(ax)
for pt in augmented_points:
    ax.scatter(pt[0], pt[1], color="red")

points = np.random.randn(100,2)
p1 = geometry_utils.order_points(points)
p2 = geometry_utils.order_points_2(p1)
print(p1)
print(p2)