"""Module for testing utility functions
"""


import sys


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from random_matrix.utils.geometry_utils import (
    get_circle_coordinate,
    polar_to_cartesian,
    cartesian_to_polar,
    is_rectangle,
    rotate_points,
    get_small_angular_difference,
    order_points,
    get_convex_polygon_area,
    get_edge_area,
    get_line_segment_circle_intersection_points,
    get_polygon_circle_intersection_points,
)
from random_matrix.utils.plotting_utils import draw_line

np.random.seed(128)

print("get_circle_coordinate\n")
r = 1.2
x1 = 0.1
x2 = np.array([0.1, 0.1, 0.1])
x3 = np.array([[0.1, 0.1], [0.1, 0.1]])
print(get_circle_coordinate(x1, r))
print(get_circle_coordinate(x2, r))
print(get_circle_coordinate(x3, r))
print("------------")

print("cartesian_to_polar and polar_to_cartesian\n")
x1 = np.array([1.0, 1.0])
print(cartesian_to_polar(x1))
print(polar_to_cartesian(cartesian_to_polar(x1)))
x2 = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
print(cartesian_to_polar(x2))
print(polar_to_cartesian(cartesian_to_polar(x2)))
print("------------")

print("get_small_angular_difference\n")
t1 = 0
t2 = np.pi / 2
print(get_small_angular_difference(t1, t2) - np.pi / 2)
t3 = np.pi
print(get_small_angular_difference(t1, t3))
print(get_small_angular_difference(t1, t3 + 0.01))
print(get_small_angular_difference(-t1, -t2 + 100 * np.pi))
print("------------")

print("rotate_points")
points = np.random.randn(3, 2)
angle = np.random.randn()
rotated = rotate_points(points, angle)
print(np.linalg.norm(points, axis=1))
print(np.linalg.norm(rotated, axis=1))
print("------------")

print("is_rectangle")
points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [1.0, 5.0]])
print(is_rectangle(points))
points = rotate_points(points, np.random.randn())
print(is_rectangle(points))
points[3] += 0.000001
print(is_rectangle(points))
print("------------")

print("order_points")
points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [1.0, 5.0]])
ordered = order_points(points)
print(points)
print(ordered)
print("------------")

print("get_convex_polygon_area")
points = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
print(get_convex_polygon_area(points))
points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
print(get_convex_polygon_area(points))
points = np.array([[-0.5, 0], [0.5, 0], [0.0, np.sqrt(3) / 2]])
print(get_convex_polygon_area(points))
print("------------")

print("get_edge_area")
points = np.array(
    [[-np.sqrt(2) / 2, np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
)
print(get_edge_area(points))
print("------------")

print("get_line_segment_circle_intersection_points")
points = np.array([[0.0, 0.0], [1.0, 1.0]])
print(get_line_segment_circle_intersection_points(points))
print("------------")

print("get_polygon_circle_intersection_points")
points = np.array([[0.9, 0.9], [-0.9, 0.9], [0.9, -0.9], [-0.9, -0.9]])
print(get_polygon_circle_intersection_points(points))

dx = 0.1
dy = 0.0001
points = np.array([[-dx,0.0],[dx,0.0],[-dx,1.0-dy],[dx,1.0-dy]])
test=get_polygon_circle_intersection_points(points)
test = order_points(test)
test2 = np.array([test[0],test[-1]])

print("------------")
