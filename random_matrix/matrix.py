import numpy as np
from random_matrix.utils import geometry_utils, array_utils
import matplotlib.pyplot as plt
import shapely

np.random.seed(7)

a = np.abs(np.random.randn(3, 2))
b = np.abs(np.random.randn(3, 2)) + np.array([3.0, 0.0])
c = geometry_utils.minkowski_sum(a, b) / 2

xs = c[:, 0]
ys = c[:, 1]
x_min = np.min(xs)
x_max = np.max(xs)
y_min = np.min(ys)
y_max = np.max(ys)

n_x = 10**3
n_y = 10**3
xs = np.linspace(x_min, x_max, n_x)
ys = np.linspace(y_max, y_min, n_y)

XS, YS = np.meshgrid(xs, ys)
areas = np.zeros((n_x, n_y))


for i in range(n_x):
    for j in range(n_y):
        x = XS[i, j]
        y = YS[i, j]

        reflected = 2*np.array([x,y]) - b
        
        poly_one = shapely.Polygon(a)
        poly_two = shapely.Polygon(reflected)
        intersection = poly_one.intersection(poly_two)
        area = intersection.area
        areas[i,j] = area

areas = areas/np.max(areas)
fig, ax = plt.subplots()
n_contours = 20
ax.contourf(XS, YS, areas, [0.001] + [i*1/n_contours for i in range(1,n_contours+1)])

indices = np.where(np.isclose(areas, 1.0))
coordinates = list(zip(indices[0], indices[1]))

# fig, axes = plt.subplots()
# axes.set_aspect("equal")

# for one, two in array_utils.get_pairs(a, cyclic=True):
#     axes.plot((one[0], two[0]), (one[1], two[1]), color="black")

# for one, two in array_utils.get_pairs(b, cyclic=True):
#     axes.plot((one[0], two[0]), (one[1], two[1]), color="blue")

# for one, two in array_utils.get_pairs(c, cyclic=True):
#     axes.plot((one[0], two[0]), (one[1], two[1]), color="green")


