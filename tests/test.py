import matplotlib.pyplot as plt
import numpy as np
from random_matrix.utils import array_utils, geometry_utils
import skspatial.objects
from shapely.geometry import Polygon, Point, MultiPolygon
import scipy.spatial

# create a rectangle
x0 = 0.5
y0 = 0.9
rectangle_points = [(x0,0),(x0,y0),(-x0,y0),(-x0,0)]
rectangle = Polygon(rectangle_points)

circle = Point((0.0,0.0)).buffer(1.0,resolution=1000)

innies = rectangle.intersection(circle)
outies = rectangle.difference(circle)

innie_points = np.array(innies.exterior.xy).T

outies_points = []
for outie in outies.geoms:
    outies_points.append(np.array(outie.exterior.xy).T)

outie_1_points = outies_points[0]
outie_2_points = outies_points[1]

out_1_r = np.linalg.norm(outie_1_points, axis=1)
out_2_r = np.linalg.norm(outie_2_points, axis=1)
in_r = np.linalg.norm(innie_points, axis=1)

fig, ax = plt.subplots()
for point in innie_points:
    ax.scatter(point[0], point[1])
    
t = np.linspace(-1,1,1000)
y = np.sqrt(1-t**2)
ax.plot(t,y)