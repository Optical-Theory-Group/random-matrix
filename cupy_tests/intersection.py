import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import cupy as cp
import time
import sklearn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random_matrix.utils import array_utils, geometry_utils

# -----------------------------------------------------------------------------
# Pairing functions


def cantor_pair(xy):
    x, y = xy[:, 0], xy[:, 1]
    return ((x + y) * (x + y + 1)) // 2 + y


def inverse_cantor(z):
    xp = cp.get_array_module(z)
    w = xp.floor((xp.sqrt(8 * z + 1) - 1) / 2).astype(xp.int64)
    t = (w * (w + 1)) // 2
    y = (z - t).astype(xp.uint64)
    x = (w - y).astype(xp.uint64)
    return xp.column_stack((x, y))


def bitwise_hash(xy: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    xp = cp.get_array_module(xy)
    xy = xy.astype(xp.uint64)
    x, y = xy[:, 0], xy[:, 1]
    return (x << 32) | y  # Store x in upper 32 bits


def inverse_bitwise_hash(
    z: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    xp = cp.get_array_module(z)
    z = z.astype(xp.uint64)
    x = z >> 32  # Extract upper 32 bits
    y = z & 0xFFFFFFFF  # Extract lower 32 bits
    return xp.column_stack((x, y))


def intersect_hull_with_hyperplane_old(
    points: np.ndarray,
    simplices: np.ndarray,
    hyperplane: tuple[np.ndarray, float],
) -> np.ndarray:
    """Find the intersection of a convex hull object with a hyperplane

    The convex hull object is defined by the two variables points, an array of
    the points, and simplicies, which is a list of lists of indices telling you
    which points form each simplex. These basically should look like
    hull.points and hull.simplices.

    The hyperplane should be given as a tuple containing an array n defining
    the normal vector n and a float d such that the hyperplane is defined by
    the equation

    r * n = d

    The output is an array of points that bound the intersection. Note that
    these points may contain redundancies, i.e. may contain additional points
    on the edges of the boundary, rather than purely the vertices. This is a
    feature, not a bug.
    """
    n, d = hyperplane

    # Compute dot products and signs for all points
    values = np.dot(points, n) - d
    values[np.isclose(values, 0.0)] = 0.0  # Clean edge cases
    signs = np.sign(values)

    # Extract edges from simplices and remove duplicates
    edges = np.vstack(
        [
            np.sort(
                np.array(list(itertools.combinations(simplex, 2)), dtype=int),
                axis=1,
            )
            for simplex in simplices
        ]
    )
    edges = np.unique(edges, axis=0)

    # Get the signs of the two vertices of each edge
    signs1 = signs[edges[:, 0]]
    signs2 = signs[edges[:, 1]]
    product = signs1 * signs2

    # Identify edges intersecting the hyperplane
    crossing_edges = edges[product == -1]  # Opposite signs
    on_plane_edges = edges[product == 0]  # At least one vertex on the plane

    # Initialize list of intersections
    intersections = []

    # Add vertices lying on the plane
    for edge in on_plane_edges:
        for idx in edge:
            if signs[idx] == 0:
                intersections.append(points[idx])

    intersections = np.unique(intersections, axis=0)
    # Compute intersections for crossing edges
    v1 = points[crossing_edges[:, 0]]
    v2 = points[crossing_edges[:, 1]]
    t = (d - np.dot(v1, n)) / np.dot(v2 - v1, n)
    intersection_points = v1 + t[:, None] * (v2 - v1)
    if np.shape(intersections) == (0,):
        intersections = np.zeros((0, 8), dtype=np.float128)

    intersections = np.vstack((intersections, intersection_points))
    return intersections


def intersect_hull_with_hyperplane_new(
    points: np.ndarray | cp.ndarray,
    simplices: np.ndarray | cp.ndarray,
    hyperplane: tuple[np.ndarray | cp.ndarray, float],
) -> np.ndarray | cp.ndarray:
    xp = cp.get_array_module(points)
    n, d = hyperplane
    values = xp.dot(points, n) - d
    values[xp.isclose(values, 0.0)] = 0.0  # Handle edge cases
    signs = xp.sign(values)

    _, num_dim = points.shape
    simplices = xp.sort(simplices)
    pairs = xp.array(list(itertools.combinations(range(num_dim), 2)))
    edge_indices = simplices[:, pairs].reshape(-1, 2)
    coded = xp.unique(bitwise_hash(edge_indices))
    edges = inverse_bitwise_hash(coded)

    product = signs[edges[:, 0]] * signs[edges[:, 1]]
    crossing_edges = edges[product == -1]  # Opposite signs

    # Add vertices lying on the plane
    on_plane_point_indices = np.where(np.isclose(signs, 0.0))
    on_plane_points = points[on_plane_point_indices]

    # Compute intersections for crossing edges
    v1 = points[crossing_edges[:, 0]]
    v2 = points[crossing_edges[:, 1]]
    t = (d - xp.dot(v1, n)) / xp.dot(v2 - v1, n)
    crossing_points = v1 + t[:, None] * (v2 - v1)

    return xp.vstack((on_plane_points, crossing_points))


def get_degenerate_hull_simplices_old(
    points: np.ndarray,
) -> np.ndarray:
    """Given a set of points that fill out a lower dimensional subspace of the
    ambient space in which they reside, find a lower dimensional simplex
    decomposition.

    This works by adding in an extra points to the hull to flesh it out into
    the unspanned dimensions. The convex hull of the resulting hull is then
    found. Finally, the additional points are thrown away.

    Example: Image a 2D planar polygon in 3D space. We can imagine
    triangulating it, but scipy won't. Add a point out of the plane of the
    polygon and decompose the resulting 3D shape into simplices. The edges of
    these simplices will also triangulate the original polygon. We then throw
    away the added point and keep the 2D triangulation.
    """
    dimension = points.shape[1]
    new_point = np.random.randn(dimension)
    augmented_points = np.vstack([points, new_point])
    new_point_index = len(augmented_points) - 1

    hull = scipy.spatial.ConvexHull(augmented_points)
    # Filter out simplices containing the new point index
    mask = ~np.isin(hull.simplices, new_point_index).any(axis=1)
    filtered_simplices = hull.simplices[mask]
    return filtered_simplices


np.random.seed(2)

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
diff = np.random.randn(2)
domain = geometry_utils.iterated_cartesian_product(
    [
        square,
        square + diff,
        square,
        square + diff,
    ]
)
n1 = np.array([1, 0, -1, 0, -1, 0, 1, 0])
n2 = np.array([0, 1, 0, -1, 0, -1, 0, 1])
d = 0.0

start = time.perf_counter()
hull = geometry_utils.get_convex_hull_iterative(domain)
end = time.perf_counter()
print(f"Hull found in {end-start}")

points = hull.points
simplices = hull.simplices
hyperplane1 = (n1, d)
hyperplane2 = (n2, d)
filter_one = [0, 1, 2, 3, 4, 5, 7]
filter_two = [0, 1, 2, 3, 4, 5]


# -----------------------------------------------------------------------------
# Old vs new, numpy
start = time.perf_counter()
intersection_points_old = intersect_hull_with_hyperplane_old(
    points, simplices, hyperplane1
)
end = time.perf_counter()
print(f"Numpy old: {end-start}")

start = time.perf_counter()
intersection_points_new = intersect_hull_with_hyperplane_new(
    points, simplices, hyperplane1
)
end = time.perf_counter()
print(f"Numpy new: {end-start}")

start = time.perf_counter()
intersection_points_cupy = intersect_hull_with_hyperplane_new(
    cp.asarray(points), cp.asarray(simplices), (cp.asarray(n1), d)
)
end = time.perf_counter()
print(f"Cupy new first run: {end-start}")


start = time.perf_counter()
intersection_points_cupy = intersect_hull_with_hyperplane_new(
    cp.asarray(points), cp.asarray(simplices), (cp.asarray(n1), d)
)
end = time.perf_counter()
print(f"Cupy new: {end-start}")

# # -----------------------------------------------------------------------------
# # Compare old and new answers
print(
    array_utils.is_equal_array(
        intersection_points_old, intersection_points_new
    )
)
print(
    array_utils.is_equal_array(
        intersection_points_old, cp.asnumpy(intersection_points_cupy)
    )
)

assert False
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# FIND SIMPLICES OF THE REDUCED SHAPE
# -----------------------------------------------------------------------------


def get_degenerate_hull_simplices_old(
    points: np.ndarray,
) -> np.ndarray:
    """Given a set of points that fill out a lower dimensional subspace of the
    ambient space in which they reside, find a lower dimensional simplex
    decomposition.

    This works by adding in an extra points to the hull to flesh it out into
    the unspanned dimensions. The convex hull of the resulting hull is then
    found. Finally, the additional points are thrown away.

    Example: Image a 2D planar polygon in 3D space. We can imagine
    triangulating it, but scipy won't. Add a point out of the plane of the
    polygon and decompose the resulting 3D shape into simplices. The edges of
    these simplices will also triangulate the original polygon. We then throw
    away the added point and keep the 2D triangulation.
    """
    dimension = points.shape[1]
    new_point = np.random.randn(dimension)
    augmented_points = np.vstack([points, new_point])
    new_point_index = len(augmented_points) - 1

    hull = scipy.spatial.ConvexHull(augmented_points, qhull_options="QJ")
    # Filter out simplices containing the new point index
    mask = ~np.isin(hull.simplices, new_point_index).any(axis=1)
    filtered_simplices = hull.simplices[mask]
    return filtered_simplices


def get_degenerate_hull_simplices_new(
    points: np.ndarray,
) -> np.ndarray:
    """Given a set of points that fill out a lower dimensional subspace of the
    ambient space in which they reside, find a lower dimensional simplex
    decomposition.

    This works by adding in an extra points to the hull to flesh it out into
    the unspanned dimensions. The convex hull of the resulting hull is then
    found. Finally, the additional points are thrown away.

    Example: Image a 2D planar polygon in 3D space. We can imagine
    triangulating it, but scipy won't. Add a point out of the plane of the
    polygon and decompose the resulting 3D shape into simplices. The edges of
    these simplices will also triangulate the original polygon. We then throw
    away the added point and keep the 2D triangulation.
    """
    reduced_points = points[:, 1:]
    hull = scipy.spatial.ConvexHull(reduced_points)
    return hull.simplices


# Old method
# start = time.perf_counter()
# filtered_simplices_old = get_degenerate_hull_simplices_old(
#     intersection_points_old
# )
# end = time.perf_counter()
# print(f"Filtered simplices numpy old: {end-start}")

# -----------------------------------------------------------------------------
# Whole method
start = time.perf_counter()

intersection_points = intersect_hull_with_hyperplane_new(
    hull.points, hull.simplices, hyperplane1
)
print("Found intersection 1")
reduced_points = intersection_points[:, 1:]
reduced_hull = geometry_utils.get_convex_hull_iterative(reduced_points)
print("Found hull 1")
intersection_points_two = intersect_hull_with_hyperplane_new(
    reduced_hull.points, reduced_hull.simplices, (n2[1:], 0.0)
)
print("Found intersection 2")
reduced_points = intersection_points_two[:, 1:]
reduced_hull = geometry_utils.get_convex_hull_iterative(reduced_points)
print("Found hull 2")

end = time.perf_counter()
print(f"Time: {end - start}")
print(f"Vertices: {reduced_hull.vertices.shape}")
print(f"Simplices: {reduced_hull.simplices.shape}")
print(f"Volume: {reduced_hull.volume}")
print("---------")
# --------------------
# NO REDUNDANCY CHECKING
start = time.perf_counter()

intersection_points = intersect_hull_with_hyperplane_new(
    hull.points, hull.simplices, hyperplane1
)
reduced_points = intersection_points[:, 1:]
reduced_hull = scipy.spatial.ConvexHull(reduced_points)
intersection_points_two = intersect_hull_with_hyperplane_new(
    reduced_hull.points, reduced_hull.simplices, (n2[1:], 0.0)
)
reduced_points = intersection_points_two[:, 1:]
reduced_hull = scipy.spatial.ConvexHull(reduced_points)

end = time.perf_counter()
print(f"Time: {end - start}")
print(f"Vertices: {reduced_hull.vertices.shape}")
print(f"Simplices: {reduced_hull.simplices.shape}")
print(f"Volume: {reduced_hull.volume}")
print("---------")

# --------------------
# NO REDUNDANCY CHECKING
start = time.perf_counter()

a = time.perf_counter()
intersection_points = intersect_hull_with_hyperplane_new(
    hull.points, hull.simplices, hyperplane1
)
b = time.perf_counter()
print(f"First intersection: {b-a}")

a = time.perf_counter()
reduced_points = intersection_points[:, filter_one]
reduced_hull = scipy.spatial.ConvexHull(reduced_points,qhull_options="Qbb")
b = time.perf_counter()
print(f"First hull: {b-a}")

a = time.perf_counter()
intersection_points_two = intersect_hull_with_hyperplane_new(
    reduced_hull.points, reduced_hull.simplices, (n2[filter_one], 0.0)
)
b = time.perf_counter()
print(f"Second intersection: {b-a}")

a = time.perf_counter()
reduced_points = intersection_points_two[:, filter_two]
reduced_hull = geometry_utils.get_convex_hull_iterative(reduced_points)
b = time.perf_counter()
print(f"Second hull: {b-a}")

end = time.perf_counter()
print(f"Time: {end - start}")
print(f"Vertices: {reduced_hull.vertices.shape}")
print(f"Simplices: {reduced_hull.simplices.shape}")
print(f"Volume: {reduced_hull.volume}")
print("---------")


start = time.perf_counter()
D = scipy.spatial.Delaunay(reduced_points)
C = geometry_utils.get_convex_hull_iterative(reduced_points)
end = time.perf_counter()
print(f"Time: {end - start}")


# CHeck volume of first simplex
v = 0
vertices = C.points[C.vertices]


centroid = np.mean(vertices, axis=0)
for simplex_indices in C.simplices:
    simplex = C.points[simplex_indices]
    simplex = np.vstack((centroid, C.points[simplex_indices]))
    v += scipy.spatial.ConvexHull(simplex,qhull_options="QJ").volume
print(f"Computed Volume: {v}")
print(f"Hull Volume: {C.volume}")
print(f"Relative Error: {abs(v - C.volume) / C.volume:.2e}")
