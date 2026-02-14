"""Utility functions that perform various geometric calculations."""

from itertools import combinations
import itertools
import numpy as np
import numpy.typing as npt
import scipy.spatial
import scipy.stats
import shapely
import skspatial.objects
import cupy as cp
import cdd
import cdd.gmp
from random_matrix.utils import array_utils
from random_matrix.utils.types import Numeric
from fractions import Fraction


def get_unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def get_circle_coordinate(
    x: npt.NDArray[np.float64] | float, r: float = 1.0
) -> npt.NDArray[np.float64] | float:
    """Returns the y-coordinate(s) of a point on a circle, given the
    x-coordinate(s) and radius.

    Parameters
    ----------
        x : float or numpy.ndarray
            The x-coordinate(s) of the point(s) on the circle.
        r : float
            The radius of the circle.

    Returns
    -------
        y : float or numpy.ndarray
            The y-coordinate(s) of the point(s) on the circle, corresponding
            to the input x-coordinate(s).
    """

    output = np.sqrt(r * r - x * x)
    return output


def cartesian_to_polar(
    points_cartesian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an array of 2D points from Cartesian coordinates to polar
    coordinates.

    Parameters:
    -----------
        points_cartesian : numpy.ndarray
            An array of shape (N, 2) representing the 2D points in Cartesian
            coordinates.

    Returns:
    --------
        points_polar : numpy.ndarray
            An array of shape (N, 2) representing the 2D points in polar
            coordinates. The first column contains the radius and the second
            column contains the angle in radians.
    """

    if points_cartesian.ndim != 2 or points_cartesian.shape[1] != 2:
        # Reshape the input array to the correct shape (N, 2)
        points_cartesian = np.asarray(points_cartesian)
        if points_cartesian.ndim == 1:
            points_cartesian = points_cartesian.reshape((1, 2))
        else:
            raise ValueError("Input array must have shape (N, 2)")

    x = points_cartesian[:, 0]
    y = points_cartesian[:, 1]
    r = np.linalg.norm(points_cartesian, axis=1)
    t = np.mod(np.arctan2(y, x), 2 * np.pi)
    points_polar = np.column_stack((r, t))

    # Check if only one point is present
    nx, ny = np.shape(points_polar)
    if nx == 1:
        lone_point: npt.NDArray[np.float64] = points_polar[0]
        return lone_point

    return points_polar


def polar_to_cartesian(
    points_polar: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an array of 2D points from polar coordinates to Cartesian
    coordinates.

    Parameters:
    -----------
        points_polar : numpy.ndarray
            An array of shape (N, 2) representing the 2D points in polar
            coordinates. The first column should contain the radius and the
            second column should contain the angle in radians.

    Returns:
    --------
        points_cartesian : numpy.ndarray
            An array of shape (N, 2) representing the 2D points in Cartesian
            coordinates. The first column contains the x-coordinate and the
            second column contains the y-coordinate.
    """

    if points_polar.ndim != 2 or points_polar.shape[1] != 2:
        # Reshape the input array to the correct shape (N, 2)
        points_polar = np.asarray(points_polar)
        if points_polar.ndim == 1:
            points_polar = points_polar.reshape((1, 2))
        else:
            raise ValueError("Input array must have shape (N, 2)")

    r = points_polar[:, 0]
    t = points_polar[:, 1]
    x = r * np.cos(t)
    y = r * np.sin(t)
    points_cartesian = np.column_stack((x, y))

    # Check if only one point is present
    nx, ny = np.shape(points_cartesian)
    if nx == 1:
        lone_point: npt.NDArray[np.float64] = points_cartesian[0]
        return lone_point

    return points_cartesian


def get_small_angular_difference(t_1: float, t_2: float) -> float:
    """Compute the smallest (less than PI) angular difference between
    two angles in radians, accounting for periodicity (i.e., angles are
    treated modulo 2*pi).

    Parameters:
    -----------
        t_1, t_2 : float, float
            The two angles in radians.

    Returns:
    --------
        dt : float
            The smallest angular difference between the two angles in radians.
    """

    t_1 = np.mod(t_1, 2 * np.pi)
    t_2 = np.mod(t_2, 2 * np.pi)
    dt: float = np.abs(t_2 - t_1)
    if dt > np.pi:
        dt = 2 * np.pi - dt
    return dt


def get_signed_small_angular_difference(
    first_vector: npt.NDArray, second_vector: npt.NDArray
) -> np.float128:
    t_1 = np.arctan2(first_vector[1], first_vector[0])
    t_2 = np.arctan2(second_vector[1], second_vector[0])
    dt = get_small_angular_difference(t_1, t_2)
    sign = np.sign(np.cross(first_vector, second_vector))
    return sign * dt


def is_rectangle(points: npt.NDArray[np.float64]) -> bool:
    """Determines if a set of 4 2D points form a rectangle.

    Parameters
    ----------
        points : numpy.ndarray
            A 2D numpy array of shape (4, 2) representing the 4 points.

    Returns
    -------
        bool
            True if the points form a rectangle, False otherwise.

    """

    # Must be four points
    if len(points) != 4:
        return False

    # Find all 6 possible lengths between different pairs of points
    pairs = np.array([pair for pair in combinations(points, 2)])
    side_lengths = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1)
    unique_side_lengths = array_utils.remove_duplicate_points(side_lengths)

    # There should be 2 or 3 different values for the legnths
    # 2 in the case of a square (side length + diagonal)
    # 3 in the case of a rectangle (2 side lenghts + diagonal)
    if len(unique_side_lengths) not in [2, 3]:
        return False

    sorted_lengths = np.sort(unique_side_lengths)
    if len(sorted_lengths) == 2:
        sorted_lengths = np.insert(sorted_lengths, 0, sorted_lengths[0])
    a, b, c = sorted_lengths

    # Check that the two non-diagonal sides meet at right angles
    is_close = bool(np.isclose(a**2 + b**2, c**2))

    return is_close


def rotate_points(
    points: npt.NDArray[np.float64],
    rotation_angle: float,
    axis: npt.NDArray[np.float64] = np.array([0.0, 0.0]),
) -> npt.NDArray[np.float64]:
    """Rotate a set of 2D points around a specified axis by a given angle.

    Parameters
    ----------
        points : numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the coordinates of
            n points to be rotated.
        axis : numpy.ndarray
            A 1D numpy array of length 2 representing the center of rotation.
        rotation_angle : float
            The angle (in radians) by which to rotate the points.

    Returns:
        numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the rotated points.
    """

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    translated_points = points - axis
    rotated_points = rotation_matrix @ translated_points.T
    output: npt.NDArray[np.float64] | npt.NDArray[np.float64] = (
        rotated_points.T + axis
    )
    return output


def translate_points(
    points: npt.NDArray[np.float64],
    translation_vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Translate a set of 2D points by a given vector.

    Parameters
    ----------
        points : numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the coordinates of
            n points to be translated.
        translation_vector : numpy.ndarray
            A 1D numpy array of length 2 representing the translation vector.

    Returns:
        numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the translated
            points.
    """
    if (np.ndim(points) == 1 and len(points) != len(translation_vector)) or (
        np.ndim(points) == 2 and np.shape(points)[1] != len(translation_vector)
    ):
        raise ValueError(
            "Dimension of point and translation vector do not " "match"
        )

    translated_points = points + translation_vector
    return translated_points


def order_points(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x = points[:, 0]
    y = points[:, 1]

    x0 = np.mean(x)
    y0 = np.mean(y)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    angles = np.where(
        (y - y0) > 0,
        np.arccos((x - x0) / r),
        2 * np.pi - np.arccos((x - x0) / r),
    )

    mask = np.argsort(angles)
    x_sorted = x[mask]
    y_sorted = y[mask]

    return np.array([x_sorted, y_sorted]).T


def get_convex_polygon_area(
    convex_hull: scipy.spatial.ConvexHull | npt.NDArray[np.float64],
) -> float:
    """Compute the area of a 2D convex polygon. If a non-convex polygon is
    given, the area of its convex hull will be computed instead.

    Parameters:
    ----------
        convex_hull : numpy.ndarray or scipy.spatial.ConvexHull
            Either a 2D NumPy array of shape (N, 2), representing the
            coordinates of the vertices of the convex hull in counterclockwise
            order, or a ConvexHull object representing the 2D convex hull.

    Returns:
    -------
        area : float
            The area of the convex polygon.
    """

    if isinstance(convex_hull, np.ndarray):
        convex_hull = scipy.spatial.ConvexHull(convex_hull)
    # Note that 'volume' is in fact area for a convex hull of 2D points
    # convex_hull.area instead returns the perimeter
    area: float = convex_hull.volume
    return area


def get_edge_area(
    points: npt.NDArray[np.float64], radius: np.float64
) -> np.float64:
    """Compute the area of a small circle segment bounded by a chord connecting
    two points lying on the circle and the arc in between.

    Parameters
    ----------
        points : numpy.ndarray
            The two points lying on the circle boundary.

    Returns
    -------
        area: float
            The area of the described region.

    """

    points_polar = cartesian_to_polar(points)
    t_1, t_2 = points_polar[:, 1]
    dt = get_small_angular_difference(t_1, t_2)
    area: np.float64 = radius**2 / 2 * (dt - np.sin(dt))
    return area


def get_line_segment_circle_intersection_points(
    line_segment: npt.NDArray[np.float64],
    circle: skspatial.objects.Circle = skspatial.objects.Circle(
        [0.0, 0.0], 1.0
    ),
) -> npt.NDArray[np.float64] | None:
    """Computes the intersection points between a line segment and a circle.

    Parameters
    ----------
        line_segment : numpy.ndarray
            A NumPy array of shape (2,2) representing the two endpoints of the
            line segment.
        circle : skspatial.objects.Circle
            An instance of the Circle class representing the circle.
            Default is a unit circle centered at the origin.

    Retruns
    ----------
        numpy.ndarray:
            A NumPy array of shape (N,2) representing the N intersection points
            between the line segment and the circle. If no intersection points
            are found, returns None.
    """

    if np.shape(line_segment) != (2, 2):
        raise ValueError(
            "line_segment should be given as a (2,2) array of " "points"
        )

    intersection_points = np.empty((0, 2), dtype=np.float32)
    first_point = line_segment[0]
    second_point = line_segment[1]
    line = skspatial.objects.Line.from_points(first_point, second_point)
    line_segment_obj = skspatial.objects.LineSegment(first_point, second_point)

    # Note that scikit spatial raises an error if there is no intersection
    # thus the try block
    try:
        new_intersection_points = circle.intersect_line(line)
        # Check if intersection points lie in the line segment or not
        # If yes, add it to the intersection points
        for point in new_intersection_points:
            # Handle weird cases where edge points don't trigger contains_point
            is_on_line_seg = line_segment_obj.contains_point(point)
            if is_on_line_seg:
                intersection_points = np.append(
                    intersection_points, point.reshape(1, 2), axis=0
                )

    # No intersections, even for the infinite line
    except ValueError:
        return None

    # If intersection_points has length 0, there were intersections, but they
    # didn't lie on the line segment.
    if len(intersection_points) == 0:
        return None
    else:
        return intersection_points  # type: ignore


def get_polygon_circle_intersection_points(
    points: npt.NDArray[np.float64],
    circle: skspatial.objects.Circle = skspatial.objects.Circle(
        [0.0, 0.0], 1.0
    ),
) -> npt.NDArray[np.float64] | None:
    """Returns a matrix of intersection points between a polygon defined by its
    boundary points and a circle.

    Parameters
    ----------
        points : npt.NDArray[np.float64]
            A 2D matrix of shape (N, 2) representing the vertices of the
            polygon.
        circle : skspatial.objects.Circle, optional
            A Circle object representing the circle.
            Defaults to Circle([0.0, 0.0], 1.0).

    Retruns
    ----------
        npt.NDArray[np.float64] | None
            A 2D matrix of shape (M, 2) representing M intersection points
            between the polygon and the circle, or None if there are
            no intersection points.
    """

    # Order points to avoid possible bugs
    points = order_points(points)

    intersection_points = np.empty((0, 2), dtype=np.float64)

    # First, check if vertices lie on the circle. If so add them.
    # The intersection method doesn't always capture these.
    r_vals = np.linalg.norm(points, axis=1)
    circle_points = points[np.isclose(r_vals, circle.radius)]
    intersection_points = np.vstack((intersection_points, circle_points))

    # Find intersections between line segments and the circle
    pairs = array_utils.get_pairs(points, cyclic=True)
    for line_segment in pairs:
        new_intersection_points = get_line_segment_circle_intersection_points(
            line_segment, circle
        )
        # Add new points if there are any
        if new_intersection_points is not None:
            intersection_points = np.vstack(
                (intersection_points, new_intersection_points)
            )

    if len(intersection_points) == 0:
        return None
    else:
        # Remove possible duplicates
        intersection_points = array_utils.remove_duplicate_points(
            intersection_points
        )
        return intersection_points


def get_angularly_separated_edge_points(
    circle_points: npt.NDArray[np.float64], radius: float = 1.0
) -> npt.NDArray[np.float64]:
    if len(circle_points) < 3:
        raise ValueError(
            "Must be at least two circle points for "
            "get_angularly_separated_edge_points to work."
        )

    original_points = np.copy(circle_points)
    # Get points that are in the upper semi-circle
    positive_points_indices = circle_points[:, 1] >= 0.0
    positive_points = circle_points[positive_points_indices]
    if len(positive_points) != 0:
        # There is at least one positive point
        # We will try to rotate the points so that they are all negative
        min_x_val = np.min(positive_points[:, 0])
        max_x_val = np.max(positive_points[:, 0])
        left_positive_point = np.array(
            [min_x_val, get_circle_coordinate(min_x_val, r=radius)]
        )
        right_positive_point = np.array(
            [max_x_val, get_circle_coordinate(max_x_val, r=radius)]
        )
        theta_left = cartesian_to_polar(left_positive_point)[1]
        theta_right = cartesian_to_polar(right_positive_point)[1]
        # Try rotating anti-clockwise by PI - theta_right
        rotated_points = rotate_points(circle_points, np.pi - theta_right)
        # Are these now all negative?
        max_y_val_rotated = np.max(rotated_points[:, 1])
        all_negative = np.isclose(max_y_val_rotated, 0)
        if all_negative:
            circle_points = rotated_points
        else:
            # In this case we must instead rotate points clockwise by
            # theta_left
            rotated_points = rotate_points(circle_points, -theta_left)
            circle_points = rotated_points
    # By this point, all of circle points have negative y value
    # We now need the indices of the points in circle_points that have extreme
    # x values
    min_x_val = np.min(circle_points[:, 0])
    max_x_val = np.max(circle_points[:, 0])
    min_index = array_utils.get_array_index(min_x_val, circle_points)
    max_index = array_utils.get_array_index(max_x_val, circle_points)
    output = np.array([original_points[min_index], original_points[max_index]])
    return output


def minkowski_sum(points_one: Numeric, points_two: Numeric) -> Numeric:
    points_one = order_points(points_one)
    y_coordinates = points_one[:, 1]
    x_coordinates = points_one[:, 0]
    min_y_index = np.argmin(y_coordinates)
    min_y_values = np.where(
        np.isclose(y_coordinates, y_coordinates[min_y_index])
    )[0]
    min_x_index = min_y_values[np.argmin(x_coordinates[min_y_values])]
    points_one = np.roll(points_one, -2 * min_x_index)
    points_one = np.append(points_one, [points_one[0]], axis=0)

    points_two = order_points(points_two)
    y_coordinates = points_two[:, 1]
    x_coordinates = points_two[:, 0]
    min_y_index = np.argmin(y_coordinates)
    min_y_values = np.where(
        np.isclose(y_coordinates, y_coordinates[min_y_index])
    )[0]
    min_x_index = min_y_values[np.argmin(x_coordinates[min_y_values])]
    points_two = np.roll(points_two, -2 * min_x_index)
    points_two = np.append(points_two, [points_two[0]], axis=0)

    num_points_one = len(points_one)
    num_points_two = len(points_two)
    i = 0
    j = 0

    new_points = []

    while i < num_points_one and j < num_points_two:
        first_point = points_one[i % num_points_one]
        second_point = points_two[j % num_points_two]

        next_first = points_one[(i + 1) % num_points_one]
        next_second = points_two[(j + 1) % num_points_two]

        new_points.append(first_point + second_point)

        theta_one = np.arctan2(
            next_first[1] - first_point[1], next_first[0] - first_point[0]
        )

        theta_one_close_to_zero = np.isclose(theta_one, 0.0)
        theta_one = (
            2 * np.pi + theta_one
            if theta_one < 0 and not theta_one_close_to_zero
            else theta_one
        )

        theta_two = np.arctan2(
            next_second[1] - second_point[1],
            next_second[0] - second_point[0],
        )
        theta_two_close_to_zero = np.isclose(theta_two, 0.0)

        theta_two = (
            2 * np.pi + theta_two
            if theta_two < 0 and not theta_two_close_to_zero
            else theta_two
        )

        if np.isclose(theta_one, theta_two):
            i += 1
            j += 1
        elif theta_one > theta_two:
            j += 1
        else:
            i += 1

    output = np.array(new_points)
    if np.all(np.isclose(output[0], output[-1])):
        output = output[:-1]

    return output


def minkowski_difference(points_one: Numeric, points_two: Numeric) -> Numeric:
    return minkowski_sum(points_one, -points_two)


def intersects(points_one: Numeric, points_two: Numeric) -> bool:
    """Returns a bool that is true if the two polygons intersect"""

    polygon_one = shapely.Polygon(points_one)
    polygon_two = shapely.Polygon(points_two)
    return not np.isclose(polygon_one.intersection(polygon_two).area, 0.0)


def create_geometry(points):
    if np.ndim(points) == 1:
        points = points[np.newaxis, :]

    match len(points):
        case 1:
            polygon = shapely.Point(points)
        case 2:
            polygon = shapely.LineString(points)
        case _:
            polygon = shapely.Polygon(points)
    return polygon


def intersection(points_one, points_two):
    # Shapes are identical. In this case, there's no point intersecting
    # anything
    if len(points_one) == len(points_two) and np.allclose(
        points_one, points_two
    ):
        return points_one

    # Form a shapely shape depending on the number of points present
    polygon_one = create_geometry(points_one)
    polygon_two = create_geometry(points_two)

    # Search for intersection between two shapes
    # The search boolean tracks whether or not we needed to use incremental
    # buffering. Note that we expect only a single point here.
    try:
        is_intersection = polygon_one.intersects(polygon_two)
    except shapely.GEOSException:
        is_intersection = polygon_one.buffer(1e-6).intersects(polygon_two)

    if is_intersection:
        search = False
        intersection = polygon_one.intersection(polygon_two)
    else:
        search = True
        # There was no intersection. But there might be a numerical error
        # we try incremental buffering

        found = False
        buffer_start = 1e-10
        num_buffer_steps = 5
        buffers = [buffer_start * 10**i for i in range(num_buffer_steps)]

        for buffer in buffers:
            if not polygon_one.buffer(buffer).intersects(polygon_two):
                continue

            # Getting here means an intersection was found
            found = True
            intersection = polygon_one.buffer(buffer).intersection(polygon_two)

            # If resultant object contains multiple shapes, just take one.
            # Because of how it was found, these shapes must be very similar
            if hasattr(intersection, "geoms"):
                intersection = intersection.geoms[0]

            if hasattr(intersection, "exterior"):
                intersection = np.mean(intersection.exterior.coords, axis=0)
            else:
                intersection = np.mean(intersection.coords, axis=0)

            intersection = shapely.Point(intersection)
            break

    # If nothing was found above
    if search and not found:
        return np.array([])

    # If we get here, we definitely have something
    if hasattr(intersection, "geoms"):
        intersection = intersection.geoms[0]

    if hasattr(intersection, "exterior"):
        return np.array(intersection.exterior.coords[:-1])
    else:
        return np.array(intersection.coords)


def cartesian_product(polygon1: Numeric, polygon2: Numeric) -> Numeric:
    # Get the number of vertices in each polygon
    n1 = np.shape(polygon1)[0]
    n2 = np.shape(polygon2)[0]

    # Repeat each vertex of polygon1 n2 times
    repeated_polygon1 = np.repeat(polygon1, n2, axis=0)

    # Tile polygon2 to match the repeated polygon1
    tiled_polygon2 = np.tile(polygon2, (n1, 1))

    # Concatenate the repeated polygon1 and tiled polygon2
    cartesian_product = np.concatenate(
        (repeated_polygon1, tiled_polygon2), axis=1
    )

    return cartesian_product


def iterated_cartesian_product(shapes: list[np.ndarray]) -> np.ndarray:
    """Given a list of shapes, A1, A2, A3, ..., find A1xA2xA3x..."""
    output = shapes[0]
    for shape in shapes[1:]:
        output = cartesian_product(output, shape)
    return output


def reflect_through_point(shape: Numeric, point: Numeric) -> Numeric:
    reflected = point - (shape - point)
    return reflected


def get_angle_plane(v1, v2, n):
    """ """

    # Special cases
    cosine = np.dot(v1, v2)

    if cosine >= 1.0 or np.isclose(cosine, 1.0):
        return 0.0
    elif cosine <= -1.0 or np.isclose(cosine, -1.0):
        return np.pi

    theta = np.arccos(cosine)

    # check if n is in the same direction as v1xv2
    k = np.cross(v1, v2)
    k = k / np.linalg.norm(k)

    alignment = np.dot(k, n)
    if np.isclose(alignment, 1.0):
        alpha = theta
    else:
        alpha = -theta

    return alpha


def get_polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    return shapely.Polygon(polygon).area


def invert_shape(shape):
    centroid = np.mean(shape, axis=0)
    reflected_shape = reflect_through_point(shape, centroid)
    return reflected_shape


def get_circular_angle(point):
    norm = np.linalg.norm(point)
    if np.isclose(norm, 0.0):
        return 0.0

    point = point / np.linalg.norm(point)
    theta = np.arctan2(point[1], point[0])
    if theta < 0.0:
        theta = theta + 2 * np.pi
    return theta


def get_symmetric_reduced_angle(angle, angular_range=2 * np.pi):
    """Given an angle and a range, reduce the angle to the interval
    [-range/2, range/2]
    """

    mod = np.mod(angle, angular_range)
    if np.isclose(mod, angular_range / 2) or mod > angular_range / 2:
        mod = -(angular_range - mod)
    if np.isclose(mod, -angular_range / 2):
        mod = angular_range / 2
    return mod


def is_simplex_non_degenerate(simplex):
    try:
        hull = scipy.spatial.ConvexHull(simplex)
        return True
    except scipy.spatial.qhull.QhullError:
        return False


def get_simplices_volume(simplices):
    volume = 0.0
    for simplex in simplices:
        try:
            hull = scipy.spatial.ConvexHull(simplex)
            volume += hull.volume
        except scipy.spatial.qhull.QhullError:
            pass
    return volume


def get_simplices_regular(simplices):
    num_simplices = len(simplices)
    new_simplices = np.zeros((num_simplices, 7, 8))

    k1_x = simplices[:, :, 0]
    k1_y = simplices[:, :, 1]
    k2_x = simplices[:, :, 2]
    k2_y = simplices[:, :, 3]
    d_x = simplices[:, :, 4]
    d_y = simplices[:, :, 5]

    ki_x = k1_x + d_x / 2
    ki_y = k1_y + d_y / 2
    kj_x = k1_x - d_x / 2
    kj_y = k1_y - d_y / 2
    ku_x = k2_x + d_x / 2
    ku_y = k2_y + d_y / 2
    kv_x = k2_x - d_x / 2
    kv_y = k2_y - d_y / 2

    new_simplices[:, :, 0] = ki_x
    new_simplices[:, :, 1] = ki_y
    new_simplices[:, :, 2] = kj_x
    new_simplices[:, :, 3] = kj_y
    new_simplices[:, :, 4] = ku_x
    new_simplices[:, :, 5] = ku_y
    new_simplices[:, :, 6] = kv_x
    new_simplices[:, :, 7] = kv_y
    return new_simplices


def intersect_hull_with_hyperplane(
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

    # Loop over vertices and compute the signs of r*n - d
    signs = []
    for point in points:
        value = np.dot(n, point) - d

        # Clean up edge cases
        value = 0.0 if np.isclose(value, 0.0) else value

        # Get the sign of r*n -d
        sign = np.sign(value)
        signs.append(sign)
    counter = 0
    counter2 = 0
    # Loop over edges and find intersections where they exist
    sorted_edges = set()
    intersections = []
    for simplex in simplices:
        for edge in itertools.combinations(simplex, 2):
            # Check if this edge has been done already. There may be repeats
            # because edges can be shared among multiple simplices
            sorted_edge = tuple(sorted(edge))
            if sorted_edge in sorted_edges:
                continue
            sorted_edges.add(sorted_edge)

            # This is a new edge that hasn't been checked before. Check for
            # intersection
            sign1 = signs[sorted_edge[0]]
            sign2 = signs[sorted_edge[1]]
            product = sign1 * sign2

            # Both vertices lie on the same side of the hyperplane
            if product == 1:
                continue

            v1 = points[sorted_edge[0]]
            v2 = points[sorted_edge[1]]

            # At least one vertex is in the plane
            if product == 0:
                if sign1 == 0:
                    if not any(np.allclose(v1, pt) for pt in intersections):
                        counter += 1
                        intersections.append(v1)
                if sign2 == 0:
                    if not any(np.allclose(v2, pt) for pt in intersections):
                        intersections.append(v2)
                        counter += 1
                continue

            # The product must be -1, so there is an intersection
            intersection = v1 + (d - np.dot(v1, n)) / np.dot(v2 - v1, n) * (
                v2 - v1
            )
            counter2 += 1
            intersections.append(intersection)
    print(counter)
    print(counter2)
    print(len(intersections))
    return np.array(intersections)


def intersect_hull_with_hyperplane_v2(
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
    intersections = np.vstack((intersections, intersection_points))
    return intersections


def get_degenerate_hull_simplices(
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
    coded = xp.unique(array_utils.bitwise_hash(edge_indices))
    edges = array_utils.inverse_bitwise_hash(coded)

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


def get_convex_hull_iterative(
    points: np.ndarray,
    max_iterations: int = 50,
    verbose: bool = False,
    qhull_options: str | None = None,
) -> scipy.spatial.ConvexHull:
    """Iteratively compute the convex hull of a set of points, using the
    computed vertices as the input points for the next iteration. This ensures
    that redundant points are gradually filtered out, resulting in a hull with
    minimal vertices and simplices."""
    old_points = points
    num_old_vertices = len(old_points)

    for i in range(max_iterations):
        hull = scipy.spatial.ConvexHull(
            old_points, qhull_options=qhull_options
        )

        # Check if the number of points has decreased or not.
        # If it hasn't, continue.
        new_vertices = hull.vertices
        new_simplices = hull.simplices
        new_points = hull.points[new_vertices]
        num_new_vertices = len(new_vertices)
        num_new_simplices = len(new_simplices)

        if verbose:
            print(
                f"Iteration {i + 1}: "
                f"Vertices={num_new_vertices}, "
                f"Simplices={num_new_simplices}, "
                f"Volume={hull.volume:.4f}."
            )

        is_equal_vertices = num_new_vertices == num_old_vertices
        is_equal_simplices = (
            True if i == 0 else num_new_simplices == num_old_simplices
        )
        is_finished = (
            num_new_vertices == num_old_vertices and is_equal_simplices
        )
        if is_finished:
            break

        # The number of vertices has changed, so repeat
        old_points = new_points
        num_old_vertices = num_new_vertices
        num_old_simplices = num_new_simplices

    return hull


def get_intersection_vertices_numpy(
    vertices: np.ndarray,
    correlation_signature: list[int] | None = None,
) -> np.ndarray:
    """Compute the intersection of the polytope with given vertices with
    hyperplanes as defined by the memory effect condition described by the
    given correlation signature."""
    # Set up cdd matrix object for the initial polytope
    t = np.ones(len(vertices))

    polytope_mat = cdd.matrix_from_array(
        np.column_stack((t.T, vertices)), rep_type=cdd.RepType.GENERATOR
    )

    # Get the halfspace representation inequalities
    polytope = cdd.polyhedron_from_matrix(polytope_mat)
    polytope_inequalities = np.array(cdd.copy_inequalities(polytope).array)

    # Intersect the polytope with the hyperplanes
    if correlation_signature is None:
        correlation_signature = [1, -1, -1, 1]
    a, b, c, d = correlation_signature
    hyperplane_equations = np.array(
        [[0, a, 0, b, 0, c, 0, d, 0], [0, 0, a, 0, b, 0, c, 0, d]]
    )
    lin_set = set([0, 1])
    augmented_inequalities = np.vstack(
        (hyperplane_equations, polytope_inequalities)
    )

    intersection_mat = cdd.matrix_from_array(
        augmented_inequalities,
        rep_type=cdd.RepType.INEQUALITY,
        lin_set=lin_set,
    )
    intersection = cdd.polyhedron_from_matrix(intersection_mat)
    intersection_vertices = np.array(cdd.copy_generators(intersection).array)

    if intersection_vertices.ndim == 1:
        return None

    truncated_vertices = intersection_vertices[:, 1:]
    return truncated_vertices


def get_intersection_vertices_fraction(
    vertices: np.ndarray,
    correlation_signature: list[int] | None = None,
) -> np.ndarray | None:
    """Compute the intersection of the polytope with given vertices with
    hyperplanes as defined by the memory effect condition described by the
    given correlation signature, using exact fractions and GMP mode."""

    # Convert vertices to Fractions for exact arithmetic
    vertices_frac = np.vectorize(Fraction)(vertices)

    # Generator matrix with ones in first column
    t = np.ones(len(vertices_frac), dtype=object)
    gen_matrix = np.column_stack((t, vertices_frac))
    polytope_mat = cdd.gmp.matrix_from_array(
        gen_matrix, rep_type=cdd.RepType.GENERATOR
    )

    # Get inequalities of the polytope in exact arithmetic
    polytope = cdd.gmp.polyhedron_from_matrix(polytope_mat)
    polytope_inequalities = np.array(
        cdd.gmp.copy_inequalities(polytope).array, dtype=object
    )

    # Hyperplanes to intersect
    if correlation_signature is None:
        correlation_signature = [1, -1, -1, 1]
    a, b, c, d = correlation_signature
    hyperplane_equations = np.array(
        [[0, a, 0, b, 0, c, 0, d, 0], [0, 0, a, 0, b, 0, c, 0, d]],
        dtype=object,
    )

    # Set equality indices
    lin_set = {0, 1}

    # Stack hyperplanes and inequalities
    augmented_matrix = np.vstack((hyperplane_equations, polytope_inequalities))

    # Convert all entries to Fractions
    augmented_matrix_frac = np.vectorize(Fraction)(augmented_matrix)

    # Build CDD matrix in GMP mode
    intersection_mat = cdd.gmp.matrix_from_array(
        augmented_matrix_frac,
        rep_type=cdd.RepType.INEQUALITY,
        lin_set=lin_set,
    )

    # Compute intersection polyhedron
    intersection = cdd.gmp.polyhedron_from_matrix(intersection_mat)
    intersection_vertices_frac = np.array(
        cdd.gmp.copy_generators(intersection).array, dtype=object
    )

    # No intersection
    if intersection_vertices_frac.ndim == 1:
        return None

    # Drop first column (weights)
    truncated_vertices = intersection_vertices_frac[:, 1:]

    # Convert Fractions back to floats
    truncated_vertices_float = np.vectorize(float)(truncated_vertices)

    return truncated_vertices_float


def get_intersection_vertices_dirac_density(
    vertices: np.ndarray | cp.ndarray,
    kj: np.ndarray | cp.ndarray,
    kv: np.ndarray | cp.ndarray,
    correlation_signature: list[int] | None = None,
) -> np.ndarray | cp.ndarray:
    """Compute the intersection of the polytope with given vertices with
    hyperplanes as defined by the memory effect condition described by the
    given correlation signature."""
    xp = cp.get_array_module(vertices)
    # Set up cdd matrix object for the initial polytope
    t = xp.ones(len(vertices))
    polytope_mat = cdd.matrix_from_array(
        xp.column_stack((t.T, vertices)), rep_type=cdd.RepType.GENERATOR
    )

    # Get the halfspace representation inequalities
    polytope = cdd.polyhedron_from_matrix(polytope_mat)
    polytope_inequalities = xp.array(cdd.copy_inequalities(polytope).array)

    # Intersect the polytope with the hyperplanes
    if correlation_signature is None:
        correlation_signature = [1, -1, -1, 1]
    a, b, c, d = correlation_signature
    hyperplane_equations = xp.array(
        [
            [0, a, 0, b, 0, c, 0, d, 0],
            [0, 0, a, 0, b, 0, c, 0, d],
            [kj[0], 0, 0, -1, 0, 0, 0, 0, 0],
            [kj[1], 0, 0, 0, -1, 0, 0, 0, 0],
            [kv[0], 0, 0, 0, 0, 0, 0, -1, 0],
            [kv[1], 0, 0, 0, 0, 0, 0, 0, -1],
        ]
    )
    lin_set = set([0, 1, 2, 3, 4, 5])
    augmented_inequalities = xp.vstack(
        (hyperplane_equations, polytope_inequalities)
    )
    intersection_mat = cdd.matrix_from_array(
        augmented_inequalities,
        rep_type=cdd.RepType.INEQUALITY,
        lin_set=lin_set,
    )
    intersection = cdd.polyhedron_from_matrix(intersection_mat)
    intersection_vertices = xp.array(cdd.copy_generators(intersection).array)

    # Truncate the
    truncated_vertices = intersection_vertices[:, 1:]
    return truncated_vertices


def get_minkowski_filter_area(
    mode_i: np.ndarray,
    mode_j: np.ndarray,
    mode_u: np.ndarray,
    mode_v: np.ndarray,
) -> np.float128:
    """Given four shapes (defined by vertices), find the area of the minkowski filtering process"""
    try:
        # Find the centroids
        mean_i = np.mean(mode_i, axis=0)
        mean_j = np.mean(mode_j, axis=0)
        centre_ij = np.mean(np.vstack((mean_i, mean_j)), axis=0)
        mean_u = np.mean(mode_u, axis=0)
        mean_v = np.mean(mode_v, axis=0)
        centre_uv = np.mean(np.vstack((mean_u, mean_v)), axis=0)

        # Find the difference space associated with centre_ij
        mode_j_ref = reflect_through_point(mode_j, centre_ij)
        ij_intersect = intersection(mode_i, mode_j_ref)
        new_ij = 2 * translate_points(ij_intersect, -centre_ij)

        # Find the difference space associated with centre_uv
        mode_v_ref = reflect_through_point(mode_v, centre_uv)
        uv_intersect = intersection(mode_u, mode_v_ref)
        new_uv = 2 * translate_points(uv_intersect, -centre_uv)

        # Find the intersection of the difference spaces and get
        # its area
        ijuv_intersect = intersection(new_ij, new_uv)
        area = get_polygon_area(ijuv_intersect)
        return np.float128(area if np.isfinite(area) else 0.0)
    except Exception as e:
        return np.float128(0.0)


def get_six_dimensional_intersection_volume(
    mode_i: np.ndarray,
    mode_j: np.ndarray,
    mode_u: np.ndarray,
    mode_v: np.ndarray,
) -> float:
    """Find the volume associated with the intersection of the Cartesian
    product of the four modes"""
    # Find the volume associated with no shift
    columns_to_keep = [0, 1, 2, 3, 4, 5]
    cartesian_product = iterated_cartesian_product(
        [mode_i, mode_j, mode_u, mode_v]
    )
    try:
        reduced_intersection = get_intersection_vertices_numpy(
            cartesian_product
        )[:, columns_to_keep]
    except RuntimeError:
        reduced_intersection = get_intersection_vertices_fraction(
            cartesian_product
        )[:, columns_to_keep]

    try:
        reduced_hull = scipy.spatial.ConvexHull(
            reduced_intersection, qhull_options="QJ"
        )
    except Exception as _:
        return 0.0
    return reduced_hull.volume
