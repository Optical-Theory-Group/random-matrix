"""Utility functions that perform various geometric calculations."""

from itertools import combinations

import numpy as np
import numpy.typing as npt
import scipy.spatial
import shapely
import skspatial.objects

from random_matrix.utils import array_utils, geometry_utils, plotting_utils
from random_matrix.utils.types import FloatLike


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


def minkowski_sum(polygon_one: FloatLike, polygon_two: FloatLike) -> FloatLike:
    points_one = order_points(polygon_one)
    points_two = order_points(polygon_two)
    num_points_one = len(points_one)
    num_points_two = len(points_two)
    i = 0
    j = 0

    new_points = []

    while i <= num_points_one and j <= num_points_two:
        first_point = points_one[i % num_points_one]
        second_point = points_two[j % num_points_two]

        next_first = points_one[(i + 1) % num_points_one]
        next_second = points_two[(j + 1) % num_points_two]

        new_points.append(first_point + second_point)

        theta_one = (
            np.arctan2(
                next_first[1] - first_point[1], next_first[0] - first_point[0]
            )
            + np.pi
        )
        theta_two = (
            np.arctan2(
                next_second[1] - second_point[1],
                next_second[0] - second_point[0],
            )
            + np.pi
        )
        if np.isclose(theta_one, theta_two):
            i += 1
            j += 1
        elif theta_one > theta_two:
            j += 1
        else:
            i += 1

    return np.array(new_points)


def minkowski_sum(points_one: FloatLike, points_two: FloatLike) -> FloatLike:
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
        theta_one = 2 * np.pi + theta_one if theta_one < 0 else theta_one

        theta_two = np.arctan2(
            next_second[1] - second_point[1],
            next_second[0] - second_point[0],
        )
        theta_two = 2 * np.pi + theta_two if theta_two < 0 else theta_two

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


def minkowski_difference(
    points_one: FloatLike, points_two: FloatLike
) -> FloatLike:
    return minkowski_sum(points_one, -points_two)


def intersects(points_one: FloatLike, points_two: FloatLike) -> bool:
    """Returns a bool that is true if the two polygons intersect"""

    polygon_one = shapely.Polygon(points_one)
    polygon_two = shapely.Polygon(points_two)
    return not np.isclose(polygon_one.intersection(polygon_two).area, 0.0)

def intersection(points_one, points_two):
    polygon_one = shapely.Polygon(points_one)
    polygon_two = shapely.Polygon(points_two)
    intersection = polygon_one.intersection(polygon_two)
    intersection = np.array(intersection.exterior.coords)
    return intersection[:-1]

def cartesian_product(polygon1: FloatLike, polygon2: FloatLike) -> FloatLike:
    # Get the number of vertices in each polygon
    n1 = polygon1.shape[0]
    n2 = polygon2.shape[0]

    # Repeat each vertex of polygon1 n2 times
    repeated_polygon1 = np.repeat(polygon1, n2, axis=0)

    # Tile polygon2 to match the repeated polygon1
    tiled_polygon2 = np.tile(polygon2, (n1, 1))

    # Concatenate the repeated polygon1 and tiled polygon2
    cartesian_product = np.concatenate(
        (repeated_polygon1, tiled_polygon2), axis=1
    )

    return cartesian_product

def reflect_through_point(shape: FloatLike, point: FloatLike) -> FloatLike:
    reflected = point - (shape - point)
    return reflected