import numpy as np
from itertools import combinations
from random_matrix.utils.array_utils import remove_duplicate_points
from scipy.spatial import ConvexHull

def circle(x, r):
    """
    Compute the y-coordinate of a point on a circle given the x-coordinate and radius.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The x-coordinate(s) of the point(s) on the circle.
    r : float
        The radius of the circle.

    Returns:
    --------
    y : float or numpy.ndarray
        The y-coordinate(s) of the point(s) on the circle corresponding to the input x-coordinate(s).
    """

    return np.sqrt(r**2-x**2)

def cartesian_to_polar(points_cartesian):
    """
    Convert an array of 2D points from Cartesian coordinates to polar coordinates.

    Parameters:
    -----------
    points_cartesian : numpy.ndarray
        An array of shape (N, 2) representing the 2D points in Cartesian coordinates.

    Returns:
    --------
    points_polar : numpy.ndarray
        An array of shape (N, 2) representing the 2D points in polar coordinates. The first
        column contains the radius and the second column contains the angle in radians.
    """

    x = points_cartesian[:,0]
    y = points_cartesian[:,1]
    r = np.linalg.norm(points_cartesian, axis=1)
    t = np.arctan2(y, x)
    points_polar = np.column_stack((r, t)) 
    return points_polar

def polar_to_cartesian(points_polar):
    """
    Convert an array of 2D points from polar coordinates to Cartesian coordinates.

    Parameters:
    -----------
    points_polar : numpy.ndarray
        An array of shape (N, 2) representing the 2D points in polar coordinates. The first
        column should contain the radius and the second column should contain the angle in radians.

    Returns:
    --------
    points_cartesian : numpy.ndarray
        An array of shape (N, 2) representing the 2D points in Cartesian coordinates.
        The first column contains the x-coordinate and the second column contains the y-coordinate.
    """

    r = points_polar[:,0]
    t = points_polar[:,1]
    x = r*np.cos(t)
    y = r*np.sin(t)
    points_cartesian = np.column_stack((x, y))
    return points_cartesian

def is_rectangle(points):
    """
    Determines if a set of 4 2D points form a rectangle.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D numpy array of shape (4, 2) representing the 4 points.

    Returns
    -------
    bool
        True if the points form a rectangle, False otherwise.

    """

    if len(points) != 4:
        return False

    pairs = np.array([pair for pair in combinations(points, 2)])
    side_lengths = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1)
    unique_side_lengths = remove_duplicate_points(side_lengths)

    if len(unique_side_lengths) not in [2,3]:
        return False

    sorted_lengths = np.sort(unique_side_lengths)
    if len(sorted_lengths) == 2:
        sorted_lengths = np.insert(sorted_lengths, 0, sorted_lengths[0])
    a, b, c = sorted_lengths
    return np.isclose(a**2 + b**2, c**2)

def rotate_points(points, axis, rotation_angle):
    """
    Rotate a set of 2D points around a specified axis by a given angle.

    Parameters
    ----------
        points (np.ndarray): 
            A 2D numpy array of shape (n, 2) representing the coordinates of n points to be rotated.
        axis (np.ndarray): 
            A 1D numpy array of length 2 representing the center of rotation.
        rotation_angle (float): 
            The angle (in radians) by which to rotate the points.

    Returns:
        np.ndarray: A 2D numpy array of shape (n, 2) representing the rotated points.

    """

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    translated_points = points - axis
    rotated_points = rotation_matrix @ translated_points.T
    output = rotated_points.T + axis
    return output

def points_to_ordered_convex_hull_vertices(points):
    hull = ConvexHull(points)
    vertices = hull.vertices
    new_points = points[vertices]
    return new_points

def get_convex_hull_area(convex_hull):
    if isinstance(convex_hull, np.ndarray):
        convex_hull = ConvexHull(convex_hull)
    return convex_hull.volume

