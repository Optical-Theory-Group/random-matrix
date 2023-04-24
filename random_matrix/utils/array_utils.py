"""This module contains utility functions for frequently used array manipulations.
"""

import numpy as np
from scipy.spatial import cKDTree

from random_matrix.types.array_types import Matrix, Vector


def remove_duplicate_points(
    points: Vector[np.float32] | Matrix[np.float32], tolerance: float = 1e-8
) -> Vector[np.float32] | Matrix[np.float32]:
    """
    Remove duplicate points from an array of points.
    Complexity: O(N log N)

    Parameters:
    -----------
    points : numpy.ndarray
        An array of shape (N,) (Vector) or (N, 2) (Matrix)
        representing the 1D or 2D points.
    tolerance : float, optional (default=1e-8)
        The distance tolerance used to identify duplicate points. Points that
        are within a distance of `tolerance` of each other are considered
        duplicates.

    Returns:
    --------
    unique_points : numpy.ndarray
        An array of shape (M,) or (M, 2) representing the unique 1D or 2D
        points, where M <= N.
    """

    # In the 1D case, artifically add extra axis. Required to make scipy's tree
    # object happy. This is later removed with squeeze().
    if points.ndim == 1:
        points = points[:, np.newaxis]

    tree = cKDTree(points)
    duplicates = tree.query_ball_point(points, r=tolerance)
    unique_indices = sorted(set([min(d) for d in duplicates]))
    new_points = points[unique_indices].squeeze()
    return new_points


def get_pairs(
    points: Matrix[np.float32], cyclic: bool = False
) -> Matrix[np.float32]:
    """
    Given an array of shape (N,), or an array of 2D points stored in a (N,2)
    matrix, return a NumPy array containing pairs of values of the form
    (1st, 2nd)
    (2nd, 3rd)
    ...
    (N-1th, Nth)

    If cyclic is true, include also
    (Nth, 1st)

    Parameters:
    ----------
    points : numpy.ndarray
        An array of 1D or 2D points.

    Returns:
    -------
    pairs : numpy.ndarray
        An array of pairs of points in the manner described. If cyclic is false
        the array will contain N-1 points. If cyclic is true, it will contain N
        points.
    """

    dimension = points.ndim
    pairs = np.stack((points, np.roll(points, -dimension)), axis=1)
    if not cyclic:
        pairs = pairs[:-1]
    return pairs


def get_point_index(
    point: Vector[np.float32], point_array: Matrix[np.float32]
) -> int | None:
    """
    Return the index of the first point in `point_array` that is close to the
    given `point`.

    Parameters
    ----------
    point : Vector[np.float32]
        The point to search for in `point_array`.
    point_array : Matrix[np.float32]
        The matrix of points to search through, where each row represents a
        point.

    Returns
    -------
    int or None
        If a close point is found in `point_array`, return its index
        (i.e., row number). If no close point is found, return `None`.
    """

    indices = np.where(np.all(np.isclose(point, point_array), axis=1))[0]
    if len(indices) == 0:
        return None
    else:
        index: int = indices[0]
        return index
