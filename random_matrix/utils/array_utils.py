"""
Module containing utility functions for common array manipulations.
"""

import numpy as np
from scipy.spatial import cKDTree
from ._array_types import Vector, Matrix


def remove_duplicate_points(points: Vector[np.float32] | Matrix[np.float32],
                            tolerance: float = 1e-8
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

    # In 1D case, artifically add extra axis. Required to make tree happy.
    # This is later removed with squeeze().
    if points.ndim == 1:
        points = points[:, np.newaxis]

    tree = cKDTree(points)
    duplicates = tree.query_ball_point(points, r=tolerance)
    unique_indices = sorted(set([min(d) for d in duplicates]))
    new_points = points[unique_indices].squeeze()
    return new_points


def get_pairs(points: Matrix[np.float32], cyclic: bool = False
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
