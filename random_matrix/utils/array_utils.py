import numpy as np
from scipy.spatial import cKDTree

def remove_duplicate_points(points, tolerance=1e-8):
    """
    Remove duplicate points from an array of points.
    Complexity: O(N log N)

    Parameters:
    -----------
    points : numpy.ndarray
        An array of shape (N,) or (N, 2) or (N, 3) representing the 1D, 2D or 3D points.
    tolerance : float, optional (default=1e-8)
        The distance tolerance used to identify duplicate points. Points that are within
        a distance of `tolerance` of each other are considered duplicates.

    Returns:
    --------
    unique_points : numpy.ndarray
        An array of shape (M,) or (M, 2) or (M, 3) representing the unique 1D, 2D or 3D points, where
        M <= N.
    """

    if points.ndim == 1:
        points = points[:, np.newaxis]
    
    tree = cKDTree(points)
    duplicates = tree.query_ball_point(points, r=tolerance)
    unique_indices = sorted(set([min(d) for d in duplicates]))
    new_points = points[unique_indices].squeeze()
    return new_points

def get_cyclic_pairs(points):
    pairs = np.stack((points, np.roll(points, -2)), axis=1)    
    return pairs