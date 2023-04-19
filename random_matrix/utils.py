"""
This module contains various utility functions that are used in other parts of the codebase.

If this file becomes too large, it may be split into multiple files for better organization.

Please refer to the documentation of each individual function for more information on how to use it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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

def draw_ray(ax, theta=0, r_min=0, r_max=1, color="tab:blue", linestyle="--",alpha=1.0):
    x = np.array([r_min*np.cos(theta), r_max*np.cos(theta)])
    y = np.array([r_min*np.sin(theta), r_max*np.sin(theta)])
    ax.plot(x, y, linestyle=linestyle, color=color,alpha=alpha)
 
def draw_circle(ax, r=1, t_min=0, t_max=2*np.pi, color="black", linestyle="-"):
    t = np.linspace(t_min, t_max)
    x = r*np.cos(t)
    y = r*np.sin(t)
    ax.plot(x,y,color=color, linestyle=linestyle)

def draw_k_space():
    fig, ax = plt.subplots()
    draw_ray(ax, r_min=-1, theta=0, linestyle="-", color="black", alpha=0.4)
    draw_ray(ax, r_min=-1, theta=np.pi/2, linestyle="-", color="black", alpha=0.4)
    ax.set_aspect("equal")
    draw_circle(ax)
    return ax