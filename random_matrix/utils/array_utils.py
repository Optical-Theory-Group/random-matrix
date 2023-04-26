"""This module contains utility functions for frequently used array
manipulations.
"""
from typing import Any

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


def is_in_array(val: float, array: Vector[np.float32]) -> np.bool_:
    """
    Check if a given value is present in a NumPy array of floats within a
    small tolerance.

    Parameters
    ----------
    val : float
        The value to look for in the array.
    array : numpy.ndarray or list of float
        A 1D array or list of float values to search for the given value.

    Returns
    -------
    numpy.bool_
        A boolean value indicating whether the given value is present in the
        array within a small tolerance (specified by the default
        `atol` parameter of `np.isclose()`).
    """
    in_array = np.any(np.isclose(val, array))
    return in_array


def is_equal_array(
    first_array: np.ndarray[Any, Any],
    second_array: np.ndarray[Any, Any],
    order_matters: bool = False,
) -> np.bool_:
    """
    Check if two NumPy arrays of dimensions 1 or 2 are equal
    (within a small tolerance).

    Parameters
    ----------
    first_array : numpy.ndarray
        The first array to compare.
    second_array : numpy.ndarray
        The second array to compare.
    order_matters : bool, optional
        If False (default), sort both arrays before comparison.
        If True, do not sort the arrays.

    Returns
    -------
    numpy.bool_
        A boolean value indicating whether the two arrays are equal within a
        small tolerance (specified by the default `atol` parameter of
        `np.isclose()`).
    """

    dim_first = first_array.ndim
    dim_second = second_array.ndim
    if dim_first != dim_second:
        raise ValueError("Cannot compare arrays of different dimensions.")

    # Sort arrays if the order doesnt matter
    if not order_matters:
        match dim_first:
            case 1:
                first_array = np.sort(first_array)
                second_array = np.sort(second_array)
            case 2:
                sorted_indices_first = np.lexsort(
                    (first_array[:, 1], first_array[:, 0])
                )
                first_array = first_array[sorted_indices_first]

                sorted_indices_second = np.lexsort(
                    (second_array[:, 1], second_array[:, 0])
                )
                second_array = second_array[sorted_indices_second]

    are_equal = np.all(np.isclose(first_array, second_array))

    return are_equal


def get_array_index(
    val: float, array: Vector[np.float32] | Matrix[np.float32]
) -> int:
    """
    Find the index of a value within a NumPy array.

    Parameters
    ----------
    val : float
        The value to search for.
    array : numpy.ndarray
        The array to search in.

    Returns
    -------
    int
        The index of the first occurrence of the value in the array.
    """
    index: int = np.where(np.isclose(array, val))[0][0]
    return index


def remove_element_by_index(
    array: np.ndarray[Any, Any], index: int
) -> np.ndarray[Any, Any]:
    """
    Remove an element from a numpy array by index.

    Parameters
    ----------
    array : np.ndarray
        The numpy array from which to remove an element.
    index : int
        The index of the element to remove.

    Returns
    -------
    np.ndarray
        A new numpy array with the specified element removed.
    """

    lower = array[:index]
    upper = array[index + 1 :]
    filtered: np.ndarray[Any, Any] = np.concatenate((lower, upper), axis=0)
    return filtered


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


def vals_to_box(
    first_vals: Vector[np.float32], second_vals: Vector[np.float32]
) -> Matrix[np.float32]:
    """
    Given lists of 2 first_vals and 2 second_vals, return a list of points of
    the corresponding rectangle.

    Parameters
    ----------
    first_vals : np.ndarray
        First array of coordinates.
    second_vals : np.ndarray
        Second array of coordiantes.

    Returns
    -------
    box_points : np.ndarray
        Array of points in the resultant rectangle.
    """

    box_first_grid, box_second_grid = np.meshgrid(first_vals, second_vals)
    box_points = np.column_stack(
        (box_first_grid.ravel(), box_second_grid.ravel())
    )
    return box_points


def sort_by_reference_list(
    to_be_sorted: np.ndarray[Any, Any], reference_list: list[float]
) -> np.ndarray[Any, Any]:
    """
    Sort a list or array by the order of a reference list.

    Parameters
    ----------
    to_be_sorted : List[Any]
        The list or array to be sorted.
    reference_list : List[Any]
        The reference list used to determine the order of the sorted list.

    Returns
    -------
    np.ndarray
        The sorted array.
    """

    to_be_sorted = np.array(to_be_sorted)
    sorted_indices = np.argsort(reference_list)
    sorted_list = to_be_sorted[np.array(sorted_indices)]
    return sorted_list
