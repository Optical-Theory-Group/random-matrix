"""Utility functions for frequently used array manipulations."""

from typing import Any, Type
import cupy as cp
import numpy as np
import numpy.typing as npt
import scipy.spatial
from fractions import Fraction

def get_module(array: np.ndarray | cp.ndarray) -> Any:
    return np if cp.get_array_module(array) == np else cp


def split_list(old_list: list[Any], num_parts: int) -> list[Any]:
    new_list = [[] for _ in range(num_parts)]
    for i, val in enumerate(old_list):
        index = i % num_parts
        new_list[index].append(val)
    return new_list


def remove_duplicate_points(
    points: npt.NDArray[np.float64],
    tolerance: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Remove duplicate points from an array of points.

    Parameters:
    -----------
        points : numpy.ndarray
            An array of shape (N,) (Vector) or (N, 2) (Matrix)
            representing the 1D or 2D points.
        tolerance : float, optional (default=1e-8)
            The distance tolerance used to identify duplicate points. Points
            that are within a distance of `tolerance` of each other are
            considered duplicates.

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

    tree = scipy.spatial.cKDTree(points)
    duplicates = tree.query_ball_point(points, r=tolerance)
    unique_indices = sorted(set([min(d) for d in duplicates]))
    new_points = points[unique_indices].squeeze()

    if not isinstance(new_points, np.ndarray):
        new_points = np.array([new_points])

    return new_points


def get_pairs(
    points: npt.NDArray[np.float64],
    cyclic: bool = False,
) -> npt.NDArray[np.float64]:
    """Get sequential pairs of elements in an array.

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
            An array of pairs of points in the manner described. If cyclic is
            false the array will contain N-1 points. If cyclic is true, it will
            contain N points.
    """

    dimension = points.ndim
    pairs = np.stack((points, np.roll(points, -dimension)), axis=1)
    if not cyclic:
        pairs = pairs[:-1]
    return pairs


def is_in_array(val: np.float64, array: npt.NDArray[np.float64]) -> bool:
    """Check if a given value is present in a NumPy array within a
    small tolerance.

    Parameters
    ----------
        val : float
            The value to look for in the array.
        array : numpy.ndarray
            A 1D array or list of float values to search for the given value.

    Returns
    -------
        bool
            A boolean value indicating whether the given value is present in
            the array within a small tolerance (specified by the default
            `atol` parameter of `np.isclose()`).
    """
    in_array = bool(np.any(np.isclose(val, array)))
    return in_array


def is_equal_array(
    first_array: npt.NDArray[np.float64],
    second_array: npt.NDArray[np.float64],
    order_matters: bool = False,
) -> bool:
    """Check if two NumPy arrays of dimensions 1 or 2 are equal within a
       small tolerance.

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
        bool
            A boolean value indicating whether the two arrays are equal within
            a small tolerance (specified by the default `atol` parameter of
            `np.isclose()`).
    """

    dim_first = first_array.ndim
    dim_second = second_array.ndim
    if dim_first != dim_second:
        raise ValueError("Cannot compare arrays of different dimensions.")

    shape_first = np.shape(first_array)
    shape_second = np.shape(second_array)
    if shape_first != shape_second:
        return False

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

    are_equal = bool(np.all(np.isclose(first_array, second_array)))
    return are_equal


def is_equal_cyclic(
    first_array: npt.NDArray[np.float64],
    second_array: npt.NDArray[np.float64],
):
    """Assumed that the arrays are both 2D"""

    if not np.all(np.isclose(np.shape(first_array), np.shape(second_array))):
        return False

    max_roll, roll_size = np.shape(first_array)
    for roll in (-i * roll_size for i in range(max_roll)):
        rolled_second = np.roll(second_array, roll)
        equal = np.allclose(first_array, rolled_second)
        if equal:
            return True
    return False


def get_array_index(val: np.float64, array: npt.NDArray[np.float64]) -> int:
    """Find the index of a value within a NumPy array.

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


def get_point_index(
    point: npt.NDArray[np.float64], point_array: npt.NDArray[np.float64]
) -> int | None:
    """Return the index of the first point in `point_array` that is close to
    the given `point`.

    Parameters
    ----------
        point : numpy.ndarray
            The point to search for in `point_array`.
        point_array : numpy.ndarray
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


def sort_by_reference_list(
    to_be_sorted: npt.NDArray[np.float64] | list[Any],
    reference_list: list[np.float64] | list[Any],
) -> npt.NDArray[np.float64] | list[Any]:
    """Sort a list or array by the order of a reference list.

    Parameters
    ----------
        to_be_sorted : numpy.ndarray
            The list or array to be sorted.
        reference_list : List[np.float64]
            The reference list used to determine the order of the sorted list.

    Returns
    -------
        numpy.ndarray
            The sorted array.
    """

    to_be_sorted = np.array(to_be_sorted)
    sorted_indices = np.argsort(reference_list)
    sorted_list = to_be_sorted[np.array(sorted_indices)]
    return sorted_list


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


def clean_zeros(array: np.ndarray, tol_power: float = 10) -> np.ndarray:
    """Set elements that are very small in magnitude to zero.

    Tol power should be such that values are maintained up to 10^-tol_power"""
    return np.round(array, decimals=tol_power)

def to_fraction_array(array: np.ndarray, max_den: int=1000000) -> np.ndarray:
    """Transform a numpy array of floats to an array of Fraction objects"""
    f = np.vectorize(lambda x: Fraction(x).limit_denominator(max_den))
    return f(array)