"""Unit tests for the array_utils module"""

import numpy as np

from random_matrix.utils import array_utils


def test_remove_duplicate_points() -> None:
    # 1D - distinct points
    array = np.array([1.0, 2.0, 3.0, 4.0])
    actual = array_utils.remove_duplicate_points(array)
    desired = np.copy(array)
    np.testing.assert_allclose(actual, desired)

    # 1D - repetitions
    array = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0])
    actual = array_utils.remove_duplicate_points(array)
    desired = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(actual, desired)

    # 2D - distinct points
    array = np.array([[1.0, 2.0], [-2.0, 5.0], [4.0, 3.0]])
    actual = array_utils.remove_duplicate_points(array)
    desired = np.copy(array)
    np.testing.assert_allclose(actual, desired)

    # 2D - repetitions
    array = np.array(
        [[-1.0, 2.0], [4.0, 3.0], [-1.0, 2.0], [0.0, 0.0], [4.0, 3.0]]
    )
    actual = array_utils.remove_duplicate_points(array)
    desired = np.array([[-1.0, 2.0], [4.0, 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(actual, desired)


def test_get_pairs() -> None:
    # 1D non cyclic
    array = np.array([1.0, 2.0, 3.0])
    actual = array_utils.get_pairs(array)
    desired = np.array([[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(actual, desired)

    # 1D cyclic
    actual = array_utils.get_pairs(array, cyclic=True)
    desired = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    np.testing.assert_allclose(actual, desired)

    # 2D non cyclic
    array = np.array([[1.0, 2.0], [3.0, -1.0], [-1.0, 0.0]])
    actual = array_utils.get_pairs(array)
    desired = np.array([[[1.0, 2.0], [3.0, -1.0]], [[3.0, -1.0], [-1.0, 0.0]]])
    np.testing.assert_allclose(actual, desired)

    # 2D cyclic
    actual = array_utils.get_pairs(array, cyclic=True)
    desired = np.array(
        [
            [[1.0, 2.0], [3.0, -1.0]],
            [[3.0, -1.0], [-1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 2.0]],
        ]
    )
    np.testing.assert_allclose(actual, desired)


def test_is_in_array() -> None:
    # 1D is in
    array = np.array([1.0, 2.0, 3.0])
    actual = array_utils.is_in_array(array, 1.0)
    assert actual

    # 1D not in
    actual = array_utils.is_in_array(array, 1.5)
    assert not actual

    # 2D is in
    array = np.array([[1.0, 2.0], [3.0, 1.0], [-1.0, 2.0]])
    actual = array_utils.is_in_array(array, np.array([1.0, 2.0]))
    assert actual

    # 2D not in
    actual = array_utils.is_in_array(array, np.array([2.0, 3.0]))
    assert not actual

    # scalar within 2D array?
    actual = array_utils.is_in_array(array, 3.0)
    # assert not actual
