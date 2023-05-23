import pytest

from random_matrix.utils import matrix_utils


def test_get_matrix_sub_block_indices():
    out = matrix_utils.get_matrix_sub_block_indices(
        "r", (0, 0), False, 100
    )
    assert out == (0, 0)

    out = matrix_utils.get_matrix_sub_block_indices(
        "r", (0, 0), True, 5
    )
    assert out == (4, 4)

    out = matrix_utils.get_matrix_sub_block_indices(
        "r", (-2, 0), True, 5
    )
    assert out == (0, 4)

    out = matrix_utils.get_matrix_sub_block_indices(
        "t", (1, -1), True, 5
    )
    assert out == (16, 2)

    out = matrix_utils.get_matrix_sub_block_indices(
        "r2", (2, 4), False, 5
    )

    assert out == (14, 18)
