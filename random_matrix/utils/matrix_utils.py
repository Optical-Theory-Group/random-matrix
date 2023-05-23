import numpy as np
from random_matrix.utils.types import FloatLike


def get_sub_block_indices(
    block: str,
    sub_block: tuple[int, int],
    is_reciprocal: bool,
    num_propagating: int,
    num_evanescent: int = 0,
    wave_block: str = "pp",
) -> tuple[int, int]:
    """Get the matrix indices from information about the block"""

    second_half = 2 * num_propagating

    if block == "r":
        row = 0
        col = 0
    elif block == "t":
        row = second_half
        col = 0
    elif block == "t2":
        row = 0
        col = second_half
    elif block == "r2":
        row = second_half
        col = second_half

    j = sub_block[0]
    i = sub_block[1]
    row = row + 2 * j
    col = col + 2 * i

    if is_reciprocal:
        row = row + num_propagating - 1
        col = col + num_propagating - 1

    return (row, col)


def r_sym(matrix: FloatLike) -> FloatLike:
    """Returns the 'reciprocal operator' applied to a matrix as defined in

    https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.013129
    """
    x, y = np.shape(matrix)
    b = np.ones((y, x), dtype=np.complex128)
    b[1::2, ::2] = -1.0
    b[::2, 1::2] = -1.0
    out = np.multiply(matrix.T, b)
    return out


def get_reciprocal_sub_block_indices(
    block: str,
    sub_block: tuple[int, int],
    num_propagating: int,
    num_evanescent: int = 0,
    wave_block: str = "pp",
) -> tuple[int, int]:
    """Given a sub block of S, find the location of its reciprocal partner."""

    reciprocal_blocks = {"r": "r", "r2": "r2", "t": "t2", "t2": "t"}
    reciprocal_block = reciprocal_blocks[block]
    reciprocal_sub_block = (-sub_block[1], -sub_block[0])

    return get_sub_block_indices(
        reciprocal_block,
        reciprocal_sub_block,
        True,
        num_propagating,
        num_evanescent,
        wave_block,
    )
