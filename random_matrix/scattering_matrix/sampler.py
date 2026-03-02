import numpy as np

from random_matrix.utils import matrix_utils


def reorder_block(elements: np.ndarray) -> np.ndarray:
    length_of_elements, num_matrices = np.shape(elements)
    size_of_block = int(np.sqrt(len(elements)))

    even_row_indices = np.sort(
        np.concatenate(
            (
                np.arange(0, length_of_elements, 4),
                np.arange(1, length_of_elements, 4),
            )
        )
    )
    odd_row_indices = np.sort(
        np.concatenate(
            (
                np.arange(2, length_of_elements, 4),
                np.arange(3, length_of_elements, 4),
            )
        )
    )

    even_rows = elements[even_row_indices].reshape(
        int(size_of_block / 2), size_of_block, num_matrices
    )
    odd_rows = elements[odd_row_indices].reshape(
        int(size_of_block / 2), size_of_block, num_matrices
    )
    final = np.empty(
        (
            even_rows.shape[0] + odd_rows.shape[0],
            even_rows.shape[1],
            num_matrices,
        ),
        dtype=np.complex128,
    )
    final[::2, :, :] = even_rows
    final[1::2, :, :] = odd_rows
    return final


def S_sampler(
    mean_S: np.ndarray,
    chol: np.ndarray,
    num_matrices: int = 1,
    symmetrize: bool = True,
) -> np.ndarray:
    size_of_S, _ = np.shape(mean_S)
    size_of_block = int(size_of_S / 2)
    num_random_numbers, _ = np.shape(chol)

    # Generate random numbers for the matrices
    random_numbers = np.random.randn(num_random_numbers, num_matrices)
    random_numbers = chol @ random_numbers
    reals = random_numbers[0 : int(num_random_numbers / 2)]
    imags = random_numbers[
        int(num_random_numbers / 2) : int(num_random_numbers)
    ]

    # Extract matrix elements from random numbers
    num_random_numbers = int(num_random_numbers / 2)
    r = (
        reals[0 : int(num_random_numbers / 4)]
        + 1j * imags[0 : int(num_random_numbers / 4)]
    )
    t = (
        reals[int(num_random_numbers / 4) : int(num_random_numbers / 2)]
        + 1j * imags[int(num_random_numbers / 4) : int(num_random_numbers / 2)]
    )
    t2 = (
        reals[int(num_random_numbers / 2) : int(num_random_numbers * 3 / 4)]
        + 1j
        * imags[int(num_random_numbers / 2) : int(num_random_numbers * 3 / 4)]
    )
    r2 = (
        reals[int(num_random_numbers * 3 / 4) : num_random_numbers]
        + 1j * imags[int(num_random_numbers * 3 / 4) : num_random_numbers]
    )

    # Reorder the randomly generated numbers into the correct shapes
    r = reorder_block(r)
    t = reorder_block(t)
    t2 = reorder_block(t2)
    r2 = reorder_block(r2)

    # Add identity to transmission matrices
    identity = np.identity(size_of_block)
    t = t + identity[:, :, np.newaxis]
    t2 = t2 + identity[:, :, np.newaxis]

    top = np.hstack([r, t2])
    bottom = np.hstack([t, r2])
    whole = np.vstack([top, bottom])

    # Add the mean matrix to each instance
    whole = whole - mean_S[:, :, np.newaxis]
    output = np.transpose(whole, (2, 0, 1))

    if symmetrize:
        output = matrix_utils.get_closest_unitary_approximation(output)
    return output
