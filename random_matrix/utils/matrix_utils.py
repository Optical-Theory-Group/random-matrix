import numpy as np
import cupy as cp
import scipy
import sksparse.cholmod

# def get_sub_block_indices_vector(independent_element_indices):
#     indices = np.empty((0, 3), dtype="object")
#     for wave_block, inner in independent_element_indices.items():
#         for block, sub_block_locations in inner.items():
#             indices = np.vstack(indices, np.array([wave_block, blo]))
#             pass


def get_block_indices(block: str, num_propagating: int):
    """Get indices that pick out a particular block of a scattering matrix,
    i.e., r, t, t2 or r2."""
    half_size = num_propagating * 2
    is_top_row_half = block in ["r", "t2", "a", "b"]
    is_left_col_half = block in ["r", "t", "a", "c"]

    if is_top_row_half:
        row_start = 0
        row_end = half_size
    else:
        row_start = half_size
        row_end = 2 * half_size

    if is_left_col_half:
        col_start = 0
        col_end = half_size
    else:
        col_start = half_size
        col_end = 2 * half_size

    row_slice = slice(row_start, row_end)
    col_slice = slice(col_start, col_end)
    return (row_slice, col_slice)


def get_block(matrix: np.ndarray | cp.ndarray, block: str):
    """Get a block of the given S matrix, i.e. r, t, t2 or r2

    If S is 2 dimensional, it should be of shape n x n
    If S is 3 dimensional, it should be of shape M x n x n"""
    num_propagating = int(matrix.shape[1] / 4)
    block_indices = get_block_indices(block, num_propagating)
    if matrix.ndim == 2:
        return matrix[block_indices]
    elif matrix.ndim == 3:
        return matrix[:, block_indices[0], block_indices[1]]
    else:
        raise ValueError(
            f"matrix has ndim {matrix.ndim}, but it must be 2 or 3"
        )


def get_sub_block_indices(
    block: str,
    sub_block: tuple[int, int],
    mode_indices: list[int],
) -> tuple[int, int]:
    """Get the S matrix indices as slice objects from information about the
    block"""

    second_half = 2 * len(mode_indices)

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

    i, j = sub_block
    i_sequence = mode_indices.index(i)
    j_sequence = mode_indices.index(j)

    row = row + 2 * j_sequence
    col = col + 2 * i_sequence

    row_slice = slice(row, row + 2)
    col_slice = slice(col, col + 2)

    return (row_slice, col_slice)


def get_sub_block(
    matrix: np.ndarray | cp.ndarray,
    block: str,
    sub_block: tuple[int, int],
    mode_indices: list[int],
):
    """Get the sub-block from a S matrix determined by the block and "
    sub_block variables"""
    sub_block_indices = get_sub_block_indices(block, sub_block, mode_indices)
    sb = matrix[sub_block_indices]
    return sb


# def keep_block_antidiagonal(matrix: np.ndarray) -> np.ndarray:
#     A = matrix.copy()
#     N = A.shape[0] // 2
#     for i in range(N):
#         for j in range(N):
#             if j != N - 1 - i:
#                 A[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = 0
#     return A


def get_sub_block_antidiagonal(A):
    M, H, W = A.shape
    N = H // 2

    out = np.zeros_like(A)

    for m in range(M):
        for i in range(N):
            j = N - 1 - i
            out[m, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = A[
                m, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2
            ]

    return out


def get_sub_block_from_indices(
    row_index: int,
    col_index: int,
    is_reciprocal: bool,
    num_propagating: int,
    num_evanescent: int = 0,
    wave_block: str = "pp",
) -> tuple[int, int]:
    """Get the sub block from a given pair of indices"""

    second_half = 2 * num_propagating

    if row_index < second_half:
        if col_index < second_half:
            block = "r"
            reduced_row_index = row_index // 2
            reduced_col_index = col_index // 2
        else:
            block = "t2"
            reduced_row_index = row_index // 2
            reduced_col_index = (col_index - second_half) // 2
    else:
        if col_index < second_half:
            block = "t"
            reduced_row_index = (row_index - second_half) // 2
            reduced_col_index = col_index // 2
        else:
            block = "r2"
            reduced_row_index = (row_index - second_half) // 2
            reduced_col_index = (col_index - second_half) // 2

    if is_reciprocal:
        reduced_row_index = reduced_row_index - num_propagating // 2
        reduced_col_index = reduced_col_index - num_propagating // 2

    return block, (reduced_col_index, reduced_row_index)


def get_cov_starting_index(
    block: str,
    sub_block: tuple[int, int],
    mode_indices: list[int],
) -> int:
    """Get the starting index within the covariance matrix from information
    about the block for which the statistics will correspond to"""
    num_propagating = len(mode_indices)

    index = 0
    match block:
        case "r":
            index += 0
        case "t":
            index += num_propagating**2
        case "t2":
            index += 2 * num_propagating**2
        case "r2":
            index += 3 * num_propagating**2

    i, j = sub_block
    i_sequence = mode_indices.index(i)
    j_sequence = mode_indices.index(j)

    partial = j_sequence * num_propagating + i_sequence

    index += partial
    index *= 4

    return index


def get_cov_sub_block_indices(
    blocks: str,
    sub_blocks: tuple[int, int, int, int],
    mode_indices: list[int],
) -> tuple[int, int]:
    """Get the cov matrix indices as slice objects from information about the
    block"""

    block_ij, block_uv = blocks.split(",")
    sub_block_ij = sub_blocks[:2]
    sub_block_uv = sub_blocks[2:]

    row_index = get_cov_starting_index(block_ij, sub_block_ij, mode_indices)
    col_index = get_cov_starting_index(block_uv, sub_block_uv, mode_indices)
    row_slice = slice(row_index, row_index + 4)
    col_slice = slice(col_index, col_index + 4)

    return (row_slice, col_slice)


def get_cov_block_indices(
    blocks: str,
    num_propagating: int,
    num_evanescent: int = 0,
    wave_block: str = "pp",
) -> tuple[int, int]:
    """Get the cov matrix indices as slice objects from information about the
    block"""

    block_ij, block_uv = blocks.split(",")

    start_row_index = get_cov_starting_index(
        block_ij, (0, 0), False, num_propagating
    )
    end_row_index = get_cov_starting_index(
        block_ij,
        (num_propagating - 1, num_propagating - 1),
        False,
        num_propagating,
    )

    start_col_index = get_cov_starting_index(
        block_uv, (0, 0), False, num_propagating
    )
    end_col_index = get_cov_starting_index(
        block_uv,
        (num_propagating - 1, num_propagating - 1),
        False,
        num_propagating,
    )

    row_slice = slice(start_row_index, end_row_index + 4)
    col_slice = slice(start_col_index, end_col_index + 4)

    return (row_slice, col_slice)


def get_cov_block(
    cov: np.ndarray,
    blocks: str,
    is_reciprocal: bool = True,
):
    """Get the sub-block from a cov matrix determined by the block and "
    sub_block variables"""
    num_propagating = int(np.sqrt(cov.shape[0] / 16))
    cov_block_indices = get_cov_block_indices(blocks, num_propagating)
    cov_block = cov[cov_block_indices]
    return cov_block


def get_cov_sub_block(
    cov: np.ndarray,
    blocks: str,
    sub_blocks: tuple[int, int, int, int],
    is_reciprocal: bool = True,
):
    """Get the sub-block from a cov matrix determined by the block and "
    sub_block variables"""
    num_propagating = int(np.sqrt(cov.shape[0] / 16))
    cov_sub_block_indices = get_cov_sub_block_indices(
        blocks, sub_blocks, is_reciprocal, num_propagating
    )
    cov_sub_block = cov[cov_sub_block_indices]
    return cov_sub_block


def get_cov_sub_block_from_indices(
    row_index: int,
    col_index: int,
    is_reciprocal: bool,
    num_propagating: int,
    num_evanescent: int = 0,
    wave_block: str = "pp",
) -> tuple[int, int]:
    """Get the cov matrix sub block given indices"""

    first = 1 * 4 * num_propagating**2
    second = 2 * 4 * num_propagating**2
    third = 3 * 4 * num_propagating**2

    if row_index < first:
        block_ij = "r"
        reduced_row_index = row_index // 4
    elif row_index < second:
        block_ij = "t"
        reduced_row_index = (row_index - first) // 4
    elif row_index < third:
        block_ij = "t2"
        reduced_row_index = (row_index - second) // 4
    else:
        block_ij = "r2"
        reduced_row_index = (row_index - third) // 4

    if col_index < first:
        block_uv = "r"
        reduced_col_index = col_index // 4
    elif col_index < second:
        block_uv = "t"
        reduced_col_index = (col_index - first) // 4
    elif col_index < third:
        block_uv = "t2"
        reduced_col_index = (col_index - second) // 4
    else:
        block_uv = "r2"
        reduced_col_index = (col_index - third) // 4

    j, i = divmod(reduced_row_index, num_propagating)
    v, u = divmod(reduced_col_index, num_propagating)

    if is_reciprocal:
        i = i - num_propagating // 2
        j = j - num_propagating // 2
        u = u - num_propagating // 2
        v = v - num_propagating // 2
    return f"{block_ij},{block_uv}", (i, j, u, v)


def r_sym(matrix: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    """Returns the 'reciprocal operator' applied to a matrix as defined in

    https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.013129
    """
    xp = cp.get_array_module(matrix)

    if matrix.ndim == 2:
        # Single matrix
        M, N = matrix.shape
        b = xp.ones((M, N), dtype=matrix.dtype)
        b[1::2, ::2] = -1.0
        b[::2, 1::2] = -1.0
        return matrix.T * b
    elif matrix.ndim == 3:
        # Batch of matrices (M, N, N)
        _, M, N = matrix.shape
        b = xp.ones((M, N), dtype=matrix.dtype)
        b[1::2, ::2] = -1.0
        b[::2, 1::2] = -1.0
        return matrix.transpose(0, 2, 1) * b[None, :, :]
    else:
        raise ValueError("matrix must be (N,N) or (M,N,N)")


def get_reciprocal_sub_block_indices(
    block: str, sub_block: tuple[int, int], mode_indices: list[int]
) -> tuple[int, int]:
    """Given a sub block of S, find the location of its reciprocal partner."""

    reciprocal_blocks = {"r": "r", "r2": "r2", "t": "t2", "t2": "t"}
    reciprocal_block = reciprocal_blocks[block]
    reciprocal_sub_block = (-sub_block[1], -sub_block[0])

    return get_sub_block_indices(
        reciprocal_block, reciprocal_sub_block, mode_indices
    )


def get_closest_unitary_approximation(
    matrix: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Get the closest unitary matrix to a given matrix.

    This is achieved using the svd and forcing all singular values to be 1"""
    xp = cp.get_array_module(matrix)
    u, _, vh = xp.linalg.svd(matrix)
    return u @ vh


def get_exchange_matrix(
    size: int, use_cupy: bool = False
) -> np.ndarray | cp.ndarray:
    """Return the exchange matrix with 1s on the anti diagonal and zeros
    elsewhere"""
    xp = cp if use_cupy else np
    return xp.fliplr(xp.eye(size, dtype=xp.complex128))


def get_pauli_x(use_cupy: bool = False) -> np.ndarray | cp.ndarray:
    """Return sigma_y. See thesis"""
    return get_exchange_matrix(2, use_cupy)


def get_pauli_z(use_cupy: bool = False) -> np.ndarray | cp.ndarray:
    """Return sigma_y. See thesis"""
    xp = cp if use_cupy else np
    return xp.array([[1.0, 0.0], [0.0, -1.0]])


def get_pauli_y(use_cupy: bool = False) -> np.ndarray | cp.ndarray:
    """Return sigma_y. See thesis"""
    xp = cp if use_cupy else np
    return 1j * xp.array([[0.0, -1.0], [1.0, 0.0]])


def get_S_block_reciprocity_matrix(
    size: int, use_cupy: bool = False
) -> np.ndarray | cp.ndarray:
    """Return the sigma_p matrix defined in Eq. (3.153) of Niall's thesis

    size should be the size of a block of S (e.g. r or t)"""
    xp = cp if use_cupy else np
    num_modes = int(size // 2)
    return xp.kron(get_exchange_matrix(num_modes, use_cupy), xp.identity(2))


def get_S_reciprocity_matrix(
    size: int, use_cupy: bool = False
) -> np.ndarray | cp.ndarray:
    """Return w as defined in Niall's thesis such that S = w^* S^T w when the
    medium satisfies reciprocity.

    size should be the size of the scattering matrix"""
    xp = cp if use_cupy else np
    num_modes = int(size // 4)
    identity = xp.eye(2, dtype=np.complex128)
    exchange = get_exchange_matrix(num_modes, use_cupy)
    sigma_z = get_pauli_z(use_cupy)
    product = xp.kron(identity, xp.kron(exchange, sigma_z))
    return product


def get_M_energy_matrix(
    size: int, use_cupy: bool = False
) -> np.ndarray | cp.ndarray:
    """Return Omega as defined in Niall's thesis such that

    M^dag Omega M = Omega

    expresses energy conservation"""
    xp = cp if use_cupy else np
    identity_size = int(size // 2)
    identity = xp.identity(identity_size)
    sigma_z = get_pauli_z(use_cupy)
    Omega = xp.kron(sigma_z, identity)
    return Omega


def get_M_reciprocity_matrix(
    size: int, use_cupy: bool = False
) -> np.ndarray | cp.ndarray:
    """Return eta as defined in Niall's thesis such that M = eta M^* eta when
    the medium satisfies reciprocity.

    size should be the size of the transfer matrix"""
    xp = cp if use_cupy else np
    num_modes = int(size // 4)
    exchange = get_exchange_matrix(num_modes, use_cupy)
    sigma_x = get_pauli_x(use_cupy)
    sigma_z = get_pauli_z(use_cupy)
    product = xp.kron(sigma_x, xp.kron(exchange, sigma_z))
    return product


def get_M_from_S(S: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    """Given S of shape M x n x n, where M is the number of matrices, get an
    M x n x n array of transfer matrices"""
    xp = cp.get_array_module(S)

    r = get_block(S, "r")
    r2 = get_block(S, "r2")
    t = get_block(S, "t")
    t2 = get_block(S, "t2")

    t2_inv = xp.linalg.inv(t2)
    a = t - r2 @ t2_inv @ r
    b = r2 @ t2_inv
    c = -t2_inv @ r
    d = t2_inv

    if S.ndim == 3:
        axis = 2
    elif S.ndim == 2:
        axis = 1

    top = xp.concatenate((a, b), axis=axis)
    bottom = xp.concatenate((c, d), axis=axis)
    return xp.concatenate((top, bottom), axis=axis - 1)


def get_S_from_M(M: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    """Given M of shape M x n x n, where M is the number of matrices, get an
    M x n x n array of scattering matrices"""
    xp = cp.get_array_module(M)

    a = get_block(M, "a")
    b = get_block(M, "b")
    c = get_block(M, "c")
    d = get_block(M, "d")

    d_inv = xp.linalg.inv(d)
    r = -d_inv @ c
    t = a - b @ d_inv @ c
    t2 = d_inv
    r2 = b @ d_inv

    if M.ndim == 3:
        axis = 2
    elif M.ndim == 2:
        axis = 1

    top = xp.concatenate((r, t2), axis=axis)
    bottom = xp.concatenate((t, r2), axis=axis)
    return xp.concatenate((top, bottom), axis=axis - 1)


def S_product(
    S_1: np.ndarray | cp.ndarray, S_2: np.ndarray | cp.ndarray
) -> np.ndarray | cp.ndarray:
    """Compute the scattering matrix for the medium combined of two partial
    media described by S1 and S2

    left space S1 S2 right space

    This corresponds to the transfer matrix product M2 * M1"""
    xp = cp.get_array_module(S_1)
    use_cupy = xp == cp

    num_modes = int(len(S_1) // 4)

    sigma_z = get_pauli_z(use_cupy)
    r_1 = get_block(S_1, "r")
    r2_1 = get_block(S_1, "r2")
    t_1 = get_block(S_1, "t")
    t2_1 = get_block(S_1, "t2")
    r_2 = get_block(S_2, "r")
    r2_2 = get_block(S_2, "r2")
    t_2 = get_block(S_2, "t")
    t2_2 = get_block(S_2, "t2")
    J = get_exchange_matrix(num_modes, use_cupy)

    Q = xp.linalg.inv(xp.identity(len(r_1), dtype=xp.complex128) - r2_1 @ r_2)

    partial_one = t_2 @ Q @ t_1
    partial_two = xp.kron(J, sigma_z)

    r = r_1 + t2_1 @ r_2 @ Q @ t_1
    r2 = r2_2 + t_2 @ Q @ r2_1 @ t2_2
    t = partial_one
    t2 = partial_two @ partial_one.T @ partial_two

    top = xp.concatenate((r, t2), axis=1)
    bottom = xp.concatenate((t, r2), axis=1)
    S = xp.concatenate((top, bottom), axis=0)
    return S


def block_cholesky(A: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a 2x2 block matrix.

    A is assumed to be symmetric positive definite. Method adapted from
    https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices
    """
    n = A.shape[0]
    k = n // 2

    A11 = A[:k, :k]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    L11 = np.linalg.cholesky(A11)
    L21 = np.conj(np.linalg.solve(L11, np.conj(A21).T)).T
    S = A22 - L21 @ np.conj(L21).T
    L22 = np.linalg.cholesky(S)

    L_top = np.hstack([L11, np.zeros((k, n - k))])
    L_bottom = np.hstack([L21, L22])
    L = np.vstack([L_top, L_bottom])

    return L


def sparse_cholesky(A: scipy.sparse.csc_matrix) -> scipy.sparse.csc_matrix:
    """Get the cholesky decomposition of a sparse matrix using sksparse"""
    return sksparse.cholmod.cholesky(A, ordering_method="natural").L()


def block_cholesky_sparse(
    A: scipy.sparse.csc_matrix,
) -> scipy.sparse.csc_matrix:
    """
    Compute the Cholesky decomposition of a 2x2 block sparse matrix.

    A: symmetric/Hermitian positive definite sparse matrix (csc_matrix)
    Method based on:
    https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices

    """
    n = A.shape[0]
    k = n // 2

    # Split blocks
    A11 = A[:k, :k]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    # Sparse calculations
    A11_factor = sksparse.cholmod.cholesky(A11)
    L11 = A11_factor.L()
    X = A11_factor.solve_L(A21.conj().T, use_LDLt_decomposition=False).conj().T
    S = A22 - X @ X.conj().T
    Ls = sksparse.cholmod.cholesky(S).L()

    L = scipy.sparse.bmat([[L11, None], [X, Ls]], format="csc")
    return L


def block_cholesky_sparse_recursive(
    A: scipy.sparse.csc_matrix, max_depth: int = 1, depth: int = 0
) -> scipy.sparse.csc_matrix:
    """
    Recursively compute the Cholesky decomposition of a sparse
    symmetric/Hermitian matrix using block_cholesky_sparse.

    Parameters:
        A : csc_matrix
            Symmetric/Hermitian positive definite sparse matrix
        max_depth : int
            Maximum recursion depth
        depth : int
            Current recursion depth (internal use)

    Returns:
        L : csc_matrix
            Lower-triangular Cholesky factor of A
    """

    if depth >= max_depth:
        return A

    n = A.shape[0]
    k = n // 2

    # Split blocks
    A11 = A[:k, :k]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    # Sparse calculations
    A11_factor = sksparse.cholmod.cholesky(A11)
    L11 = A11_factor.L()
    X = A11_factor.solve_L(A21.conj().T, use_LDLt_decomposition=False).conj().T
    S = A22 - X @ X.conj().T
    Ls = sksparse.cholmod.cholesky(S).L()

    L = scipy.sparse.bmat([[L11, None], [X, Ls]], format="csc")

    return L


def get_real_covariance_matrix(
    covariance: scipy.sparse.spmatrix,
    pseudo_covariance: scipy.sparse.spmatrix | None = None,
) -> scipy.sparse.spmatrix:
    """Construct a real-valued covariance matrix based on a supplied covariance
    and pseudo-covariance matrix"""
    if pseudo_covariance is None:
        pseudo_covariance = scipy.sparse.csr_matrix(
            covariance.shape, dtype=covariance.dtype
        )

    return 0.5 * scipy.sparse.bmat(
        [
            [
                covariance.real + pseudo_covariance.real,
                -covariance.imag + pseudo_covariance.imag,
            ],
            [
                covariance.imag + pseudo_covariance.imag,
                covariance.real - pseudo_covariance.imag,
            ],
        ]
    )


def get_cholesky_decomposition(
    covariance_matrix: scipy.sparse.spmatrix,
    first_power: int = -15,
    final_power: int = 0,
) -> scipy.sparse.spmatrix:
    """Calculate the cholesky decomposition of positive semi-definite sparse
    matrix correcting for small negative eigenvalues"""
    size_of_covariance_matrix = np.shape(covariance_matrix)[0]
    covariance_matrix = scipy.sparse.csc_array(covariance_matrix)

    for i in range(first_power, final_power, 1):
        try:
            covariance_matrix_altered = (
                covariance_matrix
                + scipy.sparse.identity(size_of_covariance_matrix) * 10 ** (i)
            )
            chol = sksparse.cholmod.cholesky(
                covariance_matrix_altered, ordering_method="natural"
            ).L()
            break
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            pass

    print(f"POWER USED FOR CHOL: 10^{i}")
    return chol
