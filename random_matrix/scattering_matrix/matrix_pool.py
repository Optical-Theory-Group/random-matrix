import numpy as np
from random_matrix.scattering_matrix import sampler
from random_matrix.input_statistics import medium_parameters, medium_statistics
from pathlib import Path
import warnings
import scipy
from random_matrix.modes import mode_grid
from random_matrix.utils import matrix_utils
from typing import Callable
import random
from tqdm import tqdm
import cupy as cp
import cupyx.scipy.sparse as cpsparse


class MatrixPool:
    def __init__(
        self,
        simulation_name: str,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        mode_grid: mode_grid.ModeGrid | None = None,
        mean_S: np.ndarray | None = None,
        chol: np.ndarray | None = None,
        parent_data_dir: str | Path | None = None,
    ) -> None:
        self.single_pool_S = None
        self.multi_pool_S = None
        self.single_pool_M = None
        self.multi_pool_M = None

        self.simulation_name = simulation_name
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics

        # Default to the current working directory if none is given
        self.parent_data_path = (
            Path(parent_data_dir) if parent_data_dir else Path.cwd() / "data"
        )
        if not parent_data_dir:
            warnings.warn(
                f"No parent_data_dir provided. Defaulting to current working"
                f" directory: {Path.cwd() / 'data'}.",
                stacklevel=2,  # Makes the warning point to the caller, not here
            )
        self._validate_parent_data_path()

        self.simulation_path = self.parent_data_path / Path(simulation_name)
        mode_grid_path = self.simulation_path / "mode_grid.pkl"
        mean_S_path = self.simulation_path / "mean_S.npy"
        chol_path = self.simulation_path / "chol.npz"

        # Load from disk
        if mode_grid is None:
            mode_grid_exists = mode_grid_path.exists()
            if not mode_grid_exists:
                raise FileNotFoundError(
                    f"File at {str(mode_grid_path)} was not found."
                )
            self.mode_grid = np.load(mode_grid_path)
        else:
            self.mode_grid = mode_grid

        # Load from disk
        if mean_S is None or chol is None:
            statistics_exist = mean_S_path.exists() and chol_path.exists()
            if not statistics_exist:
                raise FileNotFoundError(
                    f"Files at {str(mean_S_path)} and "
                    f"{str(chol_path)} were not found."
                )
            self.mean_S = np.load(mean_S_path)
            self.chol = scipy.sparse.load_npz(chol_path)
        else:
            self.mean_S = mean_S
            self.chol = chol

    def _validate_parent_data_path(self) -> None:
        """Check that the parent directory exists. If not, raise an error"""
        if not self.parent_data_path.exists():
            raise FileNotFoundError(
                f"Directory {str(self.parent_data_path)} does not exist. "
                f"Please update the `parent_data_dir` variable passed to the "
                f"pool constructor."
            )

    @property
    def S(self) -> np.ndarray | cp.ndarray:
        """Alias for the single pool S matrices"""
        return self.single_pool_S

    @property
    def M(self) -> np.ndarray | cp.ndarray:
        """Alias for the single pool M matrices"""
        return self.single_pool_M

    @property
    def matrix_size(self) -> int:
        """Get the size of the M or S matrices involved"""
        return len(self.mean_S)

    @property
    def half_matrix_size(self) -> int:
        """Get the size of the t,r etc. matrices involved"""
        return int(len(self.mean_S) // 2)

    @property
    def pool_size(self) -> int:
        """Get the size of the pool"""
        return len(self.single_pool_S)

    @property
    def single_pool_S_array_module(self):
        """Get the array module used for the matrices in single_pool_S"""
        return cp.get_array_module(self.S[0])

    @staticmethod
    def _reorder_block(
        elements: np.ndarray | cp.ndarray,
    ) -> np.ndarray | cp.ndarray:
        xp = cp.get_array_module(elements)

        length_of_elements, num_matrices = xp.shape(elements)
        size_of_block = int(xp.sqrt(len(elements)))

        even_row_indices = xp.sort(
            xp.concatenate(
                (
                    xp.arange(0, length_of_elements, 4),
                    xp.arange(1, length_of_elements, 4),
                )
            )
        )
        odd_row_indices = xp.sort(
            xp.concatenate(
                (
                    xp.arange(2, length_of_elements, 4),
                    xp.arange(3, length_of_elements, 4),
                )
            )
        )

        even_rows = elements[even_row_indices].reshape(
            int(size_of_block / 2), size_of_block, num_matrices
        )
        odd_rows = elements[odd_row_indices].reshape(
            int(size_of_block / 2), size_of_block, num_matrices
        )
        final = xp.empty(
            (
                even_rows.shape[0] + odd_rows.shape[0],
                even_rows.shape[1],
                num_matrices,
            ),
            dtype=xp.complex128,
        )
        final[::2, :, :] = even_rows
        final[1::2, :, :] = odd_rows
        return final

    def S_sampler(
        self,
        num_matrices: int = 1,
        symmetrize: bool = True,
        use_cupy: bool = False,
    ) -> np.ndarray | cp.ndarray:

        xp = cp if use_cupy else np

        if use_cupy:
            mean_S = cp.asarray(self.mean_S)
            chol = cpsparse.csr_matrix(self.chol)
        else:
            mean_S = self.mean_S
            chol = self.chol

        size_of_S, _ = xp.shape(mean_S)
        size_of_block = int(size_of_S / 2)
        num_random_numbers, _ = xp.shape(chol)

        # Generate random numbers for the matrices
        random_numbers = xp.random.randn(num_random_numbers, num_matrices)
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
            + 1j
            * imags[int(num_random_numbers / 4) : int(num_random_numbers / 2)]
        )
        t2 = (
            reals[
                int(num_random_numbers / 2) : int(num_random_numbers * 3 / 4)
            ]
            + 1j
            * imags[
                int(num_random_numbers / 2) : int(num_random_numbers * 3 / 4)
            ]
        )
        r2 = (
            reals[int(num_random_numbers * 3 / 4) : num_random_numbers]
            + 1j * imags[int(num_random_numbers * 3 / 4) : num_random_numbers]
        )

        # Reorder the randomly generated numbers into the correct shapes
        r = self._reorder_block(r)
        t = self._reorder_block(t)
        t2 = self._reorder_block(t2)
        r2 = self._reorder_block(r2)

        # Add identity to transmission matrices
        identity = xp.identity(size_of_block)
        t = t + identity[:, :, xp.newaxis]
        t2 = t2 + identity[:, :, xp.newaxis]

        top = xp.hstack([r, t2])
        bottom = xp.hstack([t, r2])
        whole = xp.vstack([top, bottom])

        # Add the mean matrix to each instance
        whole = whole - mean_S[:, :, xp.newaxis]
        output = xp.transpose(whole, (2, 0, 1))

        if symmetrize:
            output = matrix_utils.get_closest_unitary_approximation(output)
        return output

    def populate_single_pool(
        self,
        num_matrices: int = 1,
        save_matrices: bool = True,
        symmetrize: bool = True,
        populate_single_pool_M: bool = False,
        use_cupy: bool = False,
    ) -> None | np.ndarray:
        """Populate the single pools with matrices

        if save_matrices is false, the matrices are returned rather than saved
        in the pool

        symmetrize determines wheter the matrices should be made unitary or not

        If populate_M_single_pool is true, save the transfer matrices alongside
        the S matrices. This might need to be false if memory is a concern
        """
        S_matrices = self.S_sampler(num_matrices, symmetrize, use_cupy)

        # Exit here if matrices are not to be saved
        if not save_matrices:
            return S_matrices

        # Save the pool
        self.single_pool_S = S_matrices

        if populate_single_pool_M:
            self.single_pool_M = matrix_utils.get_M_from_S(S_matrices)

    def cascade(
        self,
        num_samples: int,
        analysis_points: list[int] | list[np.float64],
        analysis_functions: dict[str, Callable],
        use_transfer_matrices: bool = False,
    ):
        """Method for doing small-scale cascades. Ideal for quick tests."""
        xp = self.single_pool_S_array_module

        # Check if analysis points has ints or floats
        if isinstance(analysis_points[0], int):
            pass
        else:
            # Convert to appropriate ints based on the thickness of the
            # elementary slab
            pass
        max_iteration = analysis_points[-1]

        # Initialize working matrix array
        if use_transfer_matrices:
            M_empty = xp.identity(self.matrix_size, dtype=xp.complex128)
            working_matrices = xp.tile(M_empty, (num_samples, 1, 1))
        else:
            half_identity = xp.identity(
                self.half_matrix_size, dtype=xp.complex128
            )
            half_zeros = xp.zeros(
                (self.half_matrix_size, self.half_matrix_size), xp.complex128
            )
            top = xp.concatenate((half_zeros, half_identity), axis=1)
            bottom = xp.concatenate((half_identity, half_zeros), axis=1)
            S_empty = xp.concatenate((top, bottom), axis=0)
            working_matrices = xp.tile(S_empty, (num_samples, 1, 1))

        # Initialize data collection dictionary
        data = {key: [] for key in analysis_functions}

        # Main cascade loop
        for i in tqdm(range(1, max_iteration + 1)):

            # Check if need to swtich to using scattering matrices
            # TO BE IMPLEMENTED

            # Do matrix products
            for j in range(num_samples):
                random_matrix_index = random.randrange(0, self.pool_size)

                if use_transfer_matrices:
                    working_matrices[j] = (
                        self.single_pool_M[random_matrix_index]
                        @ working_matrices[j]
                    )
                else:
                    working_matrices[j] = matrix_utils.S_product(
                        working_matrices[j],
                        self.single_pool_S[random_matrix_index],
                    )

            # Do the analysis
            if i in analysis_points:
                print(f"Collecting data at index {i}")
                for key, analysis_function in analysis_functions.items():
                    new_output = analysis_function(working_matrices)
                    data[key].append(new_output)

        return data

    def cascade_hdf5(
        self,
        cascade_name: str,
        num_samples: int,
        batch_size: int,
        analysis_points: list[int] | list[np.float64],
        analysis_functions: dict[str, Callable],
        use_transfer_matrices: bool = False,
    ):
        """Method for more intense data runs. Data is automatically saved"""
        xp = self.single_pool_S_array_module

        # Check for cascade data files
        cascade_data_parent_path = self.simulation_path / Path("cascade_data")
        cascade_data_path = cascade_data_parent_path / Path(cascade_name)
        if not cascade_data_parent_path.exists():
            cascade_data_parent_path.mkdir()
            print(
                f"Creating parent directory for cascade data at " 
                f"{cascade_data_parent_path}"
            )
        if not cascade_data_path.exists():
            cascade_data_path.mkdir()
            print(
                f"Creating data directory for cascade data at " 
                f"{cascade_data_path}"
            )




        # Check if analysis points has ints or floats
        if isinstance(analysis_points[0], int):
            pass
        else:
            # Convert to appropriate ints based on the thickness of the
            # elementary slab
            pass
        max_iteration = analysis_points[-1]

        # Initialize working matrix array
        if use_transfer_matrices:
            M_empty = xp.identity(self.matrix_size, dtype=xp.complex128)
            working_matrices = xp.tile(M_empty, (num_samples, 1, 1))
        else:
            half_identity = xp.identity(
                self.half_matrix_size, dtype=xp.complex128
            )
            half_zeros = xp.zeros(
                (self.half_matrix_size, self.half_matrix_size), xp.complex128
            )
            top = xp.concatenate((half_zeros, half_identity), axis=1)
            bottom = xp.concatenate((half_identity, half_zeros), axis=1)
            S_empty = xp.concatenate((top, bottom), axis=0)
            working_matrices = xp.tile(S_empty, (num_samples, 1, 1))

        # Initialize data collection dictionary
        data = {key: [] for key in analysis_functions}

        # Main cascade loop
        for i in tqdm(range(1, max_iteration + 1)):

            # Check if need to swtich to using scattering matrices
            # TO BE IMPLEMENTED

            # Do matrix products
            for j in range(num_samples):
                random_matrix_index = random.randrange(0, self.pool_size)

                if use_transfer_matrices:
                    working_matrices[j] = (
                        self.single_pool_M[random_matrix_index]
                        @ working_matrices[j]
                    )
                else:
                    working_matrices[j] = matrix_utils.S_product(
                        working_matrices[j],
                        self.single_pool_S[random_matrix_index],
                    )

            # Do the analysis
            if i in analysis_points:
                print(f"Collecting data at index {i}")
                for key, analysis_function in analysis_functions.items():
                    new_output = analysis_function(working_matrices)
                    data[key].append(new_output)

        return data
