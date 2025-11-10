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
import h5py


class MatrixPoolManager:
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
        if not self.parent_data_path.exists():
            raise FileNotFoundError(
                f"Directory {str(self.parent_data_path)} does not exist. "
                f"Please update the `parent_data_dir` variable passed to the "
                f"pool constructor."
            )

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

        self.single_pool_S = None
        self.multi_pool_S = None
        self.single_pool_M = None
        self.multi_pool_M = None

    def get_pool_data(
        self, is_transfer_matrix: bool, is_multi_pool: bool
    ) -> dict:
        """Packaged data used in various places"""
        pool_map = {
            (False, False): (
                "single_pool_S",
                self.single_pool_S,
            ),
            (True, False): (
                "single_pool_M",
                self.single_pool_M,
            ),
            (False, True): (
                "multi_pool_S",
                self.multi_pool_S,
            ),
            (True, True): (
                "multi_pool_M",
                self.multi_pool_M,
            ),
        }
        return pool_map[(is_transfer_matrix, is_multi_pool)]

    def get_pool(
        self, is_transfer_matrix: bool, is_multi_pool: bool
    ) -> np.ndarray:
        _, pool = self.get_pool_data(is_transfer_matrix, is_multi_pool)
        return pool

    def pool_exists(
        self, is_transfer_matrix: bool = False, is_multi_pool: bool = False
    ) -> bool:
        """Check whether the given pool exists or not"""
        pool = self.get_pool(is_transfer_matrix, is_multi_pool)
        return pool is not None

    def save_pool(
        self, is_transfer_matrix: bool = False, is_multi_pool: bool = False
    ) -> None:
        """Save a given pool to memory for reuse in the future"""
        pool_dir_path = self.simulation_path / "pools"
        if not pool_dir_path.exists():
            pool_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {pool_dir_path}")

        # HDF5 file path
        h5_file_path = pool_dir_path / "pools.h5"

        # Determine dataset name and pool based on boolean args
        dataset_name, pool = self.get_pool_data(
            is_transfer_matrix, is_multi_pool
        )

        # Check pool exists
        if pool is None:
            raise ValueError(
                f"{dataset_name} does not exist and thus cannot be saved."
                "Populate it first!"
            )

        with h5py.File(h5_file_path, "a") as f:
            if dataset_name in f:
                del f[dataset_name]

            f.create_dataset(dataset_name, data=pool)

    def load_pool(
        self, is_transfer_matrix: bool = False, is_multi_pool: bool = False
    ) -> None:
        """Load the given pool from memory"""
        pool_dir_path = self.simulation_path / "pools"
        h5_file_path = pool_dir_path / "pools.h5"

        # Determine dataset name and target attribute based on boolean args
        attr_name, _ = self.get_pool_data(is_transfer_matrix, is_multi_pool)

        if not h5_file_path.exists():
            raise FileNotFoundError(
                f"HDF5 file does not exist: {h5_file_path}"
            )

        with h5py.File(h5_file_path, "r") as f:
            if attr_name not in f:
                raise ValueError(
                    f"Dataset '{attr_name}' not found in HDF5 file"
                )
            setattr(self, f"{attr_name}", f[attr_name][:])

    # Convenience methods
    def save_single_pool_S(self) -> None:
        self.save_pool(is_transfer_matrix=False, is_multi_pool=False)

    def load_single_pool_S(self) -> None:
        self.load_pool(is_transfer_matrix=False, is_multi_pool=False)

    def save_single_pool_M(self) -> None:
        self.save_pool(is_transfer_matrix=True, is_multi_pool=False)

    def load_single_pool_M(self) -> None:
        self.load_pool(is_transfer_matrix=True, is_multi_pool=False)

    def save_multi_pool_S(self) -> None:
        self.save_pool(is_transfer_matrix=False, is_multi_pool=True)

    def load_multi_pool_S(self) -> None:
        self.load_pool(is_transfer_matrix=False, is_multi_pool=True)

    def save_multi_pool_M(self) -> None:
        self.save_pool(is_transfer_matrix=True, is_multi_pool=True)

    def load_multi_pool_M(self) -> None:
        self.load_pool(is_transfer_matrix=True, is_multi_pool=True)

    def clear_single_pools(self) -> None:
        self.single_pool_S = None
        self.single_pool_M = None

    # -------------------------------------------------------------------------
    # Pool propreties
    # -------------------------------------------------------------------------

    @property
    def S(self) -> np.ndarray | cp.ndarray:
        """Alias for the single pool S matrices"""
        return self.single_pool_S

    @property
    def M(self) -> np.ndarray | cp.ndarray:
        """Alias for the single pool M matrices"""
        return self.single_pool_M

    @property
    def single_pool_S_exists(self) -> bool:
        return self.pool_exists(False, False)

    @property
    def single_pool_M_exists(self) -> bool:
        return self.pool_exists(True, False)

    @property
    def single_pool_exists(self) -> bool:
        return self.single_pool_S_exists or self.single_pool_M_exists

    @property
    def multi_pool_S_exists(self) -> bool:
        return self.pool_exists(False, True)

    @property
    def multi_pool_M_exists(self) -> bool:
        return self.pool_exists(True, True)

    @property
    def multi_pool_exists(self) -> bool:
        return self.multi_pool_S_exists or self.multi_pool_M_exists

    @property
    def single_pool_S_size(self) -> int:
        return 0 if self.single_pool_S is None else len(self.single_pool_S)

    @property
    def single_pool_M_size(self) -> int:
        return 0 if self.single_pool_M is None else len(self.single_pool_M)

    @property
    def single_pool_S_array_module(self):
        """Get the array module used for the matrices in single_pool_S"""
        if not self.single_pool_S_exists:
            raise ValueError(
                "single_pool_S does not exist. Generate it first!"
            )
        return cp.get_array_module(self.single_pool_S[0])

    # -------------------------------------------------------------------------
    # Other useful matrix properties
    # -------------------------------------------------------------------------

    @property
    def matrix_size(self) -> int:
        """Get the size of the M or S matrices involved"""
        return len(self.mean_S)

    @property
    def matrix_shape(self) -> int:
        return self.mean_S.shape

    @property
    def half_matrix_size(self) -> int:
        """Get the size of the t,r etc. matrices involved"""
        return int(len(self.mean_S) // 2)

    def get_initialized_S_array(
        self, num_matrices: int, use_cupy: bool = False
    ) -> np.ndarray:
        """Return a collcetion of S matrices for a non-scattering medium"""
        xp = cp if use_cupy else np
        half_identity = xp.identity(self.half_matrix_size, dtype=xp.complex128)
        half_zeros = xp.zeros(
            (self.half_matrix_size, self.half_matrix_size), xp.complex128
        )
        top = xp.concatenate((half_zeros, half_identity), axis=1)
        bottom = xp.concatenate((half_identity, half_zeros), axis=1)
        S_empty = xp.concatenate((top, bottom), axis=0)
        return xp.tile(S_empty, (num_matrices, 1, 1))

    def get_initialized_M_array(
        self, num_matrices: int, use_cupy: bool = False
    ) -> np.ndarray:
        """Return a collcetion of M matrices for a non-scattering medium"""
        xp = cp if use_cupy else np
        M_empty = xp.identity(self.matrix_size, dtype=xp.complex128)
        return xp.tile(M_empty, (num_matrices, 1, 1))

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
        modified_random_numbers = chol @ random_numbers
        reals = modified_random_numbers[: int(num_random_numbers / 2)]
        imags = modified_random_numbers[int(num_random_numbers / 2) :]

        # Extract matrix elements from random numbers
        num_random_numbers = int(num_random_numbers / 2)
        r = (
            reals[: int(num_random_numbers / 4)]
            + 1j * imags[: int(num_random_numbers / 4)]
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
            reals[int(num_random_numbers * 3 / 4) :]
            + 1j * imags[int(num_random_numbers * 3 / 4) :]
        )

        # ----------------------------------------------
        # debugging
        # ------------------------------------------

        # Reorder the randomly generated numbers into the correct shapes
        r_mat = self._reorder_block(r)
        t_mat = self._reorder_block(t)
        t2_mat = self._reorder_block(t2)
        r2_mat = self._reorder_block(r2)

        # Add identity to transmission matrices
        identity = xp.identity(size_of_block)
        t_mat = t_mat  # + identity[:, :, xp.newaxis]
        t2_mat = t2_mat  # + identity[:, :, xp.newaxis]

        top = xp.hstack([r_mat, t2_mat])
        bottom = xp.hstack([t_mat, r2_mat])
        whole = xp.vstack([top, bottom])

        # Add the mean matrix to each instance
        whole = whole  # - mean_S[:, :, xp.newaxis]
        output = xp.transpose(whole, (2, 0, 1))

        if symmetrize:
            output = matrix_utils.get_closest_unitary_approximation(output)
        return output

    def populate_single_pool(
        self,
        num_matrices: int = 1,
        symmetrize: bool = True,
        matrix_type: str = "both",
        save_matrices: bool = True,
        use_cupy: bool = False,
    ) -> None:
        """Populate the single pools with matrices

        symmetrize determines wheter the matrices should be made unitary or not

        matrix_type takes three options: "S", "M" or "both"
        """
        S_matrices = self.S_sampler(num_matrices, symmetrize, use_cupy)

        # Save the pool
        if matrix_type in ("S", "both"):
            self.single_pool_S = S_matrices
            if save_matrices:
                self.save_single_pool_S()

        if matrix_type in ("M", "both"):
            self.single_pool_M = matrix_utils.get_M_from_S(S_matrices)
            if save_matrices:
                self.save_single_pool_M()

    def populate_multi_pool(
        self,
        num_matrices: int = 1,
        num_single_pool_matrices: int = 1,
        use_transfer_matrices: bool = True,
        save_matrices: bool = True,
        use_cupy: bool = False,
    ) -> None:
        """Populate the multi pools with matrices

        num_single_pool_matrices is the number of single pool matrices used
        to generate a single multi_pool matrix
        """
        if use_transfer_matrices:
            single_pool = self.single_pool_M
            single_pool_exists = self.single_pool_M_exists
            single_pool_size = self.single_pool_M_size
        else:
            single_pool = self.single_pool_S
            single_pool_exists = self.single_pool_S_exists
            single_pool_size = self.single_pool_S_size

        if not single_pool_exists:
            raise ValueError(
                "The single pool does not exist. Please populate it first (or load it)"
            )

        # Initialize multi pool
        if use_transfer_matrices:
            multi_pool = self.get_initialized_M_array(
                num_matrices, use_cupy=use_cupy
            )
        else:
            multi_pool = self.get_initialized_S_array(
                num_matrices, use_cupy=use_cupy
            )

        for i in tqdm(range(num_matrices)):
            for _ in range(num_single_pool_matrices):
                random_matrix_index = random.randrange(0, single_pool_size)

                if use_transfer_matrices:
                    multi_pool[i] = (
                        single_pool[random_matrix_index] @ multi_pool[i]
                    )
                else:
                    multi_pool[i] = matrix_utils.S_product(
                        multi_pool[i],
                        single_pool[random_matrix_index],
                    )

        if use_transfer_matrices:
            self.multi_pool_M = multi_pool
            if save_matrices:
                self.save_multi_pool_M()
        else:
            self.multi_pool_S = multi_pool
            if save_matrices:
                self.save_multi_pool_S()

    def cascade(
        self,
        num_samples: int,
        analysis_points: list[int] | list[np.float64],
        analysis_functions: dict[str, Callable],
        use_transfer_matrices: bool = False,
        use_multi_pool: bool = False,
    ):
        """Method for doing small-scale cascades. Ideal for quick tests."""
        xp = self.single_pool_S_array_module
        use_cupy = xp == cp

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
            working_matrices = self.get_initialized_M_array(
                num_samples, use_cupy
            )
        else:
            working_matrices = self.get_initialized_S_array(
                num_samples, use_cupy
            )

        # Initialize data collection dictionary
        data = {key: [] for key in analysis_functions}

        pool = self.get_pool(use_transfer_matrices, use_multi_pool)
        pool_exists = pool is not None
        if not pool_exists:
            raise ValueError(
                f"Desired pool does not exist. Please populate it or load it first"
            )
        pool_size = len(pool)

        # Main cascade loop
        for i in tqdm(range(1, max_iteration + 1)):

            # Check if need to swtich to using scattering matrices
            # TO BE IMPLEMENTED

            # Do matrix products
            for j in range(num_samples):
                random_matrix_index = random.randrange(0, pool_size)

                if use_transfer_matrices:
                    working_matrices[j] = (
                        pool[random_matrix_index] @ working_matrices[j]
                    )
                else:
                    working_matrices[j] = matrix_utils.S_product(
                        working_matrices[j],
                        pool[random_matrix_index],
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
        use_multi_pool: bool = False,
    ) -> None:
        """Method for more intense data runs.

        Data is automatically saved in a hdf5 file. It as assumed that all
        analysis functions return numpy arrays for their outputs"""
        xp = self.single_pool_S_array_module
        use_cupy = xp == cp

        # Check if analysis points has ints or floats
        if isinstance(analysis_points[0], int):
            pass
        else:
            # Convert to appropriate ints based on the thickness of the
            # elementary slab
            pass
        max_iteration = analysis_points[-1]
        num_analysis_points = len(analysis_points)

        # Get the random matrix pool
        pool = self.get_pool(use_transfer_matrices, use_multi_pool)
        pool_exists = pool is not None
        if not pool_exists:
            raise ValueError(
                f"Desired pool does not exist. Please populate it or load it first"
            )
        pool_size = len(pool)

        # Prepare data directory
        cascade_dir_path = self.simulation_path / "cascades"
        if not cascade_dir_path.exists():
            cascade_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {cascade_dir_path}")
        h5_file_path = cascade_dir_path / cascade_name

        # Validate batch size
        batch_size = min(batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        slices = [
            slice(i, min(i + batch_size, num_samples))
            for i in range(0, num_samples, batch_size)
        ]
        batch_sizes = [s.stop - s.start for s in slices]
        is_single_batch = num_batches == 1

        # Create test matrix to assess return data shape
        if use_transfer_matrices:
            test_matrix = self.get_initialized_M_array(1)
        else:
            test_matrix = self.get_initialized_S_array(1)

        with h5py.File(h5_file_path, "w") as f:
            for dataset_name, analysis_function in analysis_functions.items():
                # Initialize return data storage
                output = analysis_function(test_matrix)
                output_shape = output.shape
                augmented_shape = (
                    num_analysis_points,
                    num_samples,
                    *output_shape[1:],
                )
                f.create_dataset(
                    dataset_name, shape=augmented_shape, dtype=output.dtype
                )

            # Initialize working matrix storage for multi batch runs
            if not is_single_batch:
                data_shape = (
                    num_samples,
                    *self.matrix_shape,
                )
                f.create_dataset(
                    "working_matrices",
                    shape=data_shape,
                    dtype=np.complex128,
                )

        # Initialize working matrix array
        if is_single_batch:
            # Matrices are directly loaded in RAM
            if use_transfer_matrices:
                working_matrices = self.get_initialized_M_array(
                    num_samples, use_cupy
                )
            else:
                working_matrices = self.get_initialized_S_array(
                    num_samples, use_cupy
                )
        else:
            with h5py.File(h5_file_path, "r+") as f:
                working_matrices = f["working_matrices"]
                for s, bs in zip(slices, batch_sizes):
                    if use_transfer_matrices:
                        working_matrices[s] = self.get_initialized_M_array(
                            bs, use_cupy
                        )
                    else:
                        working_matrices[s] = self.get_initialized_S_array(
                            bs, use_cupy
                        )

        # Main cascade loop
        for i in tqdm(range(1, max_iteration + 1)):

            # Check if need to swtich to using scattering matrices
            # TO BE IMPLEMENTED

            # Do matrix products
            if is_single_batch:
                for j in range(num_samples):
                    random_matrix_index = random.randrange(0, pool_size)

                    if use_transfer_matrices:
                        working_matrices[j] = (
                            pool[random_matrix_index] @ working_matrices[j]
                        )
                    else:
                        working_matrices[j] = matrix_utils.S_product(
                            working_matrices[j],
                            pool[random_matrix_index],
                        )

            else:
                with h5py.File(h5_file_path, "r+") as f:
                    working_matrices = f["working_matrices"]
                    for s, bs in zip(slices, batch_sizes):
                        # load the batch into RAM
                        batch_matrices = working_matrices[s]
                        for j in range(bs):
                            random_matrix_index = random.randrange(
                                0, pool_size
                            )
                            if use_transfer_matrices:
                                batch_matrices[j] = (
                                    pool[random_matrix_index]
                                    @ batch_matrices[j]
                                )
                            else:
                                batch_matrices[j] = matrix_utils.S_product(
                                    batch_matrices[j],
                                    pool[random_matrix_index],
                                )
                        # Write data back to h5 file
                        working_matrices[s] = batch_matrices

            # Do the analysis
            if i in analysis_points:
                print(f"Collecting data at index {i}")
                if is_single_batch:
                    for key, analysis_function in analysis_functions.items():
                        new_output = analysis_function(working_matrices)
                        # Save the output directly into the HDF5 dataset
                        with h5py.File(h5_file_path, "r+") as f:
                            f[key][analysis_points.index(i)] = new_output
                else:
                    with h5py.File(h5_file_path, "r+") as f:
                        working_matrices = f["working_matrices"]
                        for (
                            key,
                            analysis_function,
                        ) in analysis_functions.items():
                            batch_outputs = []
                            for s in slices:
                                batch_matrices = working_matrices[s]
                                batch_output = analysis_function(
                                    batch_matrices
                                )
                                batch_outputs.append(batch_output)
                            new_output = np.concatenate(batch_outputs, axis=0)
                            f[key][analysis_points.index(i)] = new_output

        # Remove working matrices from h5 memory to clear up memory
        if not is_single_batch:
            with h5py.File(h5_file_path, "r+") as f:
                del f["working_matrices"]
