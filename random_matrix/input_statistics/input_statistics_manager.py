import pickle
from dataclasses import asdict
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from random_matrix.input_statistics import (
    index_finder,
    input_statistics_logger,
    integration_task,
    medium_parameters,
    medium_statistics,
)
from typing import Any
from random_matrix.modes import mode_grid
from random_matrix.utils import matrix_utils, array_utils
from random_matrix.utils.types import Numeric
from random_matrix.input_statistics import matrix_pool_manager, paths
from tqdm import tqdm


class InputStatisticsManager:
    def __init__(
        self,
        simulation_name: str,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        mode_grid: mode_grid.ModeGrid,
        integration_task_config: integration_task.IntegrationTaskConfig,
        base_path: str | Path | None = None,
        _loaded: bool = False,
    ) -> None:
        self.simulation_name = simulation_name
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.mode_grid = mode_grid
        self.integration_task_config = integration_task_config
        self.base_path = base_path
        self._loaded = _loaded

        self._setup_paths()
        self._setup_loggers()
        self._setup_classes()

        # Save data if this is a new creation
        if not _loaded:
            self._save_initial_objects()
            self._plot_mode_grid()
            self._create_metadata_json()

    def _setup_paths(self) -> None:
        """Initialize paths for saving simulation data and create directories
        if they don't exist."""
        self.paths = paths.InputStatisticsPaths(
            self.simulation_name, self.base_path
        )

    def _setup_loggers(self) -> None:
        """Initialize loggers for this and helper classes."""
        self.loggers = {
            "input_statistics_manager": input_statistics_logger.InputStatisticsManagerLogger(),
            "index_finder": input_statistics_logger.IndexFinderLogger(),
            "integration_task_preparer": input_statistics_logger.IntegrationTaskPreparerLogger(),
        }

    def _setup_classes(self) -> None:
        """Initialize helper classes for calculation of statistics."""
        self.index_finder = index_finder.IndexFinder(
            self.mode_grid, self.loggers["index_finder"]
        )
        self.integration_task_preparer = (
            integration_task.IntegrationTaskPreparer(
                self.mode_grid,
                self.medium_parameters,
                self.medium_statistics,
                self.loggers["integration_task_preparer"],
                integration_task_config=self.integration_task_config,
            )
        )

    def _save_initial_objects(self) -> None:
        """Save medium parameters, medium statistics, and mode grid to disk."""
        objects_to_save = {
            "medium_parameters": self.medium_parameters,
            "medium_statistics": self.medium_statistics,
            "mode_grid": self.mode_grid,
            "integration_task_config": self.integration_task_config,
        }

        for name, obj in objects_to_save.items():
            save_path = self.paths[name]
            with save_path.open("wb") as f:
                pickle.dump(obj, f)

    def _plot_mode_grid(self):
        """Save mode grid plots with and without indices."""
        plot_kwargs_list = [
            {
                "save_path": self.paths.mode_grid_figure,
                "show_indices": False,
                "close_fig": True,
            },
            {
                "save_path": self.paths.mode_grid_figure_with_indices,
                "show_indices": True,
                "close_fig": True,
            },
        ]
        for plot_kwargs in plot_kwargs_list:
            self.mode_grid.plot(**plot_kwargs)

    def _create_metadata_json(self) -> None:
        """Create a JSON file containing metadata for the simulation."""
        if self.paths.metadata.exists():
            return

        # Initialize empty json
        wrapped_medium_parameters_json = {
            "medium_parameters": asdict(self.medium_parameters)
        }
        wrapped_medium_statistics_json = {
            "medium_statistics": {
                "particle_statistics": [
                    {
                        key: asdict(term).get(key)
                        for key in ["terms", "mixing_ratio", "particle_type"]
                    }
                    for term in self.medium_statistics.particle_terms
                ]
            }
        }

        # Fix set issue
        for p in wrapped_medium_statistics_json.get("medium_statistics").get(
            "particle_statistics"
        ):
            for t in p.get("terms"):
                t["variables"] = list(t.get("variables", []))

        wrapped_mode_grid_statistics = {
            "mode_grid": {
                "r_lim": self.mode_grid.r_lim,
                "is_reciprocal": self.mode_grid.is_reciprocal,
                "num_propagating": self.mode_grid.num_propagating,
                "num_evanescent": self.mode_grid.num_evanescent,
            }
        }

        metadata = {
            **wrapped_medium_parameters_json,
            **wrapped_medium_statistics_json,
            **wrapped_mode_grid_statistics,
        }
        with self.paths.metadata.open("w") as f:
            json.dump(metadata, f, indent=1)

    @classmethod
    def from_name(
        cls,
        simulation_name: str,
        base_path: str | Path,
    ):
        """Quick load of statistics manager when the data directory already
        exists"""

        # Default to the current working directory if none is given
        base_path = Path(base_path)
        simulation_path = base_path / simulation_name

        if not simulation_path.exists():
            raise FileNotFoundError(f"Path: {simulation_path} does not exist.")

        load_paths = paths.InputStatisticsPaths(simulation_name, base_path)
        objects_to_load = [
            "medium_parameters",
            "medium_statistics",
            "mode_grid",
            "integration_task_config",
        ]
        loaded_objects = {
            "simulation_name": simulation_name,
            "base_path": base_path,
            "_loaded": True,
        }

        for name in objects_to_load:
            load_path = load_paths[name]
            if not load_path.exists():
                raise FileNotFoundError(f"Missing file: {load_path}")
            with load_path.open("rb") as f:
                loaded_objects[name] = pickle.load(f)

        return cls(**loaded_objects)

    @property
    def logger(self):
        return self.loggers["input_statistics_manager"]

    def get_matrix_pool_manager(self) -> matrix_pool_manager.MatrixPoolManager:
        """Main method for computing the mean, covariance and pseudo-covariance
        for the elements of the scattering matrix.

        If things have already been partially calculated, it will continue from
        where it got to."""

        if not (self.paths.mean_S.exists() and self.paths.cholesky.exists()):
            # Step 1) Get the indices. These will tell the integration methods
            # which integrals need to be calculated.
            self.calculate_volumes()

            self.calculate_indices()

            # Step 2) Calculate the mean scattering matrix
            self.calculate_mean_S()

            # Step 3) Calculate the integrals and form the Cholesky matrices
            # 3.1) Pre-compute required volumes and A matrix values
            self.calculate_a_matrix_values()

            # 3.2) Get the Cholesky matrices for each scattering matrix block
            self.calculate_cholesky_blocks()

            # 3.3) Load all Cholesky decompositions into memory and form a dict
            self.calculate_cholesky_dict()

        # Construct the matrix pool
        with self.logger.log("complete"):
            pool = matrix_pool_manager.MatrixPoolManager(
                self.simulation_name,
                self.paths.base,
            )
        return pool

    # -------------------------------------------------------------------------
    # Methods for calculating statistics
    # -------------------------------------------------------------------------

    def calculate_indices(self) -> None:
        """Calculate scattering matrix indices for which statistics exist and
        save the result to memory"""
        if (
            self.paths.indices.exists()
            and self.paths.independent_elements.exists()
        ):
            with self.logger.log("indices_exists"):
                return

        with self.logger.log("indices"):
            self.index_finder.calculate_indices(
                self.paths.independent_elements, self.paths.indices
            )

    def calculate_mean_S(self) -> None:
        """Calculate the mean scattering matrix and save the result to
        memory"""
        if self.paths.mean_S.exists():
            with self.logger.log("mean_S_exists"):
                return
        else:
            with self.logger.log("mean_S"):
                with open(self.paths.indices, "rb") as f:
                    indices = pickle.load(f)

                integration_result_list = self.get_integration_results(indices)
                mean_result_list = integration_result_list.by_statistic_type(
                    "mean"
                )
                mean_S = self._get_mean_S(mean_result_list)
                np.save(self.paths.mean_S, mean_S)

    def calculate_volumes(self) -> None:
        """Pre-compute volumes for integration."""
        if self.paths.volumes.exists():
            with self.logger.log("volumes_exists"):
                return

        with self.logger.log("volumes"):
            self.integration_task_preparer.calculate_volumes(
                self.paths.volumes
            )

    def calculate_a_matrix_values(self) -> None:
        """Pre-compute A matrix values"""
        if self.paths.a_matrix_values.exists():
            with self.logger.log("a_matrix_values_exists"):
                return

        with self.logger.log("a_matrix"):
            self.integration_task_preparer.calculate_a_matrix_values(
                self.paths.a_matrix_values
            )

    def calculate_cholesky_blocks(self):
        """Wrapper function for calculating the cholesky matrices for each
        S matrix block individually"""
        if all(
            [path.exists() for path in self.paths.cholesky_blocks.values()]
        ):
            with self.logger.log("cholesky_blocks_exists"):
                return

        with self.logger.log("cholesky_blocks"):
            with open(self.paths.indices, "rb") as f:
                indices = pickle.load(f)

            for block_key in paths.BLOCK_KEYS:
                if not self.paths.cholesky_blocks[block_key].exists():
                    partial_indices = (
                        indices.get("covariance")
                        .get("pp,pp")
                        .get(f"{block_key},{block_key}")
                    )
                    self.calculate_cholesky_block(partial_indices, block_key)

    def calculate_cholesky_block(self, indices: Any, block_key: str) -> None:
        # Build the real covariance matrix
        if self.paths.covariance_blocks[block_key].exists():
            with self.logger.log("real_covariance_exists", block=block_key):
                real_covariance_matrix = scipy.sparse.load_npz(
                    self.paths.covariance_blocks[block_key]
                )
        else:
            with self.logger.log("real_covariance", block=block_key):
                real_covariance_matrix = self.calculate_real_covariance_matrix(
                    indices, block_key
                )

        # Get the cholesky decomposition of the real covariance matrix
        # Note that if we arrive here, we must calculate the cholesky. There's
        # no need to check existence.
        with self.logger.log("cholesky_block", block=block_key):
            cholesky_block = matrix_utils.get_cholesky_decomposition(
                real_covariance_matrix
            )
            scipy.sparse.save_npz(
                self.paths.cholesky_blocks[block_key], cholesky_block
            )

    def calculate_cholesky_dict(self) -> None:
        """From saved cholesky matrices, form a dictionary of sub-chols, one
        for the different blocks of the S matrix"""
        with self.logger.log("cholesky_dict"):
            cholesky_dict = {}
            for block_key in paths.BLOCK_KEYS:
                cholesky_dict[block_key] = scipy.sparse.load_npz(
                    self.paths.cholesky_blocks[block_key]
                )
            with open(self.paths.cholesky, "wb") as f:
                pickle.dump(cholesky_dict, f)

    def get_integration_results(
        self,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> integration_task.IntegrationResultList:
        return self.integration_task_preparer.get_integration_results(
            indices,
        )

    def _get_integration_results_generator(
        self,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> integration_task.IntegrationResultList:
        return self.integration_task_preparer._get_covariance_results_lattice_generator(
            indices,
        )

    def show_report(self):
        self.index_finder.show_report()
        self.shape_classifier.show_report()
        # self.integration_task_preparer.show_report()
        # self.logger.show_report()

    def _save_class_quadruple_list_bar_chart(self, class_quadruple_list):
        fig, ax = plt.subplots()
        frequencies = [c.num_members + 1 for c in class_quadruple_list.classes]
        cumulative = np.cumsum(frequencies)
        cumulative = cumulative / cumulative[-1]
        total = sum(frequencies)
        xs = list(range(len(frequencies)))
        ax.plot(xs, cumulative)
        ax.set_ylim(0, max(cumulative))
        ax.set_xlim(-100, len(frequencies))
        ax.set_title(f"{total}")
        fig.savefig(self.simulation_path / "class_bars.svg", format="svg")
        plt.close()

    # -------------------------------------------------------------------------
    # Mean
    # -------------------------------------------------------------------------

    def _get_mean_S(
        self, mean_result_list: integration_task.IntegrationResultList
    ) -> Numeric:
        """Construct the mean scattering matrix from the mean results list"""

        size_of_S = 4 * self.mode_grid.num_propagating
        mean_S = np.zeros((size_of_S, size_of_S), dtype=np.complex128)

        for result in mean_result_list.results:
            # Skip if no new results
            if len(result.sub_block_locations) == 0:
                continue

            wave_block, block = result.block_location
            for integral, sub_block_location in zip(
                result.integral, result.sub_block_locations
            ):
                sub_block = integral.reshape(2, 2)
                indices = matrix_utils.get_sub_block_indices(
                    block,
                    sub_block_location,
                    self.mode_grid.is_reciprocal,
                    self.mode_grid.num_propagating,
                )
                mean_S[indices] = sub_block

                # If reciprocal, fill out other elements of S that weren't
                # calculated
                if self.mode_grid.is_reciprocal:
                    # The transformed sub block
                    reciprocal_sub_block = matrix_utils.r_sym(sub_block)

                    # Where does the new sub block go within S?
                    indices = matrix_utils.get_reciprocal_sub_block_indices(
                        block,
                        sub_block_location,
                        self.mode_grid.num_propagating,
                    )
                    mean_S[indices] = reciprocal_sub_block

        # Multiply by weights
        mean_weight_matrix = self._get_mean_weight_matrix()
        mean_S = mean_weight_matrix @ mean_S @ mean_weight_matrix

        return mean_S

    # -------------------------------------------------------------------------
    # Covariance matrices
    # -------------------------------------------------------------------------

    def calculate_real_covariance_matrix(
        self, indices: list, block_key: str, num_batches: int = 100
    ):
        split_indices = array_utils.split_list(indices, num_batches)
        integration_result_generators = [
            self.integration_task_preparer.get_covariance_results_lattice_generator(
                index_list,
                block_key,
                self.paths.a_matrix_values,
                self.paths.volumes,
            )
            for index_list in split_indices
        ]

        partial_paths = [
            self.paths.get_covariance_blocks_partial_path(block_key, count)
            for count in range(num_batches)
        ]

        # Get partial covariance matrices
        for count, (path, integration_result_generator) in enumerate(
            zip(partial_paths, integration_result_generators)
        ):
            if path.exists():
                continue
            with self.logger.log(
                "covariance_partial", count=count + 1, total=num_batches
            ):
                self.calculate_real_covariance_matrix_partial(
                    integration_result_generator, block_key, path
                )

        # Combine all the partial sparse matrices together
        cov = None
        for path in partial_paths:
            C = scipy.sparse.load_npz(path)
            if cov is None:
                cov = C
            else:
                cov = cov + C
        scipy.sparse.save_npz(self.paths.covariance_blocks[block_key], cov)

        # Remove all the partial covs after the final one has been built
        for path in partial_paths:
            os.remove(path)

        return cov

    def calculate_real_covariance_matrix_partial(
        self,
        result_list: list[integration_task.IntegrationResult],
        block_key: str,
        path: Path,
    ) -> None:
        """Construct the real covariance matrix from a covariance results
        generator"""

        R = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Size is due to...
        # For each element of S we need 1 element
        # That's num_modes **2 * 4 (because each element is a 2x2 sub-block
        # with 4 elements
        # Then * 2 because we want the real covariance matrix that separates
        # out real and imaginary parts
        size_of_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 2
        half_size = int(size_of_cov // 2)

        # Prepare data storage lists
        master_row_inds = []
        master_col_inds = []
        master_values = []

        # Main loop
        for result in tqdm(result_list):
            _, block_location = result.block_location
            integral = result.integral[0]
            C = integral.reshape(4, 4)
            sub_block_location = result.sub_block_locations[0]
            i, j, u, v = sub_block_location

            extended_sub_block_locations = [sub_block_location]
            extended_correlation_matrices = [C]

            if block_location in ["t,t"]:
                # There are two sub-cases, depending on whether we have a
                # sub-block autocorrelation or a inter-block cross correlation
                if (i, j) == (u, v):
                    # Auto-correlations. No additional cases here.
                    pass
                else:
                    # Cross correlations. Swapping indices is an additional
                    # case (correlation in opposite order)
                    extended_sub_block_locations.append((u, v, i, j))
                    extended_correlation_matrices.append(C.T.conj())

            elif block_location in ["r,r", "r2,r2"]:
                # There are more cases here than with t because of reciprocity
                if (i, j) == (u, v):
                    pass
                    # # Auto-correlations
                    # if i + j == 0:
                    #     # On the anti-diagonal. (i,j) has reciprocal partner
                    #     # equal to itself. No additional correlations
                    #     pass
                    # else:
                    #     # Off the anti-diagonal. (i,j) has reciprocal partner
                    #     # (-j, -i), which is correlated with (i,j)
                    #     extended_sub_block_locations.append((i, j, -j, -i))
                    #     extended_correlation_matrices.append(C @ R)
                    #     extended_sub_block_locations.append((-j, -i, i, j))
                    #     extended_correlation_matrices.append((C @ R).T.conj())

                    #     # (-j, -i) is also correlated with itself
                    #     extended_sub_block_locations.append((-j, -i, -j, -i))
                    #     extended_correlation_matrices.append(R @ C @ R)
                else:
                    # Cross-correlations
                    # Swapping indices (correlation in opposite order)
                    extended_sub_block_locations.append((u, v, i, j))
                    extended_correlation_matrices.append(C.T.conj())

                    # if i + j != 0:
                    #     # (i,j) has the reciprocal partner (-j,-i), which is
                    #     # also correlated with (u,v)
                    #     extended_sub_block_locations.append((-j, -i, u, v))
                    #     extended_correlation_matrices.append(R @ C)
                    #     extended_sub_block_locations.append((u, v, -j, -i))
                    #     extended_correlation_matrices.append((R @ C).T.conj())

                    # if u + v != 0:
                    #     # (u,v) has reciprocal partner (-v,-u), which is also
                    #     # correlated with (i,j)
                    #     extended_sub_block_locations.append((i, j, -v, -u))
                    #     extended_correlation_matrices.append(C @ R)
                    #     extended_sub_block_locations.append((-v, -u, i, j))
                    #     extended_correlation_matrices.append((C @ R).T.conj())

                    # if i + j != 0 and u + v != 0:
                    #     # The reciprocal partners of (i,j) and (u,v) are also
                    #     # correlated with each other
                    #     extended_sub_block_locations.append((-j, -i, -v, -u))
                    #     extended_correlation_matrices.append(R @ C @ R)
                    #     extended_sub_block_locations.append((-v, -u, -j, -i))
                    #     extended_correlation_matrices.append(
                    #         (R @ C @ R).T.conj()
                    #     )

            for sub_block_location, correlation_matrix in zip(
                extended_sub_block_locations, extended_correlation_matrices
            ):
                # Work out the indices where the matrix must go within the cov
                # matrix
                row_slice, col_slice = matrix_utils.get_cov_sub_block_indices(
                    "r,r",
                    sub_block_location,
                    self.mode_grid.is_reciprocal,
                    self.mode_grid.num_propagating,
                )
                rows = np.arange(row_slice.start, row_slice.stop)
                cols = np.arange(col_slice.start, col_slice.stop)
                rr, cc = np.meshgrid(rows, cols, indexing="ij")
                new_row_inds = rr.ravel()
                new_col_inds = cc.ravel()
                new_values_real = correlation_matrix.ravel().real
                new_values_imag = correlation_matrix.ravel().imag

                # Find non-zero values of new_values_real and populate the cov
                # matrix at the appropriate places
                non_zero_indices = new_values_real.nonzero()

                # Top left block
                master_row_inds.extend(new_row_inds[non_zero_indices])
                master_col_inds.extend(new_col_inds[non_zero_indices])
                master_values.extend(new_values_real[non_zero_indices])

                # Bottom right block
                master_row_inds.extend(
                    new_row_inds[non_zero_indices] + half_size
                )
                master_col_inds.extend(
                    new_col_inds[non_zero_indices] + half_size
                )
                master_values.extend(new_values_real[non_zero_indices])

                # --------------------------------------------------------------
                # Same but for imaginary values.
                non_zero_indices = new_values_imag.nonzero()

                # Top right block
                master_row_inds.extend(new_row_inds[non_zero_indices])
                master_col_inds.extend(
                    new_col_inds[non_zero_indices] + half_size
                )
                master_values.extend(-new_values_imag[non_zero_indices])

                # Bottom left block
                master_row_inds.extend(
                    new_row_inds[non_zero_indices] + half_size
                )
                master_col_inds.extend(new_col_inds[non_zero_indices])
                master_values.extend(new_values_imag[non_zero_indices])

        cov = 0.5 * scipy.sparse.coo_matrix(
            (master_values, (master_row_inds, master_col_inds)),
            shape=(size_of_cov, size_of_cov),
        )
        scipy.sparse.save_npz(path, cov)

    def get_real_covariance_matrix(
        self, result_list: list[integration_task.IntegrationResult]
    ) -> Numeric:
        """Construct the real covariance matrix from a covariance results
        generator"""

        REC_BLOCKS = {"r": "r", "t": "t2", "t2": "t", "r2": "r2"}
        REC_TRANSFORM = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        WEIGHTS = self.mode_grid.propagating_modes_weights_dict

        # Size is due to...
        # For each element of S we need 1 element
        # That's num_modes **2 * 4 (because each element is a 2x2 sub-block
        # with 4 elements
        # Then *4 again because of r, t, t2, r2
        # Then * 2 because we want the real covariance matrix that separates
        # out real and imaginary parts
        size_of_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 2
        half_size = int(size_of_cov // 2)

        # Prepare data storage lists
        master_row_inds = []
        master_col_inds = []
        master_values = []

        # Main loop
        for result in tqdm(result_list):
            _, block_location = result.block_location
            integral = result.integral[0]
            sub_block_location = result.sub_block_locations[0]
            i, j, u, v = sub_block_location

            if block_location in ["r,r", "r2,r2"] and not (
                i == -v and j == -u
            ):
                # Prepare all reciprocal cases
                extended_sub_block_locations = [
                    (i, j, u, v),
                    (-j, -i, u, v),
                    (i, j, -v, -u),
                    (-j, -i, -v, -u),
                ]

                block_ij, block_uv = block_location.split(",")
                rec_block_ij = REC_BLOCKS.get(block_ij)
                rec_block_uv = REC_BLOCKS.get(block_uv)

                extended_block_locations = [
                    f"{block_ij},{block_uv}",
                    f"{rec_block_ij},{block_uv}",
                    f"{block_ij},{rec_block_uv}",
                    f"{rec_block_ij},{rec_block_uv}",
                ]

                correlation_matrix = integral.reshape(4, 4)
                extended_correlation_matrices = [
                    correlation_matrix,
                    REC_TRANSFORM @ correlation_matrix,
                    correlation_matrix @ REC_TRANSFORM,
                    REC_TRANSFORM @ correlation_matrix @ REC_TRANSFORM,
                ]
            else:
                extended_sub_block_locations = [sub_block_location]
                extended_block_locations = [block_location]
                extended_correlation_matrices = [integral.reshape(4, 4)]

            # Loop over reciprocal sub-cases
            for (
                sub_block_location,
                block_location,
                correlation_matrix,
            ) in zip(
                extended_sub_block_locations,
                extended_block_locations,
                extended_correlation_matrices,
            ):
                # Get weights
                i, j, u, v = sub_block_location
                w_i = WEIGHTS.get(i)
                w_j = WEIGHTS.get(j)
                w_u = WEIGHTS.get(u)
                w_v = WEIGHTS.get(v)
                # correlation_matrix = (
                #     correlation_matrix * 1 / np.sqrt(w_i * w_j * w_u * w_v)
                # )

                # Work out the indices
                row_slice, col_slice = matrix_utils.get_cov_sub_block_indices(
                    "r,r",
                    sub_block_location,
                    self.mode_grid.is_reciprocal,
                    self.mode_grid.num_propagating,
                )
                rows = np.arange(row_slice.start, row_slice.stop)
                cols = np.arange(col_slice.start, col_slice.stop)
                rr, cc = np.meshgrid(rows, cols, indexing="ij")
                new_row_inds = rr.ravel()
                new_col_inds = cc.ravel()
                new_values = correlation_matrix.ravel()
                master_row_inds.extend(new_row_inds)
                master_col_inds.extend(new_col_inds)
                master_values.extend(new_values)

                # Get transpose indices
                if not (i == u and j == v):
                    new_row_inds_sym = new_col_inds
                    new_col_inds_sym = new_row_inds
                    new_values_sym = np.conj(new_values.reshape(4, 4)).ravel()
                    master_row_inds.extend(new_row_inds_sym)
                    master_col_inds.extend(new_col_inds_sym)
                    master_values.extend(new_values_sym)

        # Remove duplicates if they exist
        coords = np.vstack((master_row_inds, master_col_inds)).T
        _, idx_first = np.unique(coords, axis=0, return_index=True)
        master_row_inds = list(np.array(master_row_inds)[idx_first])
        master_col_inds = list(np.array(master_col_inds)[idx_first])
        master_values = list(np.array(master_values)[idx_first])

        # Add other blocks
        offset = half_size
        master_row_inds = (
            master_row_inds
            + master_row_inds
            + [x + offset for x in master_row_inds]
            + [x + offset for x in master_row_inds]
        )
        master_col_inds = (
            master_col_inds
            + [x + offset for x in master_col_inds]
            + master_col_inds
            + [x + offset for x in master_col_inds]
        )
        master_values = (
            [x.real for x in master_values]
            + [-x.imag for x in master_values]
            + [x.imag for x in master_values]
            + [x.real for x in master_values]
        )
        cov = (
            0.5
            * scipy.sparse.coo_matrix(
                (master_values, (master_row_inds, master_col_inds)),
                shape=(size_of_cov, size_of_cov),
            ).tocsr()
        )
        return cov

    def _get_covariance_matrix(
        self, cov_result_list, is_pseudo=False
    ) -> Numeric:
        """Construct the regular covariance matrix from the covariance results
        list"""

        REC_BLOCKS = {"r": "r", "t": "t2", "t2": "t", "r2": "r2"}
        REC_TRANSFORM = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        WEIGHTS = self.mode_grid.propagating_modes_weights_dict

        # Four for the fourst matrices, another 4 for polarisation
        # 4x4 correlation matrices

        size_of_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 4

        cov = scipy.sparse.dok_array(
            (size_of_cov, size_of_cov), dtype=np.complex128
        )

        for result in tqdm(cov_result_list):
            # Skip if no new results
            if len(result.sub_block_locations) == 0:
                continue

            wave_block, block = result.block_location
            for integral, sub_block_location in zip(
                result.integral, result.sub_block_locations
            ):
                # Prepare all reciprocal cases
                i, j, u, v = sub_block_location
                extended_sub_block_locations = [
                    (i, j, u, v),
                    (-j, -i, u, v),
                    (i, j, -v, -u),
                    (-j, -i, -v, -u),
                ]

                block_ij, block_uv = block.split(",")
                rec_block_ij = REC_BLOCKS.get(block_ij)
                rec_block_uv = REC_BLOCKS.get(block_uv)

                extended_block_locations = [
                    f"{block_ij},{block_uv}",
                    f"{rec_block_ij},{block_uv}",
                    f"{block_ij},{rec_block_uv}",
                    f"{rec_block_ij},{rec_block_uv}",
                ]

                correlation_matrix = integral.reshape(4, 4)
                extended_correlation_matrices = [
                    correlation_matrix,
                    REC_TRANSFORM @ correlation_matrix,
                    correlation_matrix @ REC_TRANSFORM,
                    REC_TRANSFORM @ correlation_matrix @ REC_TRANSFORM,
                ]

                for (
                    sub_block_location,
                    block_location,
                    correlation_matrix,
                ) in zip(
                    extended_sub_block_locations,
                    extended_block_locations,
                    extended_correlation_matrices,
                ):
                    indices = matrix_utils.get_cov_sub_block_indices(
                        block_location,
                        sub_block_location,
                        self.mode_grid.is_reciprocal,
                        self.mode_grid.num_propagating,
                    )
                    # Get weights
                    i, j, u, v = sub_block_location
                    w_i = WEIGHTS.get(i)
                    w_j = WEIGHTS.get(j)
                    w_u = WEIGHTS.get(u)
                    w_v = WEIGHTS.get(v)
                    correlation_matrix = (
                        correlation_matrix * 1 / np.sqrt(w_i * w_j * w_u * w_v)
                    )
                    cov[indices] = correlation_matrix

                    if not is_pseudo:
                        cov[indices[::-1]] = np.conj(correlation_matrix.T)
                    else:
                        cov[indices[::-1]] = correlation_matrix.T

        # Multiply by weights
        # cov_weight_matrix = self._get_cov_weight_matrix()
        # cov = cov_weight_matrix @ cov @ cov_weight_matrix

        return cov

    def _get_covariance_matrix_generator(
        self, cov_result_list, is_pseudo=False
    ) -> Numeric:
        """Construct the regular covariance matrix from the covariance results
        list"""

        REC_BLOCKS = {"r": "r", "t": "t2", "t2": "t", "r2": "r2"}
        REC_TRANSFORM = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        WEIGHTS = self.mode_grid.propagating_modes_weights_dict

        # Four for the fourst matrices, another 4 for polarisation
        # 4x4 correlation matrices

        size_of_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 4

        cov = scipy.sparse.dok_array(
            (size_of_cov, size_of_cov), dtype=np.complex128
        )

        for result in cov_result_list:
            # Skip if no new results
            if len(result.sub_block_locations) == 0:
                continue

            wave_block, block = result.block_location
            for integral, sub_block_location in zip(
                result.integral, result.sub_block_locations
            ):
                # Prepare all reciprocal cases
                i, j, u, v = sub_block_location
                extended_sub_block_locations = [
                    (i, j, u, v),
                    (-j, -i, u, v),
                    (i, j, -v, -u),
                    (-j, -i, -v, -u),
                ]

                block_ij, block_uv = block.split(",")
                rec_block_ij = REC_BLOCKS.get(block_ij)
                rec_block_uv = REC_BLOCKS.get(block_uv)

                extended_block_locations = [
                    f"{block_ij},{block_uv}",
                    f"{rec_block_ij},{block_uv}",
                    f"{block_ij},{rec_block_uv}",
                    f"{rec_block_ij},{rec_block_uv}",
                ]

                correlation_matrix = integral.reshape(4, 4)
                extended_correlation_matrices = [
                    correlation_matrix,
                    REC_TRANSFORM @ correlation_matrix,
                    correlation_matrix @ REC_TRANSFORM,
                    REC_TRANSFORM @ correlation_matrix @ REC_TRANSFORM,
                ]

                for (
                    sub_block_location,
                    block_location,
                    correlation_matrix,
                ) in zip(
                    extended_sub_block_locations,
                    extended_block_locations,
                    extended_correlation_matrices,
                ):
                    indices = matrix_utils.get_cov_sub_block_indices(
                        block_location,
                        sub_block_location,
                        self.mode_grid.is_reciprocal,
                        self.mode_grid.num_propagating,
                    )
                    # Get weights
                    i, j, u, v = sub_block_location
                    w_i = WEIGHTS.get(i)
                    w_j = WEIGHTS.get(j)
                    w_u = WEIGHTS.get(u)
                    w_v = WEIGHTS.get(v)
                    correlation_matrix = (
                        correlation_matrix * 1 / np.sqrt(w_i * w_j * w_u * w_v)
                    )
                    cov[indices] = correlation_matrix

                    if not is_pseudo:
                        cov[indices[::-1]] = np.conj(correlation_matrix.T)
                    else:
                        cov[indices[::-1]] = correlation_matrix.T

        # Multiply by weights
        # cov_weight_matrix = self._get_cov_weight_matrix()
        # cov = cov_weight_matrix @ cov @ cov_weight_matrix

        return cov

    # -------------------------------------------------------------------------
    # Weight matrices
    # -------------------------------------------------------------------------

    def _get_mean_weight_matrix(self) -> Numeric:
        """Get a matrix whose elements are like 1/sqrt(w)

        Used to distribute weights across the mean matrix.
        """

        max_index = self.mode_grid.max_index
        indices = range(-max_index, max_index + 1, 1)

        weight_list = []
        for index in indices:
            mode = self.mode_grid.by_index(index)
            weight_list.append(mode.weight)

        weights = np.diag(1.0 / np.sqrt(weight_list))
        weights = np.kron(weights, np.identity(2))
        weights = np.kron(np.identity(2), weights)
        return weights

    def _get_cov_weight_matrix(self) -> Numeric:
        """Get a matrix whose elements are like 1/sqrt(wi wj)

        Used to distribute weights across the cov matrices.
        """

        max_index = self.mode_grid.max_index
        indices = range(-max_index, max_index + 1, 1)

        weight_list = []
        for index in indices:
            mode = self.mode_grid.by_index(index)
            weight_list.append(mode.weight)

        product_weight_list = []
        for w1 in weight_list:
            for w2 in weight_list:
                product_weight_list.append(w1 * w2)

        weights = scipy.sparse.diags(1.0 / np.sqrt(product_weight_list))
        weights = scipy.sparse.kron(weights, np.identity(4))
        weights = scipy.sparse.kron(np.identity(4), weights)

        return weights
