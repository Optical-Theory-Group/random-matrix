import pickle
from dataclasses import asdict
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sksparse.cholmod
from random_matrix.input_statistics import (
    index_finder,
    input_statistics_logger,
    integration_task,
    medium_parameters,
    medium_statistics,
)
from random_matrix.modes import mode_grid
from random_matrix.utils import matrix_utils
from random_matrix.utils.types import Numeric
from random_matrix.scattering_matrix import matrix_pool_manager
from tqdm import tqdm

DEFAULT_DATA_DIR = "data"
DEFAULT_METADATA_PATH = "metadata.json"
MODE_GRID_PATH = "mode_grid.svg"
MODE_GRID_WITH_INDICES_PATH = "mode_grid_with_indices.svg"
INDEPENDENT_ELEMENTS_PATH = "independent_elements.pkl"
INDICES_PATH = "indices.pkl"
INTEGRATION_RESULTS_PATH = "integration_result_list.pkl"
MEAN_S_PATH = "mean_S.npy"
CHOL_PATH = "chol.pkl"
COV_PATH = "cov.npz"
PSEUDO_COV_PATH = "pseudo_cov.npz"
REAL_COVARIANCE_PATH = "real_cov.pkl"
TRUNCATED_BLOCK_KEYS = ["r,r", "t,t", "r2,r2"]


class InputStatisticsManager:
    def __init__(
        self,
        simulation_name: str,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        mode_grid: mode_grid.ModeGrid,
        integration_task_config: integration_task.IntegrationTaskConfig,
        parent_data_dir: str | Path | None = None,
        _loaded: bool = False,
    ) -> None:
        """Input statistics manager class"""

        self.simulation_name = simulation_name
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.mode_grid = mode_grid
        self.integration_task_config = integration_task_config
        self.parent_data_dir = parent_data_dir
        self._loaded = _loaded

        self._setup_paths(parent_data_dir)
        self._setup_loggers()
        self._setup_classes()

        # Save data if this is a new creation
        if not _loaded:
            self._save_initial_objects()
            self._plot_mode_grid()
            self._create_metadata_json()

    def _setup_paths(self, parent_data_dir: str | Path | None) -> None:
        """Initialize parent and simulation paths and create directories."""
        self.parent_data_path = (
            Path(parent_data_dir)
            if parent_data_dir
            else Path.cwd() / DEFAULT_DATA_DIR
        )
        self.parent_data_path.mkdir(parents=True, exist_ok=True)
        self.simulation_path = self.parent_data_path / self.simulation_name
        self.simulation_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.simulation_path / DEFAULT_METADATA_PATH
        self.independent_elements_path = (
            self.simulation_path / INDEPENDENT_ELEMENTS_PATH
        )
        self.indices_path = self.simulation_path / INDICES_PATH
        self.integration_result_list_path = (
            self.simulation_path / INTEGRATION_RESULTS_PATH
        )
        self.mean_S_path = self.simulation_path / MEAN_S_PATH
        self.cov_path = self.simulation_path / COV_PATH
        self.pseudo_cov_path = self.simulation_path / PSEUDO_COV_PATH
        self.real_cov_path = self.simulation_path / REAL_COVARIANCE_PATH
        self.chol_path = self.simulation_path / CHOL_PATH

    def _setup_loggers(self) -> None:
        """Initialize loggers for the manager and helper classes."""
        self.loggers = {
            "input_statistics_manager": input_statistics_logger.InputStatisticsManagerLogger(),
            "index_finder": input_statistics_logger.IndexFinderLogger(),
            "integration_task_preparer": input_statistics_logger.IntegrationTaskPreparerLogger(),
        }

    def _setup_classes(self) -> None:
        """Initialize index finder, shape classifier, and integration preparer."""
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
        }

        for name, obj in objects_to_save.items():
            save_path = self.simulation_path / f"{name}.pkl"
            with save_path.open("wb") as f:
                pickle.dump(obj, f)

    def _plot_mode_grid(self):
        """Save mode grid plots with and without indices."""
        plot_kwargs = [
            {
                "savefig": self.simulation_path / MODE_GRID_PATH,
                "show_indices": False,
                "close_fig": True,
            },
            {
                "savefig": self.simulation_path / MODE_GRID_WITH_INDICES_PATH,
                "show_indices": True,
                "close_fig": True,
            },
        ]
        for kwargs in plot_kwargs:
            self.mode_grid.plot(**kwargs)

    def _create_metadata_json(self) -> None:
        """Create a JSON file containing metadata for the simulation."""
        if self.metadata_path.exists():
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
        with self.metadata_path.open("w") as f:
            json.dump(metadata, f, indent=1)

    @classmethod
    def from_name(
        cls,
        simulation_name: str | Path,
        parent_data_dir: str | Path | None = None,
    ):
        """Quick load of statistics manager when the data directory already
        exists"""

        # Default to the current working directory if none is given
        parent_data_path = (
            Path(parent_data_dir)
            if parent_data_dir
            else Path.cwd() / DEFAULT_DATA_DIR
        )
        if not parent_data_path.exists():
            raise FileNotFoundError(
                f"Parent data directory not found: {parent_data_path}"
            )

        simulation_path = parent_data_path / Path(simulation_name)
        if not simulation_path.exists():
            raise FileNotFoundError(
                f"Simulation directory not found: {simulation_path}"
            )

        objects_to_load = [
            "medium_parameters",
            "medium_statistics",
            "mode_grid",
        ]
        loaded_objects = {
            "simulation_name": simulation_name,
            "_loaded": True,
            "parent_data_dir": parent_data_dir,
        }

        for name in objects_to_load:
            load_path = simulation_path / f"{name}.pkl"
            if not load_path.exists():
                raise FileNotFoundError(f"Missing file: {load_path}")
            with load_path.open("rb") as f:
                loaded_objects[name] = pickle.load(f)

        return cls(**loaded_objects)

    @property
    def logger(self):
        return self.loggers["input_statistics_manager"]

    def get_matrix_pool_manager(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the mean, covariance and pseudo-covariance for the elements
        of the scattering matrix."""

        # If the statistics are already in memory, just load them
        if self.mean_S_path.exists() and self.chol_path.exists():
            with self.logger.log("load_statistics"):
                mean_S = np.load(self.mean_S_path)
                with open(self.chol_path, "rb") as f:
                    chol = pickle.load(f)
        else:
            index_variables_exists = (
                self.independent_elements_path.exists()
                and self.indices_path.exists()
            )
            if index_variables_exists:
                with self.logger.log("load_indices"):
                    with open(self.independent_elements_path, "rb") as f:
                        independent_elements = pickle.load(f)
                    with open(self.indices_path, "rb") as f:
                        indices = pickle.load(f)
            else:
                independent_elements, indices = self._get_indices()
                with open(self.independent_elements_path, "wb") as f:
                    pickle.dump(independent_elements, f)
                with open(self.indices_path, "wb") as f:
                    pickle.dump(indices, f)

            # # Classify shapes. We use the "t,t" indices because these
            # class_quadruple_list_exists = class_quadruple_list_path.exists()
            # if class_quadruple_list_exists:
            #     with open(class_quadruple_list_path, "rb") as f:
            #         class_quadruple_list = pickle.load(f)
            # else:
            #     class_quadruple_list = self._classify_shapes(
            #         indices["covariance"]["pp,pp"]["t,t"]
            #     )
            #     with open(class_quadruple_list_path, "wb") as f:
            #         pickle.dump(class_quadruple_list, f)

            #     # Show class histrogram (to get an idea of run time)
            #     self._save_class_quadruple_list_bar_chart(class_quadruple_list)

            # Calculate statistics and compile the covariance matrix
            real_cov_exists = self.real_cov_path.exists()
            if real_cov_exists:
                with self.logger.log("load_real_covariance"):
                    with open(self.real_cov_path, "rb") as f:
                        real_cov = pickle.load(f)
            else:
                statistics_exist = (
                    self.cov_path.exists()
                    and self.pseudo_cov_path.exists()
                    and self.mean_S_path.exists()
                )
                if statistics_exist:
                    with self.logger.log("load_partial_statistics"):
                        cov = scipy.sparse.load_npz(self.cov_path)
                        pseudo_cov = scipy.sparse.load_npz(
                            self.pseudo_cov_path
                        )
                        mean_S = np.load(self.mean_S_path)
                else:
                    integration_result_list_exists = (
                        self.integration_result_list_path.exists()
                    )
                    if integration_result_list_exists:
                        with self.logger.log("load_integration_results"):
                            with open(
                                self.integration_result_list_path, "rb"
                            ) as f:
                                integration_result_list = pickle.load(f)
                    else:
                        integration_result_list = (
                            self._get_integration_results(indices)
                        )
                        with open(
                            self.integration_result_list_path, "wb"
                        ) as f:
                            pickle.dump(integration_result_list, f)

                    # Extract results from the list and build up statistical matrices
                    with self.logger.log("mean"):
                        mean_result_list = (
                            integration_result_list.by_statistic_type("mean")
                        )
                        mean_S = self._get_mean_S(mean_result_list)
                        np.save(self.mean_S_path, mean_S)
                        del mean_result_list

                    with self.logger.log("covariance"):
                        cov_result_list = (
                            integration_result_list.by_statistic_type(
                                "covariance"
                            )
                        )
                        cov = self._get_covariance_matrix(cov_result_list)
                        scipy.sparse.save_npz(self.cov_path, cov.tocsr())
                        del cov_result_list

                    with self.logger.log("pseudo_covariance"):
                        pseudo_cov_result_list = (
                            integration_result_list.by_statistic_type(
                                "pseudo_covariance"
                            )
                        )
                        pseudo_cov = self._get_covariance_matrix(
                            pseudo_cov_result_list, is_pseudo=True
                        )
                        scipy.sparse.save_npz(
                            self.pseudo_cov_path, pseudo_cov.tocsr()
                        )
                        del pseudo_cov_result_list

                    del integration_result_list

                with self.logger.log("real_covariance"):
                    real_cov = {}
                    for key in TRUNCATED_BLOCK_KEYS:
                        real_cov[key] = (
                            matrix_utils.get_real_covariance_matrix(
                                matrix_utils.get_cov_block(cov, key)
                            ).tocsc()
                        )
                    with open(self.real_cov_path, "wb") as f:
                        pickle.dump(real_cov, f)

                del cov
                del pseudo_cov

            with self.logger.log("cholesky"):
                chol = {}
                for key, block in real_cov.items():
                    chol[key] = matrix_utils.get_cholesky_decomposition(block)

                with open(self.chol_path, "wb") as f:
                    pickle.dump(chol, f)

            del real_cov

        # Construct the matrix pool
        pool = matrix_pool_manager.MatrixPoolManager(
            self.simulation_name,
            self.medium_parameters,
            self.medium_statistics,
            self.mode_grid,
            mean_S,
            chol,
            self.parent_data_path,
        )
        return pool

    def _get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        return self.index_finder.get_indices()

    def _get_integration_results(
        self,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> integration_task.IntegrationResultList:
        return self.integration_task_preparer.get_integration_results(
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

    def _get_mean_S_vector(
        self, mean_result_list: integration_task.IntegrationResultList
    ) -> Numeric:
        """Construct the mean scattering matrix from the mean results list

        Experiments suggest that this is slower than the other method!
        """

        size_of_S = 4 * self.mode_grid.num_propagating
        mean_S = np.zeros((size_of_S, size_of_S), dtype=np.complex128)
        half_index = int(size_of_S / 2)

        for result in mean_result_list.results:
            # Skip if no new results
            if len(result.sub_block_locations) == 0:
                continue

            wave_block, block = result.block_location
            sub_block_locations = np.array(result.sub_block_locations)

            # Account for reciprocity
            if self.mode_grid.is_reciprocal:
                sub_block_locations += int(
                    (self.mode_grid.num_propagating - 1) / 2
                )

            sub_block_locations *= 2

            # Make copies
            add_y = np.copy(sub_block_locations)
            add_y[:, 1] += 1

            add_x = np.copy(sub_block_locations)
            add_x[:, 0] += 1

            add_xy = np.copy(sub_block_locations)
            add_xy += 1

            sub_block_locations = np.hstack(
                [sub_block_locations, add_y, add_x, add_xy]
            ).reshape((4 * len(add_y), 2))

            # Account for block
            match block:
                case "r":
                    pass
                case "t":
                    sub_block_locations[:, 0] += half_index
                case "t2":
                    sub_block_locations[:, 1] += half_index
                case "r2":
                    sub_block_locations += half_index

            mean_S[sub_block_locations[:, 0], sub_block_locations[:, 1]] = (
                np.ravel(result.integral)
            )

        mean_S_sym = self.mode_grid.rec_mat @ mean_S.T @ self.mode_grid.rec_mat
        mean_S = mean_S + mean_S_sym

        # Multiply by weights
        mean_weight_matrix = self._get_mean_weight_matrix()
        mean_S = mean_weight_matrix @ mean_S @ mean_weight_matrix

        return mean_S

    # -------------------------------------------------------------------------
    # Covariance matrices
    # -------------------------------------------------------------------------

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

        for result in tqdm(cov_result_list.results):
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
    # Cholesky
    # -------------------------------------------------------------------------

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
