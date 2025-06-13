import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sksparse.cholmod
from typing import Any

from random_matrix.input_statistics import (
    index_finder,
    input_statistics_logger,
    integration_task,
    medium_parameters,
    medium_statistics,
    shape_classifier,
)
from random_matrix.input_statistics.shape_classifier import ClassQuadrupleList
from random_matrix.modes import mode_grid
from random_matrix.utils import matrix_utils
from random_matrix.utils.types import Numeric


class InputStatisticsManager:
    def __init__(
        self,
        simulation_name: str,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        mode_grid: mode_grid.ModeGrid,
        parent_data_dir: str | Path | None = None,
        supplied_indices: dict | None = None,
        cubature_scheme: Any | None = None,
        use_dirac_density: bool = True
    ) -> None:
        """Input statistics manager class"""

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

        # Set up folder for simulation output and metadata
        self.simulation_name = simulation_name
        self.simulation_path = self.parent_data_path / Path(simulation_name)
        self._validate_simulation_path()

        # Save input variables
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.mode_grid = mode_grid
        objects_to_save = {
            "medium_parameters": medium_parameters,
            "medium_statistics": medium_statistics,
            "mode_grid": mode_grid,
        }
        self.supplied_indices = supplied_indices
        self.cubature_scheme = cubature_scheme
        self.use_dirac_density = use_dirac_density
        for name, obj in objects_to_save.items():
            save_path = self.simulation_path / f"{name}.pkl"
            if not save_path.exists():
                with save_path.open("wb") as f:
                    pickle.dump(obj, f)

        # Save mode grid plots
        self.mode_grid.plot(
            show_indices=False,
            savefig=str(self.simulation_path / f"mode_grid.svg"),
        )
        self.mode_grid.plot(
            show_indices=True,
            savefig=str(self.simulation_path / f"mode_grid_indices.svg"),
        )

        # Generate metadatajson if doesn't exist
        self.metadata_path = self.simulation_path / "metadata.json"
        self._create_metadata_json()

        # Set up loggers based on boolean
        self.logger = input_statistics_logger.InputStatisticsManagerLogger()
        index_finder_logger = input_statistics_logger.IndexFinderLogger()
        shape_classifier_logger = (
            input_statistics_logger.ShapeClassifierLogger()
        )
        integration_task_preparer_logger = (
            input_statistics_logger.IntegrationTaskPreparerLogger()
        )

        # Set up class attributes
        self.index_finder = index_finder.IndexFinder(
            mode_grid, index_finder_logger
        )

        self.shape_classifier = shape_classifier.ShapeClassifier(
            mode_grid,
            shape_classifier_logger,
        )

        self.integration_task_preparer = (
            integration_task.IntegrationTaskPreparer(
                mode_grid,
                medium_parameters,
                medium_statistics,
                integration_task_preparer_logger,
            )
        )

    def _validate_parent_data_path(self) -> None:
        """Create the ouput_dir based on the given path"""
        if not self.parent_data_path.exists():
            self.parent_data_path.mkdir(parents=True, exist_ok=True)

    def _validate_simulation_path(self) -> None:
        """Create folder and subfolders for simulation data"""
        if not self.simulation_path.exists():
            self.simulation_path.mkdir(parents=True, exist_ok=True)

    def _create_metadata_json(self) -> None:
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
                t["variables"] = list(t.get("variables"))

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

    def get_statistics(self) -> Numeric:
        """Compute the mean, covariance and pseudo-covariance for the elements
        of the scattering matrix."""
        independent_elements_path = (
            self.simulation_path / "independent_elements.pkl"
        )
        indices_path = self.simulation_path / "indices.pkl"
        class_quadruple_list_path = (
            self.simulation_path / "class_quadruple_list.pkl"
        )

        # Find indices
        index_variables_exists = (
            independent_elements_path.exists() and indices_path.exists()
        )
        if index_variables_exists:
            with open(independent_elements_path, "rb") as f:
                independent_elements = pickle.load(f)
            with open(indices_path, "rb") as f:
                indices = pickle.load(f)
        else:
            independent_elements, indices = self._get_indices()
            with open(independent_elements_path, "wb") as f:
                pickle.dump(independent_elements, f)
            with open(indices_path, "wb") as f:
                pickle.dump(indices, f)

        # Classify shapes. We use the "t,t" indices because these
        class_quadruple_list_exists = class_quadruple_list_path.exists()
        if class_quadruple_list_exists:
            with open(class_quadruple_list_path, "rb") as f:
                class_quadruple_list = pickle.load(f)
        else:
            class_quadruple_list = self._classify_shapes(
                indices["covariance"]["pp,pp"]["t,t"]
            )
            with open(class_quadruple_list_path, "wb") as f:
                pickle.dump(class_quadruple_list, f)

            # Show class histrogram (to get an idea of run time)
            self._save_class_quadruple_list_bar_chart(class_quadruple_list)

        # Prepare and execute integration tasks
        integration_result_list = self._get_integration_results(
            class_quadruple_list, indices
        )
        return integration_result_list
        # Extract results from the list and build up statistical matrices
        mean_result_list = integration_result_list.by_statistic_type("mean")
        cov_result_list = integration_result_list.by_statistic_type(
            "covariance"
        )
        pseudo_cov_result_list = integration_result_list.by_statistic_type(
            "pseudo_covariance"
        )

        with self.logger.log("mean"):
            mean_S = self._get_mean_S(mean_result_list)

        with self.logger.log("covariance"):
            cov = self._get_covariance_matrix(cov_result_list)

        with self.logger.log("pseudo_covariance"):
            pseudo_cov = self._get_covariance_matrix(
                pseudo_cov_result_list, is_pseudo=True
            )

        sigma = 0.5 * scipy.sparse.bmat(
            [
                [np.real(cov + pseudo_cov), np.imag(-cov + pseudo_cov)],
                [np.imag(cov + pseudo_cov), np.real(cov + -pseudo_cov)],
            ]
        )

        chol = self._get_chol(sigma)

        return integration_result_list, mean_S, cov, sigma, chol

    def _get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        if self.supplied_indices is None:
            return self.index_finder.get_indices()
        else:
            return {}, self.supplied_indices

    def _classify_shapes(self, quadruple_indices):
        return self.shape_classifier.classify_shapes(quadruple_indices)

    def _get_integration_results(
        self,
        class_quadruples_list: ClassQuadrupleList,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> integration_task.IntegrationResultList:
        return self.integration_task_preparer.get_integration_results(
            class_quadruples_list, indices, cubature_scheme=self.cubature_scheme, use_dirac_density=self.use_dirac_density
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
                j, i = matrix_utils.get_sub_block_indices(
                    block,
                    sub_block_location,
                    self.mode_grid.is_reciprocal,
                    self.mode_grid.num_propagating,
                )
                mean_S[j : j + 2, i : i + 2] = sub_block

                # If reciprocal, fill out other elements of S that weren't
                # calculated
                if self.mode_grid.is_reciprocal:
                    # The transformed sub block
                    reciprocal_sub_block = matrix_utils.r_sym(sub_block)

                    # Where does the new sub block go within S?
                    (
                        j,
                        i,
                    ) = matrix_utils.get_reciprocal_sub_block_indices(
                        block,
                        sub_block_location,
                        self.mode_grid.num_propagating,
                    )
                    mean_S[j : j + 2, i : i + 2] = reciprocal_sub_block

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

        # Four for the fourst matrices, another 4 for polarisation
        # 4x4 correlation matrices

        size_of_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 4

        cov = scipy.sparse.dok_array(
            (size_of_cov, size_of_cov), dtype=np.complex128
        )

        for result in cov_result_list.results:
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
                extended_sub_block_locations = [(i, j, u, v), (-j, -i, -v, -u)]

                block_ij, block_uv = block.split(",")
                rec_block_ij = (
                    "t2"
                    if block_ij == "t"
                    else "t" if block_ij == "t2" else block_ij
                )
                rec_block_uv = (
                    "t2"
                    if block_uv == "t"
                    else "t" if block_uv == "t2" else block_uv
                )

                extended_block_ij = [
                    block_ij,
                    rec_block_ij,
                    block_ij,
                    rec_block_ij,
                ]
                extended_block_ij = [
                    block_ij,
                    rec_block_ij,
                ]

                extended_block_uv = [
                    block_uv,
                    block_uv,
                    rec_block_uv,
                    rec_block_uv,
                ]
                extended_block_uv = [
                    block_uv,
                    rec_block_uv,
                ]

                for sub_block_location, block_ij, block_uv in zip(
                    extended_sub_block_locations,
                    extended_block_ij,
                    extended_block_uv,
                ):
                    i, j, u, v = sub_block_location
                    row = matrix_utils.get_cov_sub_block_index(
                        block_ij, (i, j), self.mode_grid.num_propagating
                    )

                    col = matrix_utils.get_cov_sub_block_index(
                        block_uv, (u, v), self.mode_grid.num_propagating
                    )
                    sub_block = integral.reshape(4, 4)
                    if not is_pseudo:
                        if row == col:
                            sub_block = (
                                sub_block + np.conj(sub_block.T)
                            ) / 2.0

                        cov[row : row + 4, col : col + 4] = sub_block
                        cov[col : col + 4, row : row + 4] = np.conj(
                            sub_block.T
                        )
                    else:
                        if row == col:
                            sub_block = (sub_block + sub_block.T) / 2.0

                        cov[row : row + 4, col : col + 4] = sub_block
                        cov[col : col + 4, row : row + 4] = sub_block.T

        # Multiply by weights
        cov_weight_matrix = self._get_cov_weight_matrix()
        cov = cov_weight_matrix @ cov @ cov_weight_matrix

        return cov

    # -------------------------------------------------------------------------
    # Cholesky
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_chol(sigma):
        size_of_sigma = np.shape(sigma)[0]
        sigma = scipy.sparse.csc_array(sigma)

        for i in range(-30, 1, 1):
            try:
                sigma_altered = sigma + scipy.sparse.identity(
                    size_of_sigma
                ) * 10 ** (i)

                chol = sksparse.cholmod.cholesky(
                    sigma_altered, ordering_method="natural"
                ).L()

                break
            except sksparse.cholmod.CholmodNotPositiveDefiniteError:
                pass

        print(f"POWER USED FOR CHOL: 10^{i}")
        return chol

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
