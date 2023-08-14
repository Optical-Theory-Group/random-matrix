from dataclasses import dataclass
import os
import pickle

import numpy as np
import scipy.sparse
import sksparse.cholmod

from random_matrix.modes import mode_grid
from random_matrix.statistics import (
    index_finder,
    integration_task,
    medium_parameters,
    medium_statistics,
)
from random_matrix.utils import geometry_utils, matrix_utils
from random_matrix.utils.types import FloatLike


class InputStatisticsManager:
    def __init__(
        self,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        mode_grid: mode_grid.ModeGrid,
    ) -> None:
        """Input statistics manager class"""

        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.mode_grid = mode_grid

        # Attributes calculated from the input data
        self.index_finder = index_finder.IndexFinder(
            mode_grid, medium_parameters
        )
        self.integration_task_preparer = (
            integration_task.IntegrationTaskPreparer(
                mode_grid, medium_parameters, medium_statistics
            )
        )

    def get_statistics(self) -> FloatLike:
        """Compute the mean, covariance and pseudo-covariance for the elements
        of the scattering matrix."""

        # Prepare and execute integration tasks
        indices = self._get_indices()
        with open('indices.pkl', 'wb') as f:
            pickle.dump(indices, f)
        integration_task_list = self._get_integration_tasks(indices)
        result_list = integration_task_list.execute_tasks()

        # # Extract results from the list and build up statistical matrices
        mean_result_list = result_list.by_statistic_type("mean")
        cov_result_list = result_list.by_statistic_type("covariance")
        pseudo_cov_result_list = result_list.by_statistic_type(
            "pseudo_covariance"
        )

        mean_S = self._get_mean_S(mean_result_list)
        cov = self._get_covariance_matrix(cov_result_list)
        pseudo_cov = self._get_pseudo_covariance_matrix(pseudo_cov_result_list)
        sigma = 0.5 * scipy.sparse.bmat(
            [
                [np.real(cov + pseudo_cov), np.imag(-cov + pseudo_cov)],
                [np.imag(cov + pseudo_cov), np.real(cov + -pseudo_cov)],
            ]
        )
        with open('cov.pkl', 'wb') as f:
            pickle.dump(cov, f)

        # chol = self._get_chol(sigma)

        return cov

    def _get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        return self.index_finder.get_indices()

    def _get_integration_tasks(
        self, indices: dict[str, dict[str, set[tuple[int, int]]]]
    ) -> integration_task.IntegrationTaskList:
        return self.integration_task_preparer.get_integration_tasks(indices)

    # -------------------------------------------------------------------------
    # Mean
    # -------------------------------------------------------------------------

    def _get_mean_S(
        self, mean_result_list: integration_task.IntegrationResultList
    ) -> FloatLike:
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
    ) -> FloatLike:
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

            mean_S[
                sub_block_locations[:, 0], sub_block_locations[:, 1]
            ] = np.ravel(result.integral)

        mean_S_sym = self.mode_grid.rec_mat @ mean_S.T @ self.mode_grid.rec_mat
        mean_S = mean_S + mean_S_sym

        # Multiply by weights
        mean_weight_matrix = self._get_mean_weight_matrix()
        mean_S = mean_weight_matrix @ mean_S @ mean_weight_matrix

        return mean_S

    # -------------------------------------------------------------------------
    # Covariance matrices
    # -------------------------------------------------------------------------

    def _get_covariance_matrix(self, cov_result_list) -> FloatLike:
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

                block_ij, block_uv = block.split(",")
                rec_block_ij = (
                    "t2"
                    if block_ij == "t"
                    else "t"
                    if block_ij == "t2"
                    else block_ij
                )
                rec_block_uv = (
                    "t2"
                    if block_uv == "t"
                    else "t"
                    if block_uv == "t2"
                    else block_uv
                )

                extended_block_ij = [
                    block_ij,
                    rec_block_ij,
                    block_ij,
                    rec_block_ij,
                ]
                extended_block_uv = [
                    block_uv,
                    block_uv,
                    rec_block_uv,
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
                    print(sub_block_location)
                    print(block_ij, block_uv)
                    print(row, col)
                    sub_block = integral.reshape(4, 4)
                    sub_block = (sub_block + np.conj(sub_block.T)) / 2
                    cov[row : row + 4, col : col + 4] = sub_block
                    cov[col : col + 4, row : row + 4] = np.conj(sub_block.T)

        # Multiply by weights
        cov_weight_matrix = self._get_cov_weight_matrix()
        cov = cov_weight_matrix @ cov @ cov_weight_matrix

        return cov

    def _get_pseudo_covariance_matrix(self, pseudo_cov_result_list):
        """Construct the pseudo covariance matrix from the psuedo covariance
        results list"""

        # Four for the fourst matrices, another 4 for polarisation
        # 4x4 correlation matrices

        size_of_pseudo_cov = (self.mode_grid.num_propagating) ** 2 * 4 * 4

        pseudo_cov = scipy.sparse.dok_array(
            (size_of_pseudo_cov, size_of_pseudo_cov), dtype=np.complex128
        )

        return pseudo_cov

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

        print(f"10^{i}")
        return chol

    # -------------------------------------------------------------------------
    # Weight matrices
    # -------------------------------------------------------------------------

    def _get_mean_weight_matrix(self) -> FloatLike:
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

    def _get_cov_weight_matrix(self) -> FloatLike:
        """Get a matrix whose elements are like 1/sqrt(wi wj)

        Used to distribute weights across the cov matrix.
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
