from dataclasses import dataclass

import numpy as np

from random_matrix.modes import mode_grid
from random_matrix.statistics import (index_finder, integration_task,
                                      medium_parameters, medium_statistics)
from random_matrix.utils import matrix_utils, geometry_utils
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
        integration_task_list = self._get_integration_tasks(indices)
        result_list = integration_task_list.execute_tasks()

        # Extract results from the list and build up statistical matrices
        mean_result_list = result_list.by_statistic_type("mean")
        mean_S = self._get_mean_S(mean_result_list)

        return mean_S

    def _get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        return self.index_finder.get_indices()

    def _get_integration_tasks(
        self, indices: dict[str, dict[str, set[tuple[int, int]]]]
    ) -> integration_task.IntegrationTaskList:
        return self.integration_task_preparer.get_integration_tasks(indices)

    def _get_mean_S(
        self, mean_result_list: integration_task.IntegrationResultList
    ) -> FloatLike:
        """Construct the mean scattering matrix from the mean results list"""

        size_of_S = 4 * self.mode_grid.num_propagating
        mean_S = np.zeros((size_of_S, size_of_S), dtype=np.complex128)

        for result in mean_result_list.results:
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

    def _get_mean_weight_matrix(self) -> FloatLike:
        """Get a matrix whose elements are like 1/sqrt(w)

        Used to distribute weights across the mean matrix.
        """

        num_modes = self.mode_grid.num_propagating
        max_index = int((num_modes - 1) / 2)
        indices = range(-max_index, max_index + 1, 1)

        weight_list = []
        for index in indices:
            mode = self.mode_grid.by_index(index)
            weight_list.append(mode.weight)

        weights = np.diag(1.0 / np.sqrt(weight_list))
        weights = np.kron(weights, np.identity(2))
        weights = np.kron(np.identity(2), weights)
        return weights

        