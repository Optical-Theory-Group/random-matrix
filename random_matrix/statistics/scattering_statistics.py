from dataclasses import dataclass

from random_matrix.modes import mode_grid
from random_matrix.statistics import (
    medium_parameters,
    medium_statistics,
    index_finder,
    integration_task,
)


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

    def get_statistics(self) -> integration_task.IntegrationTaskList:
        indices = self._get_indices()
        integration_task_list = self._get_integration_tasks(indices)
        result_list = integration_task_list.execute_tasks()
        return result_list

    def _get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        return self.index_finder.get_indices()

    def _get_integration_tasks(
        self, indices
    ) -> integration_task.IntegrationTaskList:
        return self.integration_task_preparer.get_integration_tasks(indices)


class IntegrationManager:
    pass
