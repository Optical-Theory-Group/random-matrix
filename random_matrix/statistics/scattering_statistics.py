from dataclasses import dataclass

from random_matrix.modes import mode_grid
from random_matrix.statistics import medium_parameters, medium_statistics


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
        self.is_reciprocal = mode_grid.is_reciprocal
        self.independent_element_indices = (
            self._get_independent_element_indices()
        )
        self.mean_indices = self._get_mean_indices()


class IntegrationManager:
    pass
