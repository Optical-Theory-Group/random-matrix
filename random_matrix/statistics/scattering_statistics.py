from dataclasses import dataclass
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.statistics.medium_statistics import MediumStatistics
from random_matrix.modes.mode_grid import ModeGrid


class InputStatisticsManager:
    def __init__(
        self,
        medium_parameters: MediumParameters,
        medium_statistics: MediumStatistics,
        mode_grid: ModeGrid,
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
