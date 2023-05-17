from dataclasses import dataclass, field

import numpy as np
import scipy

from random_matrix.modes import mode_grid
from random_matrix.statistics import (
    density_integrals,
    medium_parameters,
    medium_statistics,
)
from random_matrix.utils import function_utils, special_functions
from random_matrix.utils.types import FloatLike, MathematicalFunction


@dataclass
class IntegrationTask:
    """General integration task container.

    Consists of a function that is to be integrated over many different domains

    Attributes
    ----------
    integrand:
        The function to be integrated.
    domain_stack:
        A numpy array containing all of the integration domains.
    statistic_type:
        String that tracks what type of statistic is being calculated.
        The options are "mean", "covariance" and "pseudo_covariance"
    block_location:
        Pair of strings that trick which block the integrated statistic is for
        Examples include ("pp", "t") or ("pppp", "tt")
    sub_block_locations:
        list describing how the task outputs map back to positions within the
        relevant block. The slice object is for accessing parts of the
        domain stack, while the tuple tracks the modes inolved in the
        statistical calculation.
    const_factor:
        Additional constant factors that multiply all the integrals in the task
    """

    integrand: MathematicalFunction
    domain_stack: FloatLike
    statistic_type: str
    block_location: tuple[str, str]
    sub_block_locations: list[tuple[slice, tuple[int, int]]]
    const_factor: FloatLike = 1.0


@dataclass
class IntegrationTaskConfig:
    """Configuration metadata for controlling the degree of parallelisation of the
    integration tasks

    Attributes
    ----------
    integrals_per_task:
        A limit for how many integration domains should be allowed in each
        task. If None, there will be no limit.
    """

    integrals_per_task: int | None = None


class IntegrationTaskPreparer:
    def __init__(
        self,
        mode_grid: mode_grid.ModeGrid,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        integration_task_config: IntegrationTaskConfig = (
            IntegrationTaskConfig()
        ),
    ):
        self.integration_task_config = integration_task_config
        self.mode_grid = mode_grid
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.integration_tasks: list[IntegrationTask] = []

    def get_mean_integral_tasks(
        self, mean_indices: dict[str, dict[str, tuple[int, int]]]
    ) -> list[IntegrationTask]:
        """Main method for preparing mean integral tasks"""

        # Factor common to all mean integrals
        const_factor = self.medium_parameters.mean_const_factor
        integration_tasks = []

        for wave_block, d in mean_indices.items():
            for block, index_set in d.items():
                # The integrand depends only on the matrix blocks. This will
                # be a function of kappa_inc (assumed to be equal to kappa_sca
                # due to the delta function constraint)
                integrand = self._get_mean_integrand(wave_block, block)

                # Variables that will be used for constructing tasks
                # These reset each time we go to a new block
                sub_block_locations = []
                triangle_stack = np.zeros((0, 3, 2), dtype=np.float64)
                stack_length = 0

                for indices in index_set:
                    # Get the triangulation of the mode
                    index = indices[0]
                    mode = self.mode_grid.by_index(index)
                    delaunay = scipy.spatial.Delaunay(mode.vertices)
                    new_triangles = mode.vertices[delaunay.simplices]

                    # Check how long the stack will become if the new triangles
                    # are added. If this length exceeds the limit, we begin
                    # working on a new task
                    new_stack_length = stack_length + len(new_triangles)
                    if (
                        self.integration_task_config.integrals_per_task
                        is not None
                        and new_stack_length
                        > self.integration_task_config.integrals_per_task
                    ):
                        new_task = IntegrationTask(
                            integrand,
                            triangle_stack,
                            statistic_type="mean",
                            block_location=(wave_block, block),
                            sub_block_locations=sub_block_locations,
                            const_factor=const_factor,
                        )
                        integration_tasks.append(new_task)

                        # Reset the triangle stack and stack length
                        triangle_stack = np.zeros((0, 3, 2), dtype=np.float64)
                        stack_length = 0
                        sub_block_locations = []

                    # Add location to sub_block_locations
                    new_slice = slice(stack_length, new_stack_length)
                    new_indices = indices
                    new_sub_block_location = (new_slice, new_indices)
                    sub_block_locations.append(new_sub_block_location)

                    # Add new triangles to stack
                    triangle_stack = np.vstack((triangle_stack, new_triangles))
                    stack_length += len(new_triangles)

                # If this point has been reached, we have exahusted all
                # triangles for a certing block of the scattering matrix
                # We now make the final task for the group and reset ready for
                # the next block of S.
                new_task = IntegrationTask(
                    integrand,
                    triangle_stack,
                    statistic_type="mean",
                    block_location=(wave_block, block),
                    sub_block_locations=sub_block_locations,
                    const_factor=const_factor,
                )
                integration_tasks.append(new_task)

        # Loop has finished and we have a list of tasks
        return integration_tasks

    def _get_mean_integrand(self, wave_block: str, block: str):
        mean_a_matrix = self.medium_statistics.get_mean_a_matrix()
        signs = self._get_signs(block)

        # Convert to a function of kappa
        mean_a_matrix_kappa = function_utils.fix_last_components(
            mean_a_matrix, signs
        )

        # Enforce delta function constraint on arguments
        mean_a_matrix_kappa = function_utils.equate_arguments(
            mean_a_matrix_kappa
        )

        # Multiply by sec factor
        integrand = function_utils.multiply_functions(
            [mean_a_matrix_kappa, special_functions.inverse_kz]  # type:ignore
        )

        # Multiply by sinc term for r and r2 blocks
        if block in {"r", "r2"}:
            integrand = function_utils.multiply_functions(
                [
                    integrand,
                    special_functions.get_sinc_mean_kappa(
                        self.medium_parameters.k, self.medium_parameters.L
                    ),
                ]
            )

        return integrand

    @staticmethod
    def _get_signs(block: str) -> tuple[..., int]:
        """Get signs of z components of wavevectors depending on the matrix
        block."""

        match block:
            case "t":
                signs = (1, 1)
            case "r":
                signs = (1, -1)
            case "t2":
                signs = (-1, -1)
            case "r2":
                signs = (-1, 1)
        return signs
