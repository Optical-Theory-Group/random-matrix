import functools
import os
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import scipy
from pathos.pools import ProcessPool

from random_matrix.input_statistics import (
    density_integrals,
    medium_parameters,
    medium_statistics,
)
from random_matrix.modes import mode_grid
from random_matrix.utils import (
    array_utils,
    function_utils,
    geometry_utils,
    integration_utils,
    special_functions,
)
from random_matrix.utils.types import FloatLike, MathematicalFunction
from random_matrix.input_statistics.input_statistics_logger import (
    IntegrationTaskPreparerLogger,
)


@dataclass(slots=True)
class IntegrationResult:
    """General integration result container.

    Contains results from executing an IntegrationTask object.

    Attributes
    ----------
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
    integral:
        The result of the task integration.
    """

    statistic_type: str
    block_location: tuple[str, str]
    sub_block_locations: list[tuple[int, int]]
    integral: FloatLike

    def __str__(self) -> str:
        string = (
            f"Statistic type: {self.statistic_type}\n"
            f"Matrix block: {self.block_location}\n"
            f"Number of integrals: {len(self.integral)}"
        )

        return string


@dataclass(slots=True)
class IntegrationResultList:
    """Container class for many IntegrationResult objects."""

    results: list[IntegrationResult] = field(default_factory=list)

    def __str__(self) -> str:
        string = f"Number of results: {len(self.results)}\n" f"Results key:\n"

        for i, result in enumerate(self.results):
            string += (
                f"{i}, {result.statistic_type}, "
                f"{len(result.integral)} integrals, "
                f"{result.block_location}\n"
            )

        return string

    def append_result(self, new_result: IntegrationResult) -> None:
        self.results.append(new_result)

    def by_statistic_type(self, statistic_type: str) -> Self:
        new_result_list = IntegrationResultList(
            [
                result
                for result in self.results
                if result.statistic_type == statistic_type
            ]
        )
        return new_result_list


@dataclass(slots=True)
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

    def __str__(self) -> str:
        string = (
            f"Statistic type: {self.statistic_type}\n"
            f"Matrix block: {self.block_location}\n"
            f"Number of integrals: {len(self.domain_stack)}"
        )

        return string

    def execute_task(self) -> IntegrationResult:
        """Perofrm the integral associated with the task."""

        match self.statistic_type:
            case "mean":
                integral = integration_utils.basic_triangle_integral(
                    self.integrand, self.domain_stack
                )
            case "covariance":
                integral = integration_utils.basic_simplex_integral(
                    self.integrand, self.domain_stack
                )
            case "pseudo_covariance":
                integral = integration_utils.basic_simplex_integral(
                    self.integrand, self.domain_stack
                )

        # Results should be added over the slices. This corresponds to adding
        # integrals over the triangular subregions of the integration domain

        output_dim, num_outputs = np.shape(integral)
        slices = [s[0] for s in self.sub_block_locations]
        locations = [s[1] for s in self.sub_block_locations]
        num_slices = len(slices)
        output = np.zeros((output_dim, num_slices), dtype=np.complex128)

        for i, s in enumerate(slices):
            partial_output = np.sum(integral[:, s], axis=-1)
            output[:, i] = partial_output

        output = np.transpose(output, (1, 0))

        return IntegrationResult(
            self.statistic_type,
            self.block_location,
            locations,
            self.const_factor * output,
        )


@dataclass(slots=True)
class IntegrationTaskList:
    """Container class for many IntegrationTask objects."""

    tasks: list[IntegrationTask] = field(default_factory=list)

    def __str__(self) -> str:
        string = f"Number of tasks: {len(self.tasks)}\n" f"Tasks key:\n"

        for i, task in enumerate(self.tasks):
            string += (
                f"{i}, {task.statistic_type}, "
                f"{len(task.domain_stack)} triangles, "
                f"{task.block_location}\n"
            )

        return string

    def append_task(self, new_task: IntegrationTask) -> None:
        """Add a new task to the task list"""

        self.tasks.append(new_task)

    def merge_task_list(self, new_task_list: Self) -> None:
        """Append tasks from another task list object to itself"""

        self.tasks += new_task_list.tasks

    def execute_tasks(self) -> IntegrationResultList:
        """Perofrm the integral associated with all tasks."""

        results_list = IntegrationResultList()
        for task in self.tasks:
            new_result = task.execute_task()
            results_list.append_result(new_result)
        return results_list


@dataclass(slots=True)
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
        logger=IntegrationTaskPreparerLogger(),
        integration_task_config: IntegrationTaskConfig = (
            IntegrationTaskConfig()
        ),
    ):
        self.integration_task_config = integration_task_config
        self.mode_grid = mode_grid
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.logger = logger

    def get_integration_tasks(
        self,
        quadruples,
        independent_elements,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> IntegrationTaskList:
        """Get an integration task list consisting of all necessary tasks.

        This is the main function that is run by IntegrationTaskPreparer
        """

        master_task_list = IntegrationTaskList()

        with self.logger.log("mean"):
            master_task_list.merge_task_list(
                self._get_mean_integration_tasks(indices["mean"])
            )

        with self.logger.log("covariance"):
            master_task_list.merge_task_list(
                self._get_covariance_integration_tasks(
                    quadruples, independent_elements["pp"]
                )
            )
        with self.logger.log("pseudo_covariance"):
            master_task_list.merge_task_list(
                self._get_pseudo_covariance_integration_tasks(
                    quadruples, independent_elements["pp"]
                )
            )

        return master_task_list

    def show_report(self) -> None:
        self.logger.show_report()

    # -------------------------------------------------------------------------
    # Mean
    # -------------------------------------------------------------------------

    def _get_mean_integrand(
        self, wave_block: str, block: str
    ) -> MathematicalFunction:
        mean_a_matrix = self.medium_statistics.get_mean_a_matrix()
        k = self.medium_parameters.k
        L = self.medium_parameters.L

        def mean_integrand(k_x: FloatLike, k_y: FloatLike) -> FloatLike:
            ki_x = k_x
            ki_y = k_y

            kj_x = k_x
            kj_y = k_y

            ki_z = np.sqrt(1.0 - ki_x**2 - ki_y**2)
            kj_z = np.sqrt(1.0 - kj_x**2 - kj_y**2)

            # Sort out signs of k_iz and k_jz
            if block in {"r2", "t2"}:
                ki_z = -ki_z
            if block in {"r", "t2"}:
                kj_z = -kj_z

            sinc_factor = 1.0
            if block in {"r", "r2"}:
                sinc_factor = special_functions.sinc(k * L * ki_z)

            sec_factor = 1.0 / ki_z
            output = (
                mean_a_matrix(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z)
                * sinc_factor
                * sec_factor
            )
            return output

        return mean_integrand

    def _get_mean_integration_tasks(
        self, mean_indices: dict[str, dict[str, tuple[int, int]]]
    ) -> IntegrationTaskList:
        """Main method for preparing mean integral tasks"""

        # Factor common to all mean integrals
        const_factor = self.medium_parameters.mean_const_factor
        main_task_list = IntegrationTaskList()

        for wave_block, d in mean_indices.items():
            sub_task_list = IntegrationTaskList()
            for block, index_set in d.items():
                # The integrand depends only on the matrix blocks. This will
                # be a function of kappa_inc (assumed to be equal to kappa_sca
                # due to the delta function constraint)
                integrand = self._get_mean_integrand(wave_block, block)
                integration_tasks = IntegrationTaskList()

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
                        and stack_length > 0
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
                        integration_tasks.append_task(new_task)

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

                # Once this point has been reached, we have exahusted all
                # triangles for a certing block of the scattering matrix
                # We now make the final task for the group
                new_task = IntegrationTask(
                    integrand,
                    triangle_stack,
                    statistic_type="mean",
                    block_location=(wave_block, block),
                    sub_block_locations=sub_block_locations,
                    const_factor=const_factor,
                )
                integration_tasks.append_task(new_task)
                sub_task_list.merge_task_list(integration_tasks)

            main_task_list.merge_task_list(sub_task_list)

        return main_task_list

    # -------------------------------------------------------------------------
    # Covariance
    # -------------------------------------------------------------------------

    def _get_covariance_integrand(
        self,
        wave_block_one: str,
        wave_block_two: str,
        block_one: str,
        block_two: str,
    ) -> MathematicalFunction:
        covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
        k = self.medium_parameters.k
        L = self.medium_parameters.L

        def covariance_integrand(
            k1_x: FloatLike,
            k1_y: FloatLike,
            k2_x: FloatLike,
            k2_y: FloatLike,
            d_x: FloatLike,
            d_y: FloatLike,
        ) -> FloatLike:
            # Convert to complex arrays
            k1_x = np.array(k1_x, dtype=complex)
            k1_y = np.array(k1_y, dtype=complex)
            k2_x = np.array(k2_x, dtype=complex)
            k2_y = np.array(k2_y, dtype=complex)
            d_x = np.array(d_x, dtype=complex)
            d_y = np.array(d_y, dtype=complex)

            ki_x = k1_x + d_x / 2
            ki_y = k1_y + d_y / 2

            kj_x = k1_x - d_x / 2
            kj_y = k1_y - d_y / 2

            ku_x = k2_x + d_x / 2
            ku_y = k2_y + d_y / 2

            kv_x = k2_x - d_x / 2
            kv_y = k2_y - d_y / 2

            ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
            kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
            ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
            kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

            # Sort out signs of k_iz and k_jz
            if block_one in {"r2", "t2"}:
                ki_z = -ki_z
            if block_two in {"r2", "t2"}:
                ku_z = -ku_z
            if block_one in {"r", "t2"}:
                kj_z = -kj_z
            if block_two in {"r", "t2"}:
                kv_z = -kv_z

            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )

            sec_factor = 1.0 / np.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

            output = (
                covariance_a_matrix(
                    ki_x,
                    ki_y,
                    ki_z,
                    kj_x,
                    kj_y,
                    kj_z,
                    ku_x,
                    ku_y,
                    ku_z,
                    kv_x,
                    kv_y,
                    kv_z,
                )
                * sinc_factor
                * sec_factor
            )
            return output

        return covariance_integrand

    def _get_covariance_integration_tasks(
        self,
        quadruples,
        independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
    ) -> IntegrationTaskList:
        # Multiprocessing parameters
        num_quadruples = len(quadruples)
        num_processes = min(num_quadruples, os.cpu_count())

        # Prepare integrands since these can't be pickled
        blocks = [
            "t,t",
            "t,r",
            "t,t2",
            "t,r2",
            "r,r",
            "r,t2",
            "r,r2",
            "t2,t2",
            "t2,r2",
            "r2,r2",
        ]

        integrand = {}

        for key in blocks:
            block_one, block_two = key.split(",")
            integrand[key] = self._get_covariance_integrand(
                "pp", "pp", block_one, block_two
            )

        parallelised_function = functools.partial(
            self._get_covariance_integration_tasks_partial,
            independent_elements=independent_elements,
            const_factor=self.medium_parameters.cov_const_factor,
            integrals_per_task=self.integration_task_config.integrals_per_task,
            integrand=integrand,
        )

        partial_quadruples = array_utils.split_list(quadruples, num_processes)

        with ProcessPool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_quadruples)

        main_task_list = IntegrationTaskList()

        for task_list in out:
            main_task_list.merge_task_list(task_list)

        return main_task_list

    @staticmethod
    def _get_covariance_integration_tasks_partial(
        quadruples,
        independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
        const_factor,
        integrals_per_task,
        integrand,
    ) -> IntegrationTaskList:
        """Main method for preparing covariance integral tasks"""

        # Factor common to all covariance integrals
        main_task_list = IntegrationTaskList()

        # wave block will look like "pp,pp", "pp,ep" etc.
        wave_block = "pp,pp"
        wave_block_one, wave_block_two = wave_block.split(",")

        # # The integrand depends only on the matrix blocks.

        # Variables that will be used for constructing tasks
        # These reset each time we go to a new block
        blocks = [
            "t,t",
            "t,r",
            "t,t2",
            "t,r2",
            "r,r",
            "r,t2",
            "r,r2",
            "t2,t2",
            "t2,r2",
            "r2,r2",
        ]

        # Initialise empty dictionaries for storing all the data that will be
        # used in the ensuing loop
        sub_block_locations = {}
        simplex_stack = {}
        stack_length = {}
        integration_tasks = {}

        for key in blocks:
            sub_block_locations[key] = []
            simplex_stack[key] = np.zeros((0, 7, 6), dtype=np.float64)
            stack_length[key] = 0
            integration_tasks[key] = IntegrationTaskList()

            block_one, block_two = key.split(",")

        # Main loop
        for quadruple in quadruples:
            # Asses which blocks this particular quadruple will have statistics
            # for
            i, j, u, v = quadruple.singles
            first = (i, j)
            second = (u, v)

            valid_blocks = ["t,t"]
            if second in independent_elements["r"]:
                valid_blocks.append("t,r")
            if second in independent_elements["t2"]:
                valid_blocks.append("t,t2")
            if second in independent_elements["r2"]:
                valid_blocks.append("t,r2")

            if first in independent_elements["r"]:
                if second in independent_elements["r"]:
                    valid_blocks.append("r,r")
                if second in independent_elements["t2"]:
                    valid_blocks.append("r,t2")
                if second in independent_elements["r2"]:
                    valid_blocks.append("r,r2")

            if first in independent_elements["t2"]:
                if second in independent_elements["t2"]:
                    valid_blocks.append("t2,t2")
                if second in independent_elements["r2"]:
                    valid_blocks.append("t2,t2")

            if first in independent_elements["r2"]:
                if second in independent_elements["r2"]:
                    valid_blocks.append("r2,r2")

            # Check how long the stack will become if the new triangles
            # are added. If this length exceeds the limit, we begin
            # working on a new task
            new_simplices = quadruple.domain

            for block in valid_blocks:
                new_stack_length = stack_length[block] + len(new_simplices)

                if (
                    integrals_per_task is not None
                    and stack_length[block] > 0
                    and new_stack_length > integrals_per_task
                ):
                    new_task = IntegrationTask(
                        integrand[block],
                        simplex_stack[block],
                        statistic_type="covariance",
                        block_location=(wave_block, block),
                        sub_block_locations=sub_block_locations[block],
                        const_factor=const_factor,
                    )
                    integration_tasks[block].append_task(new_task)

                    # Reset the triangle stack and stack length
                    simplex_stack[block] = np.zeros(
                        (0, 7, 6), dtype=np.float64
                    )
                    stack_length[block] = 0
                    sub_block_locations[block] = []

                # Add location to sub_block_locations
                new_slice = slice(stack_length[block], new_stack_length)
                new_indices = (i, j, u, v)
                new_sub_block_location = (new_slice, new_indices)
                sub_block_locations[block].append(new_sub_block_location)

                # Add new triangles to stack
                simplex_stack[block] = np.vstack(
                    (simplex_stack[block], new_simplices)
                )
                stack_length[block] += len(new_simplices)

        # Once this point has been reached, we have exahusted all
        # triangles for a certing block of the scattering matrix
        # We now make the final task for the group
        for block in blocks:
            new_task = IntegrationTask(
                integrand[block],
                simplex_stack[block],
                statistic_type="covariance",
                block_location=(wave_block, block),
                sub_block_locations=sub_block_locations[block],
                const_factor=const_factor,
            )
            integration_tasks[block].append_task(new_task)
            main_task_list.merge_task_list(integration_tasks[block])

        return main_task_list

    # -------------------------------------------------------------------------
    # Pseudo covariance
    # -------------------------------------------------------------------------

    def _get_pseudo_covariance_integrand(
        self,
        wave_block_one: str,
        wave_block_two: str,
        block_one: str,
        block_two: str,
    ) -> MathematicalFunction:
        pseudo_covariance_a_matrix = (
            self.medium_statistics.get_pseudo_covariance_a_matrix()
        )
        k = self.medium_parameters.k
        L = self.medium_parameters.L

        def pseudo_covariance_integrand(
            k1_x: FloatLike,
            k1_y: FloatLike,
            k2_x: FloatLike,
            k2_y: FloatLike,
            d_x: FloatLike,
            d_y: FloatLike,
        ) -> FloatLike:
            # Convert to complex arrays
            k1_x = np.array(k1_x, dtype=complex)
            k1_y = np.array(k1_y, dtype=complex)
            k2_x = np.array(k2_x, dtype=complex)
            k2_y = np.array(k2_y, dtype=complex)
            d_x = np.array(d_x, dtype=complex)
            d_y = np.array(d_y, dtype=complex)

            ki_x = k1_x + d_x / 2
            ki_y = k1_y + d_y / 2

            kj_x = k1_x - d_x / 2
            kj_y = k1_y - d_y / 2

            ku_x = k2_x - d_x / 2
            ku_y = k2_y - d_y / 2

            kv_x = k2_x + d_x / 2
            kv_y = k2_y + d_y / 2

            ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
            kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
            ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
            kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

            # Sort out signs of k_iz and k_jz
            if block_one in {"r2", "t2"}:
                ki_z = -ki_z
            if block_two in {"r2", "t2"}:
                ku_z = -ku_z
            if block_one in {"t", "r2"}:
                kj_z = -kj_z
            if block_two in {"t", "r2"}:
                kv_z = -kv_z

            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z + ku_z - kv_z)
            )

            sec_factor = 1.0 / np.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

            output = (
                pseudo_covariance_a_matrix(
                    ki_x,
                    ki_y,
                    ki_z,
                    kj_x,
                    kj_y,
                    kj_z,
                    ku_x,
                    ku_y,
                    ku_z,
                    kv_x,
                    kv_y,
                    kv_z,
                )
                * sinc_factor
                * sec_factor
            )
            return output

        return pseudo_covariance_integrand

    def _get_pseudo_covariance_integration_tasks(
        self,
        quadruples,
        independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
    ) -> IntegrationTaskList:
        # Multiprocessing parameters
        num_quadruples = len(quadruples)
        num_processes = min(num_quadruples, os.cpu_count())

        # Prepare integrands since these can't be pickled
        blocks = [
            "t,t",
            "t,r",
            "t,t2",
            "t,r2",
            "r,r",
            "r,t2",
            "r,r2",
            "t2,t2",
            "t2,r2",
            "r2,r2",
        ]

        integrand = {}

        for key in blocks:
            block_one, block_two = key.split(",")
            integrand[key] = self._get_pseudo_covariance_integrand(
                "pp", "pp", block_one, block_two
            )

        parallelised_function = functools.partial(
            self._get_pseudo_covariance_integration_tasks_partial,
            independent_elements=independent_elements,
            const_factor=self.medium_parameters.cov_const_factor,
            integrals_per_task=self.integration_task_config.integrals_per_task,
            integrand=integrand,
        )

        partial_quadruples = array_utils.split_list(quadruples, num_processes)

        with ProcessPool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_quadruples)

        main_task_list = IntegrationTaskList()

        for task_list in out:
            main_task_list.merge_task_list(task_list)

        return main_task_list

    @staticmethod
    def _get_pseudo_covariance_integration_tasks_partial(
        quadruples,
        independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
        const_factor,
        integrals_per_task,
        integrand,
    ) -> IntegrationTaskList:
        """Main method for preparing covariance integral tasks"""

        # Factor common to all covariance integrals
        main_task_list = IntegrationTaskList()

        # wave block will look like "pp,pp", "pp,ep" etc.
        wave_block = "pp,pp"
        wave_block_one, wave_block_two = wave_block.split(",")

        # # The integrand depends only on the matrix blocks.

        # Variables that will be used for constructing tasks
        # These reset each time we go to a new block
        blocks = [
            "t,t",
            "t,r",
            "t,t2",
            "t,r2",
            "r,r",
            "r,t2",
            "r,r2",
            "t2,t2",
            "t2,r2",
            "r2,r2",
        ]

        # Initialise empty dictionaries for storing all the data that will be
        # used in the ensuing loop
        sub_block_locations = {}
        simplex_stack = {}
        stack_length = {}
        integration_tasks = {}

        for key in blocks:
            sub_block_locations[key] = []
            simplex_stack[key] = np.zeros((0, 7, 6), dtype=np.float64)
            stack_length[key] = 0
            integration_tasks[key] = IntegrationTaskList()

            block_one, block_two = key.split(",")

        # Main loop
        for quadruple in quadruples:
            # Asses which blocks this particular quadruple will have statistics
            # for
            i, j, u, v = quadruple.singles
            first = (i, j)
            second = (u, v)

            valid_blocks = ["t,t"]
            if second in independent_elements["r"]:
                valid_blocks.append("t,r")
            if second in independent_elements["t2"]:
                valid_blocks.append("t,t2")
            if second in independent_elements["r2"]:
                valid_blocks.append("t,r2")

            if first in independent_elements["r"]:
                if second in independent_elements["r"]:
                    valid_blocks.append("r,r")
                if second in independent_elements["t2"]:
                    valid_blocks.append("r,t2")
                if second in independent_elements["r2"]:
                    valid_blocks.append("r,r2")

            if first in independent_elements["t2"]:
                if second in independent_elements["t2"]:
                    valid_blocks.append("t2,t2")
                if second in independent_elements["r2"]:
                    valid_blocks.append("t2,t2")

            if first in independent_elements["r2"]:
                if second in independent_elements["r2"]:
                    valid_blocks.append("r2,r2")

            # Check how long the stack will become if the new triangles
            # are added. If this length exceeds the limit, we begin
            # working on a new task
            new_simplices = quadruple.domain

            for block in valid_blocks:
                new_stack_length = stack_length[block] + len(new_simplices)

                if (
                    integrals_per_task is not None
                    and stack_length[block] > 0
                    and new_stack_length > integrals_per_task
                ):
                    new_task = IntegrationTask(
                        integrand[block],
                        simplex_stack[block],
                        statistic_type="pseudo_covariance",
                        block_location=(wave_block, block),
                        sub_block_locations=sub_block_locations[block],
                        const_factor=const_factor,
                    )
                    integration_tasks[block].append_task(new_task)

                    # Reset the triangle stack and stack length
                    simplex_stack[block] = np.zeros(
                        (0, 7, 6), dtype=np.float64
                    )
                    stack_length[block] = 0
                    sub_block_locations[block] = []

                # Add location to sub_block_locations
                new_slice = slice(stack_length[block], new_stack_length)
                new_indices = (i, j, -v, -u)
                new_sub_block_location = (new_slice, new_indices)
                sub_block_locations[block].append(new_sub_block_location)

                # Add new triangles to stack
                simplex_stack[block] = np.vstack(
                    (simplex_stack[block], new_simplices)
                )
                stack_length[block] += len(new_simplices)

        # Once this point has been reached, we have exahusted all
        # triangles for a certing block of the scattering matrix
        # We now make the final task for the group
        for block in blocks:
            new_task = IntegrationTask(
                integrand[block],
                simplex_stack[block],
                statistic_type="pseudo_covariance",
                block_location=(wave_block, block),
                sub_block_locations=sub_block_locations[block],
                const_factor=const_factor,
            )
            integration_tasks[block].append_task(new_task)
            main_task_list.merge_task_list(integration_tasks[block])

        return main_task_list
