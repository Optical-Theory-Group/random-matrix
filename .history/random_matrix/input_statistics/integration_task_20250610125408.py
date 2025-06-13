import functools
import os
import time
from dataclasses import dataclass, field
from typing import Self, Any
import tqdm
import numpy as np
import cupy as cp
import scipy
from pathos.pools import ProcessPool
import psutil

from random_matrix.input_statistics import (
    density_integrals,
    input_statistics_logger,
    medium_parameters,
    medium_statistics,
    shape_classifier,
)
from random_matrix.input_statistics.shape_classifier import ClassQuadrupleList
from random_matrix.modes import mode_grid
from random_matrix.utils import (
    array_utils,
    function_utils,
    geometry_utils,
    integration_utils,
    special_functions,
)
from random_matrix.utils.types import Numeric, MathematicalFunction


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
    integral: Numeric

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

    def merge_result_list(self, new_result_list: Self) -> None:
        """Append results from another result list object to itself"""

        self.results += new_result_list.results

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
    domain_stack: Numeric
    statistic_type: str
    block_location: tuple[str, str]
    sub_block_locations: list[tuple[slice, tuple[int, int]]]
    const_factor: Numeric = 1.0

    def __str__(self) -> str:
        string = (
            f"Statistic type: {self.statistic_type}\n"
            f"Matrix block: {self.block_location}\n"
            f"Number of integrals: {len(self.domain_stack)}"
        )

        return string

    def execute_task(
        self, cubature_scheme: Any | None = None
    ) -> IntegrationResult:
        """Perofrm the integral associated with the task."""

        match self.statistic_type:
            case "mean":
                integral = integration_utils.simplex_integral(
                    self.integrand, self.domain_stack
                )
            case "covariance":
                integral = integration_utils.simplex_integral(
                    self.integrand, self.domain_stack, cubature_scheme
                )
            case "pseudo_covariance":
                integral = integration_utils.simplex_integral(
                    self.integrand, self.domain_stack, cubature_scheme
                )

        xp = cp.get_array_module(integral)

        # Results should be added over the slices. This corresponds to adding
        # integrals over the triangular subregions of the integration domain
        num_outputs, output_dim = xp.shape(integral)
        slices = [s[0] for s in self.sub_block_locations]
        locations = [s[1] for s in self.sub_block_locations]
        num_slices = len(slices)
        output = xp.zeros((num_slices, output_dim), dtype=xp.complex128)

        for i, s in enumerate(slices):
            partial_output = xp.sum(integral[s, :], axis=0)
            output[i, :] = partial_output

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
        # Multiprocessing parameters
        num_tasks = len(self.tasks)

        num_processes = min(num_tasks, os.cpu_count())
        parallelised_function = functools.partial(
            self._execute_tasks_partial,
            progress_bar=tqdm.tqdm,
        )

        partial_tasks = array_utils.split_list(self.tasks, num_processes)

        with ProcessPool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_tasks)

        results_list = IntegrationResultList()
        for partial_result_list in out:
            for result in partial_result_list.results:
                results_list.append_result(result)

        return results_list

    @staticmethod
    def _execute_tasks_partial(tasks, progress_bar):
        results_list = IntegrationResultList()
        for task in progress_bar(tasks):
            new_result = task.execute_task()
            results_list.append_result(new_result)
        return results_list

    # def execute_tasks(self) -> IntegrationResultList:
    #     """Perofrm the integral associated with all tasks."""

    #     results_list = IntegrationResultList()
    #     for task in tqdm.tqdm(self.tasks):
    #         new_result = task.execute_task()
    #         results_list.append_result(new_result)
    #     return results_list


@dataclass(slots=True)
class IntegrationTaskConfig:
    """Configuration metadata for controlling the degree of parallelisation of the
    integration tasks

    Attributes
    ----------
    integrals_per_task:
        A limit for how many integration domains should be allowed in each
        task. If None, there will be no limit.
    cpu_ram_limit:
        Maximum percentage that the CPU's RAM is allowed to reach before
        integration tasks will automatically be executed to free up space
    """

    integrals_per_task: int | None = 1
    ram_limit: float = 0.0
    use_gpu: bool = False

    def get_current_ram_usage(self, verbose: bool = False) -> float:
        if self.use_gpu:
            current_ram_usage = (
                cp.get_default_memory_pool().used_bytes()
                / cp.get_default_memory_pool().total_bytes()
                * 100
            )
        else:
            current_ram_usage = psutil.virtual_memory().percent
        if verbose:
            using = "GPU" if self.use_gpu else "CPU"
            print(f"{using} RAM: {current_ram_usage}")
        return current_ram_usage


class IntegrationTaskPreparer:
    def __init__(
        self,
        mode_grid: mode_grid.ModeGrid,
        medium_parameters: medium_parameters.MediumParameters,
        medium_statistics: medium_statistics.MediumStatistics,
        logger: input_statistics_logger.InputStatisticsLogger,
        integration_task_config: IntegrationTaskConfig = (
            IntegrationTaskConfig()
        ),
    ):
        self.integration_task_config = integration_task_config
        self.mode_grid = mode_grid
        self.medium_parameters = medium_parameters
        self.medium_statistics = medium_statistics
        self.logger = logger

    def get_integration_results(
        self,
        class_quadruple_list: ClassQuadrupleList,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
        covariance_cubature_scheme: Any | None = None,
        use_dirac_density: bool = True,
    ) -> IntegrationTaskList:
        """Get an integration task list consisting of all necessary tasks.

        This is the main function that is run by IntegrationTaskPreparer
        """

        master_result_list = IntegrationResultList()

        # with self.logger.log("mean"):
        #     master_result_list.merge_result_list(
        #         self._get_mean_integration_results(indices["mean"])
        #     )

        with self.logger.log("covariance"):
            if use_dirac_density:
                master_result_list.merge_result_list(
                    self._get_covariance_integration_results_dirac_density(
                        class_quadruple_list,
                        indices["covariance"],
                        covariance_cubature_scheme=cubature_scheme,
                    )
                )
            else:
                master_result_list.merge_result_list(
                    self._get_covariance_integration_results(
                        class_quadruple_list,
                        indices["covariance"],
                        covariance_cubature_scheme=cubature_scheme,
                    )
                )

        # with self.logger.log("pseudo_covariance"):
        #     master_result_list.merge_result_list(
        #         self._get_pseudo_covariance_integration_tasks(
        #             quadruples, independent_elements["pp"]
        #         )
        #     )

        return master_result_list

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

        # Sort out signs of k_iz and k_jz
        ki_z_factor = -1.0 if block in {"r2", "t2"} else 1.0
        kj_z_factor = -1.0 if block in {"r", "t2"} else 1.0
        sinc_factor = (
            special_functions.sinc
            if block in {"r", "r2"}
            else special_functions.identity
        )

        def mean_integrand(
            integration_domain: np.ndarray | cp.ndarray,
        ) -> np.ndarray | cp.ndarray:
            """The integrand should be of shape N x 2, where N is the number
            of points that need to be evaluated. The final dimension is
            k_x and k_y"""
            xp = cp.get_array_module(integration_domain)
            k_x, k_y = integration_domain.T

            ki_x = k_x
            ki_y = k_y
            kj_x = k_x
            kj_y = k_y

            partial = xp.sqrt(1.0 - k_x**2 - k_y**2)
            ki_z = ki_z_factor * partial
            kj_z = kj_z_factor * partial

            # SHOULD THIS BE NEGATIVE OR DO I NEED TO USE ABS???
            sec_factor = 1.0 / ki_z
            output = (
                mean_a_matrix(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z)
                * sinc_factor(k * L * ki_z)[:, xp.newaxis]
                * sec_factor[:, xp.newaxis]
            )
            return output

        return mean_integrand

    def _get_mean_integration_results(
        self, mean_indices: dict[str, dict[str, tuple[int, int]]]
    ) -> IntegrationResultList:
        """Main method for computing mean results"""

        # Factor common to all mean integrals
        const_factor = self.medium_parameters.mean_const_factor
        main_result_list = IntegrationResultList()

        # Prepare the master index set, which contains all possible indices
        # for which a mean will need to be calculated
        master_index_set = set()
        for wave_block, d in mean_indices.items():
            for block, i in d.items():
                master_index_set.update(i)

        # Prepare task dictionary
        wave_blocks = ["pp", "pe", "ep", "ee"]
        blocks = ["t", "r", "t2", "r2"]
        task_dict = {}
        for wave_block in wave_blocks:
            task_dict[wave_block] = {}
            for block in blocks:
                task_dict[wave_block][block] = IntegrationTask(
                    self._get_mean_integrand(wave_block, block),
                    np.zeros((0, 3, 2), dtype=np.float64),
                    statistic_type="mean",
                    block_location=(wave_block, block),
                    sub_block_locations=[],
                    const_factor=const_factor,
                )

        for indices in self.logger.progress_bar(master_index_set):
            # Get integration domain. We can do this at this stage because,
            # given that indices exists, it must exist in at least one of the
            # sets in master_index_set.
            index = indices[0]
            mode = self.mode_grid.by_index(index)
            delaunay = scipy.spatial.Delaunay(mode.vertices)
            new_triangles = mode.vertices[delaunay.simplices]

            for wave_block in wave_blocks:
                for block in blocks:
                    # Check if the mean needs to be calculated for this
                    # particular wave_block, block pair
                    if indices not in mean_indices.get(wave_block, {}).get(
                        block, set()
                    ):
                        continue

                    # Add domain to integral task
                    old_stack_length = len(
                        task_dict[wave_block][block].domain_stack
                    )
                    new_stack_length = old_stack_length + len(new_triangles)
                    new_slice = slice(old_stack_length, new_stack_length)
                    new_indices = indices
                    new_sub_block_location = (new_slice, new_indices)
                    task_dict[wave_block][block].sub_block_locations.append(
                        new_sub_block_location
                    )

                    # Add new triangles to stack
                    task_dict[wave_block][block].domain_stack = np.vstack(
                        (
                            task_dict[wave_block][block].domain_stack,
                            new_triangles,
                        )
                    )

                    # Execute tasks if RAM usage is getting too high.
                    # Re-initialize relevant integration task
                    current_cpu_ram_usage = psutil.virtual_memory().percent
                    if (
                        current_cpu_ram_usage
                        > self.integration_task_config.ram_limit
                    ):
                        new_result = task_dict[wave_block][
                            block
                        ].execute_task()
                        main_result_list.append_result(new_result)
                        task_dict[wave_block][block] = IntegrationTask(
                            self._get_mean_integrand(wave_block, block),
                            np.zeros((0, 3, 2), dtype=np.float64),
                            statistic_type="mean",
                            block_location=(wave_block, block),
                            sub_block_locations=[],
                            const_factor=const_factor,
                        )

        # The tasks have all been prepared. Now execute them!
        for wave_block in wave_blocks:
            for block in blocks:
                new_result = task_dict[wave_block][block].execute_task()
                main_result_list.append_result(new_result)

        return main_result_list

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

        ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
        kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
        ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
        kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

        def covariance_integrand(
            integration_domain: np.ndarray | cp.ndarray,
        ) -> np.ndarray | cp.ndarray:
            """The integrand should be of shape N x 6, where N is the number
            of points that need to be evaluated. The final dimension is
            ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
            xp = cp.get_array_module(integration_domain)
            ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T

            kv_x = -ki_x + kj_x + ku_x
            kv_y = -ki_y + kj_y + ku_y

            ki_z = ki_z_factor * xp.sqrt(1 - ki_x**2 - ki_y**2)
            kj_z = kj_z_factor * xp.sqrt(1 - kj_x**2 - kj_y**2)
            ku_z = ku_z_factor * xp.sqrt(1 - ku_x**2 - ku_y**2)
            kv_z = kv_z_factor * xp.sqrt(1 - kv_x**2 - kv_y**2)

            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )
            sec_factor = 1.0 / xp.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

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
                * sinc_factor[:, xp.newaxis]
                * sec_factor[:, xp.newaxis]
            )

            return output

        return covariance_integrand

    def _get_covariance_integrand_dirac_density(
        self,
        wave_block_one: str,
        wave_block_two: str,
        block_one: str,
        block_two: str,
        kj_centroid: np.ndarray | cp.ndarray,
        kv_centroid: np.ndarray | cp.ndarray,
    ) -> MathematicalFunction:
        covariance_a_matrix = self.medium_statistics.get_covariance_a_matrix()
        k = self.medium_parameters.k
        L = self.medium_parameters.L

        ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
        kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
        ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
        kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0

        def covariance_integrand(
            integration_domain: np.ndarray | cp.ndarray,
        ) -> np.ndarray | cp.ndarray:
            """The integrand should be of shape N x 2, where N is the number
            of points that need to be evaluated. The final dimension is
            ki_x, ki_y"""
            xp = cp.get_array_module(integration_domain)
            ki_x, ki_y = integration_domain.T
            num_entries = len(ki_x)

            kj_x = xp.repeat(kj_centroid[0], num_entries)
            kj_y = xp.repeat(kj_centroid[1], num_entries)
            kv_x = xp.repeat(kv_centroid[0], num_entries)
            kv_y = xp.repeat(kv_centroid[1], num_entries)

            ku_x = ki_x - kj_x + kv_x
            ku_y = ki_y - kj_y + kv_y

            ki_z = ki_z_factor * xp.sqrt(1 - ki_x**2 - ki_y**2)
            kj_z = kj_z_factor * xp.sqrt(1 - kj_x**2 - kj_y**2)
            ku_z = ku_z_factor * xp.sqrt(1 - ku_x**2 - ku_y**2)
            kv_z = kv_z_factor * xp.sqrt(1 - kv_x**2 - kv_y**2)

            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )
            sec_factor = 1.0 / xp.abs(np.sqrt(ki_z * kj_z * ku_z * kv_z))

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
                * sinc_factor[:, xp.newaxis]
                * sec_factor[:, xp.newaxis]
            )

            return output

        return covariance_integrand

    def _get_covariance_integration_results(
        self,
        class_quadruple_list: ClassQuadrupleList,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
        covariance_cubature_scheme: Any | None = None,
    ) -> IntegrationResultList:
        """Main method for computing covariance results"""
        start = time.perf_counter()
        xp = cp if self.integration_task_config.use_gpu else np

        # Factor common to all covariance integrals
        const_factor = self.medium_parameters.cov_const_factor
        main_result_list = IntegrationResultList()

        # Prepare the master index set, which contains all possible indices
        # for which a covariance will need to be calculated
        master_index_set = set()
        for wave_block, d in covariance_indices.items():
            for block, i in d.items():
                master_index_set.update(i)

        # Prepare task dictionary
        wave_blocks = [
            "pp,pp",
            "pp,pe",
            "pp,ep",
            "pp,ee",
            "pe,pe",
            "pe,ep",
            "pe,ee",
            "ep,ep",
            "ep,ee",
            "ee,ee",
        ]
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

        # Subset for testing purposes
        task_dict = {}
        for wave_block in wave_blocks:
            wave_block_one, wave_block_two = wave_block.split(",")
            task_dict[wave_block] = {}
            for block in blocks:
                block_one, block_two = block.split(",")

                task_dict[wave_block][block] = IntegrationTask(
                    self._get_covariance_integrand(
                        wave_block_one, wave_block_two, block_one, block_two
                    ),
                    xp.zeros((0, 7, 6), dtype=xp.float64),
                    statistic_type="covariance",
                    block_location=(wave_block, block),
                    sub_block_locations=[],
                    const_factor=const_factor,
                )

        # Hyperplane normals
        n1 = xp.array([1, 0, -1, 0, -1, 0, 1, 0])
        n2 = xp.array([0, 1, 0, -1, 0, -1, 0, 1])
        filter_one = [0, 1, 2, 3, 4, 5, 7]
        filter_two = [0, 1, 2, 3, 4, 5]
        columns_to_keep = [0, 1, 2, 3, 4, 5]

        # Main loop
        for class_number, class_quadruple in enumerate(
            self.logger.progress_bar(class_quadruple_list.classes)
        ):
            # Work out the template's integration domain
            # This involves the higher dimensional geometry
            template = class_quadruple.template
            cartesian_product = template.vertices

            # Uniform density
            start = time.perf_counter()
            reduced_intersection = geometry_utils.get_intersection_vertices(
                cartesian_product
            )
            reduced_hull = scipy.spatial.ConvexHull(
                reduced_intersection, qhull_options="QJ"
            )
            end = time.perf_counter()

            # Get the centroid and interweave it into all the simplices
            centroid = xp.mean(
                xp.asarray(reduced_hull.points[reduced_hull.vertices]), axis=0
            )
            centroid_expanded = xp.tile(
                centroid, (reduced_hull.simplices.shape[0], 1, 1)
            )
            new_simplices = xp.concatenate(
                [
                    xp.asarray(reduced_hull.points[reduced_hull.simplices]),
                    centroid_expanded,
                ],
                axis=1,
            )

            for quadruple in class_quadruple.quadruples:
                indices = quadruple.singles_indices
                new_domain = new_simplices + xp.asarray(
                    quadruple.translation_vector[columns_to_keep]
                )
                for wave_block in wave_blocks:
                    wave_block_one, wave_block_two = wave_block.split(",")
                    for block in blocks:
                        block_one, block_two = block.split(",")
                        # Check if the mean needs to be calculated for this
                        # particular wave_block, block pair
                        if indices not in covariance_indices.get(
                            wave_block, {}
                        ).get(block, set()):
                            continue

                        # Add domain to integral task
                        old_stack_length = len(
                            task_dict[wave_block][block].domain_stack
                        )
                        new_stack_length = old_stack_length + len(new_domain)
                        new_slice = slice(old_stack_length, new_stack_length)
                        new_indices = indices
                        new_sub_block_location = (new_slice, new_indices)
                        task_dict[wave_block][
                            block
                        ].sub_block_locations.append(new_sub_block_location)

                        # Add new triangles to stack
                        task_dict[wave_block][block].domain_stack = xp.vstack(
                            (
                                task_dict[wave_block][block].domain_stack,
                                new_domain,
                            )
                        )

                        # Execute tasks if RAM usage is getting too high.
                        # Re-initialize relevant integration task
                        current_ram_usage = (
                            self.integration_task_config.get_current_ram_usage()
                        )
                        if class_number % 100 == 0:
                            new_result = task_dict[wave_block][
                                block
                            ].execute_task(covariance_cubature_scheme)
                            # print("Integral done.")
                            main_result_list.append_result(new_result)
                            task_dict[wave_block][block] = IntegrationTask(
                                self._get_covariance_integrand(
                                    wave_block_one,
                                    wave_block_two,
                                    block_one,
                                    block_two,
                                ),
                                xp.zeros((0, 7, 6), dtype=xp.float64),
                                statistic_type="covariance",
                                block_location=(wave_block, block),
                                sub_block_locations=[],
                                const_factor=const_factor,
                            )

        # The tasks have all been prepared. Now execute them!

        for wave_block in wave_blocks:
            for block in blocks:
                new_result = task_dict[wave_block][block].execute_task(
                    covariance_cubature_scheme
                )
                main_result_list.append_result(new_result)
        end = time.perf_counter()
        return main_result_list

    def _get_covariance_integration_results_dirac_density(
        self,
        class_quadruple_list: ClassQuadrupleList,
        covariance_indices: dict[str, dict[str, tuple[int, int, int, int]]],
        cubature_scheme: Any | None = None,
    ) -> IntegrationResultList:
        """Main method for computing covariance results"""
        start = time.perf_counter()
        xp = cp if self.integration_task_config.use_gpu else np

        # Factor common to all covariance integrals
        const_factor = self.medium_parameters.cov_const_factor
        main_result_list = IntegrationResultList()

        # Prepare the master index set, which contains all possible indices
        # for which a covariance will need to be calculated
        master_index_set = set()
        for wave_block, d in covariance_indices.items():
            for block, i in d.items():
                master_index_set.update(i)

        # Prepare task dictionary
        wave_blocks = [
            "pp,pp",
            "pp,pe",
            "pp,ep",
            "pp,ee",
            "pe,pe",
            "pe,ep",
            "pe,ee",
            "ep,ep",
            "ep,ee",
            "ee,ee",
        ]
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

        # Hyperplane normals
        n1 = xp.array([1, 0, -1, 0, -1, 0, 1, 0])
        n2 = xp.array([0, 1, 0, -1, 0, -1, 0, 1])
        filter_one = [0, 1, 2, 3, 4, 5, 7]
        filter_two = [0, 1, 2, 3, 4, 5]
        columns_to_keep = [0, 1]

        # Main loop
        for class_number, class_quadruple in enumerate(
            self.logger.progress_bar(class_quadruple_list.classes)
        ):
            # Work out the template's integration domain
            # This involves the higher dimensional geometry
            template = class_quadruple.template
            cartesian_product = template.vertices

            # Dirac density
            _, j, _, v = class_quadruple.template.singles_indices
            kj = self.mode_grid.by_index(j).center
            kv = self.mode_grid.by_index(v).center
            intersection = (
                geometry_utils.get_intersection_vertices_dirac_density(
                    cartesian_product, kj, kv
                )
            )
            reduced_region = intersection[:, columns_to_keep]
            delaunay = scipy.spatial.Delaunay(reduced_region)
            new_simplices = xp.asarray(delaunay.points[delaunay.simplices])

            for quadruple in class_quadruple.quadruples:
                indices = quadruple.singles_indices
                new_domain = new_simplices + xp.asarray(
                    quadruple.translation_vector[columns_to_keep]
                )
                _, j, _, v = quadruple.singles_indices
                kj = self.mode_grid.by_index(j).center
                kv = self.mode_grid.by_index(v).center

                new_stack_length = len(new_domain)
                new_slice = slice(0, new_stack_length)
                new_indices = indices
                new_sub_block_location = (new_slice, new_indices)

                for wave_block in wave_blocks:
                    wave_block_one, wave_block_two = wave_block.split(",")
                    for block in blocks:
                        block_one, block_two = block.split(",")

                        # Check if the mean needs to be calculated for this
                        # particular wave_block, block pair
                        if indices not in covariance_indices.get(
                            wave_block, {}
                        ).get(block, set()):
                            continue

                        new_task = IntegrationTask(
                            self._get_covariance_integrand_dirac_density(
                                wave_block_one,
                                wave_block_two,
                                block_one,
                                block_two,
                                xp.asarray(kj),
                                xp.asarray(kv),
                            ),
                            new_domain,
                            statistic_type="covariance",
                            block_location=(wave_block, block),
                            sub_block_locations=[new_sub_block_location],
                            const_factor=const_factor,
                        )
                        new_result = new_task.execute_task(cubature_scheme)
                        main_result_list.append_result(new_result)

        return main_result_list

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
            k1_x: Numeric,
            k1_y: Numeric,
            k2_x: Numeric,
            k2_y: Numeric,
            d_x: Numeric,
            d_y: Numeric,
        ) -> Numeric:
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
            progress_bar=self.logger.progress_bar,
        )

        partial_quadruples = array_utils.split_list(quadruples, num_processes)

        with ProcessPool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_quadruples)

        main_result_list = IntegrationResultList()

        for result_list in out:
            main_result_list.merge_result_list(result_list)

        return main_result_list

    @staticmethod
    def _get_pseudo_covariance_integration_tasks_partial(
        quadruples,
        independent_elements: dict[str, dict[str, tuple[int, int, int, int]]],
        const_factor,
        integrals_per_task,
        integrand,
        progress_bar,
    ) -> IntegrationTaskList:
        """Main method for preparing covariance integral tasks"""

        # Factor common to all covariance integrals
        main_result_list = IntegrationResultList()

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
        integration_results = {}

        for key in blocks:
            sub_block_locations[key] = []
            simplex_stack[key] = np.zeros((0, 7, 6), dtype=np.float64)
            stack_length[key] = 0
            integration_results[key] = IntegrationResultList()
            block_one, block_two = key.split(",")

        # Main loop
        for quadruple in progress_bar(quadruples):
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
                    new_result = new_task.execute_task()
                    integration_results[block].append_result(new_result)

                    # Reset the triangle stack and stack length
                    simplex_stack[block] = np.zeros(
                        (0, 7, 6), dtype=np.float64
                    )
                    stack_length[block] = 0
                    sub_block_locations[block] = []

                # Add location to sub_block_locations
                new_slice = slice(stack_length[block], new_stack_length)
                new_indices = (i, j, -u, -v)
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
            new_result = new_task.execute_task()
            integration_results[block].append_result(new_result)
            main_result_list.merge_result_list(integration_results[block])

        return main_result_list
