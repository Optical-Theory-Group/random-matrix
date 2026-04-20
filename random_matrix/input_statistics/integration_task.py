import functools
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self, Any, Callable, Generator
import tqdm
import numpy as np
import cupy as cp
import scipy
from pathos.pools import ProcessPool
import psutil
import multiprocessing as mp
import h5py
from pathlib import Path
import pickle

from random_matrix.input_statistics import (
    input_statistics_logger,
    medium_parameters,
    medium_statistics,
)
from random_matrix.modes import mode_grid
from random_matrix.utils import (
    array_utils,
    geometry_utils,
    integration_utils,
    special_functions,
    system_utils,
)
from random_matrix.utils.types import Numeric, MathematicalFunction

ALLOWED_INTEGRATION_METHODS = ["cubature", "midpoint", "lattice"]
ALLOWED_STATISTIC_TYPES = ["mean", "covariance", "pseudo_covariance"]
WAVE_BLOCKS = [
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
BLOCKS = [
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

    def __post_init__(self) -> None:
        if self.statistic_type not in ALLOWED_STATISTIC_TYPES:
            raise ValueError(
                f"Invalid statistic type: {self.satistic_type}. "
                f"Pick from {ALLOWED_STATISTIC_TYPES}"
            )

    # def __str__(self) -> str:
    #     string = (
    #         f"Statistic type: {self.statistic_type}\n"
    #         f"Matrix block: {self.block_location}\n"
    #         f"Number of integrals: {len(self.integral)}"
    #     )

    #     return string


@dataclass(slots=True)
class IntegrationResultList:
    """Container class for holding many IntegrationResult objects."""

    results: list[IntegrationResult] = field(default_factory=list)

    # def __str__(self) -> str:
    #     string = f"Number of results: {len(self.results)}\n" f"Results key:\n"

    #     for i, result in enumerate(self.results):
    #         string += (
    #             f"{i}, {result.statistic_type}, "
    #             f"{len(result.integral)} integrals, "
    #             f"{result.block_location}\n"
    #         )

    #     return string

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

    def get(
        self,
        statistic_type: str = "covariance",
        wave_block: str = "pp,pp",
        block: str = "t,t",
        sub_block: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> IntegrationResult | None:
        """Get a statistic from the list based on desired properties"""
        for result in self.results:
            is_correct = (
                result.statistic_type == statistic_type
                and result.block_location == (wave_block, block)
                and sub_block in result.sub_block_locations
            )
            if is_correct:
                return result
        return None


@dataclass(slots=True, kw_only=True)
class IntegrationTask(ABC):
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
    statistic_type: str
    block_location: tuple[str, str]
    sub_block_locations: list[tuple[slice, tuple[int, int]]]

    const_factor: Numeric = 1.0
    extra_factor: Numeric = 1.0
    use_cupy: bool = False

    @property
    def get_xp(self):
        return cp if self.use_cupy else np

    def __str__(self) -> str:
        string = (
            f"Statistic type: {self.statistic_type}\n"
            f"Matrix block: {self.block_location}\n"
            f"Number of integrals: {self.num_integrals()}"
        )

        return string

    def execute_task(self) -> IntegrationResult:
        """Perofrm the integral associated with the task."""
        xp = self.get_xp
        integral = self.compute_integral()

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
            self.const_factor * self.extra_factor * output,
        )

    @abstractmethod
    def compute_integral(self) -> Numeric:
        pass

    @abstractmethod
    def num_integrals(self) -> int:
        pass


@dataclass(slots=True, kw_only=True)
class MidpointIntegrationTask(IntegrationTask):
    midpoint_array: np.ndarray | None = None
    volume_array: np.ndarray | None = None
    is_eight_dimensions: bool = True

    def __post_init__(self):
        """Initialize arrays"""
        xp = self.get_xp
        if self.midpoint_array is None:
            num_dims = 8 if self.is_eight_dimensions else 6
            self.midpoint_array = xp.zeros((0, num_dims), dtype=xp.float64)
        if self.volume_array is None:
            self.volume_array = xp.zeros((0, 1), dtype=xp.float64)

    def compute_integral(self) -> Numeric:
        return integration_utils.midpoint_integral(
            self.integrand, self.midpoint_array, self.volume_array
        )

    def num_integrals(self) -> int:
        return len(self.midpoint_array)


@dataclass(slots=True)
class CubatureIntegrationTask(IntegrationTask):
    simplex_array: np.ndarray | None = None
    cubature_scheme: Any = None
    use_dirac_density: bool = False

    def __post_init__(self):
        """Validate simplex shape based on use_dirac_density flag"""
        xp = self.get_xp
        if self.statistic_type == "mean":
            expected_shape = (3, 2)
        else:
            expected_shape = (3, 2) if self.use_dirac_density else (7, 6)

        if self.simplex_array is None:
            num_vertices, num_dimensions = expected_shape
            self.simplex_array = xp.zeros(
                (0, *expected_shape), dtype=xp.float64
            )

        # Validate dimensionality
        if self.simplex_array.ndim != 3:
            raise ValueError(
                f"Expected simplex_array to have 3 dimensions (N, {expected_shape[0]}, {expected_shape[1]}), "
                f"but got shape {self.simplex_array.shape} with {self.simplex_array.ndim} dimensions."
            )

        num_simplices, num_vertices, num_dimensions = self.simplex_array.shape
        if (num_vertices, num_dimensions) != expected_shape:
            raise ValueError(
                f"Invalid simplex shape: expected each simplex to have shape {expected_shape} "
                f"(i.e., {expected_shape[0]} vertices in {expected_shape[1]}D), "
                f"but got ({num_vertices} vertices in {num_dimensions}D). "
                f"Full array shape is {self.simplex_array.shape}."
            )

    def compute_integral(self) -> Numeric:
        return integration_utils.simplex_integral(
            self.integrand, self.simplex_array, self.cubature_scheme
        )

    def num_integrals(self) -> int:
        return len(self.simplex_array)


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
    ram_limit: float = 0
    use_gpu: bool = False
    covariance_cubature_scheme: Any | None = (None,)
    integration_method: str = "lattice"
    include_inter_block_correlations: bool = False

    def __post_init__(self):
        if self.integration_method not in ALLOWED_INTEGRATION_METHODS:
            raise ValueError(
                f"Invalid method: {self.integration_method}. Pick from "
                f"{ALLOWED_INTEGRATION_METHODS}"
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
                f"{task.num_integrals()} triangles, "
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

    def _get_covariance_method_and_kwargs(
        self, indices: dict[str, dict[str, set[tuple[int, int]]]]
    ) -> tuple[Callable, dict]:
        """Choose the correct integration method based upon the config"""
        integration_method = self.integration_task_config.integration_method
        covariance_cubature_scheme = (
            self.integration_task_config.covariance_cubature_scheme
        )

        match integration_method:
            case "lattice":
                return self._get_covariance_results_lattice, {
                    "covariance_indices": indices["covariance"]
                }
            case "cubature" | "midpoint":
                return self._get_covariance_integration_results_parallelized, {
                    "integration_method": integration_method,
                    "covariance_indices": indices["covariance"],
                    "covariance_cubature_scheme": covariance_cubature_scheme,
                }

    def get_integration_results(
        self,
        indices: dict[str, dict[str, set[tuple[int, int]]]],
    ) -> IntegrationTaskList:
        """Get an integration result list consisting of all required
        statistics."""
        master_result_list = IntegrationResultList()

        # The mean is quick, so always use a cubature based method
        with self.logger.log("mean"):
            master_result_list.merge_result_list(
                self._get_mean_integration_results(indices["mean"])
            )

        # # The covariance method needs be to chosen appropriately for the type
        # # of simulation being performed
        # covariance_method, covariance_kwargs = (
        #     self._get_covariance_method_and_kwargs(indices)
        # )
        # with self.logger.log("covariance"):
        #     master_result_list.merge_result_list(
        #         covariance_method(**covariance_kwargs)
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

                block_location = (wave_block, block)
                integrand = self._get_mean_integrand(wave_block, block)

                task = build_integration_task(
                    integration_method="cubature",
                    integrand=integrand,
                    statistic_type="mean",
                    block_location=block_location,
                    sub_block_locations=[],
                    const_factor=const_factor,
                    use_cupy=False,
                    cubature_scheme=None,
                    use_dirac_density=False,
                )
                task_dict[wave_block][block] = task

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
                        task_dict[wave_block][block].simplex_array
                    )
                    new_stack_length = old_stack_length + len(new_triangles)
                    new_slice = slice(old_stack_length, new_stack_length)
                    new_indices = indices
                    new_sub_block_location = (new_slice, new_indices)
                    task_dict[wave_block][block].sub_block_locations.append(
                        new_sub_block_location
                    )

                    # Add new triangles to stack
                    task_dict[wave_block][block].simplex_array = np.vstack(
                        (
                            task_dict[wave_block][block].simplex_array,
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

                        block_location = (wave_block, block)
                        integrand = self._get_mean_integrand(wave_block, block)

                        task = build_integration_task(
                            integration_method="cubature",
                            integrand=integrand,
                            statistic_type="mean",
                            block_location=block_location,
                            sub_block_locations=[],
                            const_factor=const_factor,
                            use_cupy=False,
                            cubature_scheme=None,
                            use_dirac_density=False,
                        )
                        task_dict[wave_block][block] = task

        # The tasks have all been prepared. Now execute them!
        for wave_block in wave_blocks:
            for block in blocks:
                new_result = task_dict[wave_block][block].execute_task()
                main_result_list.append_result(new_result)

        return main_result_list

    # -------------------------------------------------------------------------
    # Covariance
    # -------------------------------------------------------------------------

    def calculate_a_matrix_values(self, path: Path) -> None:
        """Pre-compute A matrix values for faster computation of statistics
        later on"""
        num_modes = self.mode_grid.num_propagating
        mean_mode_vertices_dict = (
            self.mode_grid.propagating_modes_mean_vertices_dict
        )
        mean_mode_vertices_array = np.stack(
            list(mean_mode_vertices_dict.values())
        )
        ki_array = np.repeat(mean_mode_vertices_array, num_modes, axis=0)
        ki_x_array, ki_y_array = ki_array[:, 0], ki_array[:, 1]
        ki_z_array = np.sqrt(1.0 - ki_x_array**2 - ki_y_array**2)
        kj_array = np.tile(mean_mode_vertices_array, (num_modes, 1))
        kj_x_array, kj_y_array = kj_array[:, 0], kj_array[:, 1]
        kj_z_array = np.sqrt(1.0 - kj_x_array**2 - kj_y_array**2)

        # A matrix values. p and m refere to positive or negative signs on
        # k_z
        get_A = self.medium_statistics.get_mean_a_matrix()
        A_matrix_values_pp = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_pm = get_A(
            ki_x_array,
            ki_y_array,
            ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )
        A_matrix_values_mp = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            kj_z_array,
        )
        A_matrix_values_mm = get_A(
            ki_x_array,
            ki_y_array,
            -ki_z_array,
            kj_x_array,
            kj_y_array,
            -kj_z_array,
        )
        with h5py.File(path, "w") as f:
            f.create_dataset("A_pp", data=A_matrix_values_pp)
            f.create_dataset("A_pm", data=A_matrix_values_pm)
            f.create_dataset("A_mp", data=A_matrix_values_mp)
            f.create_dataset("A_mm", data=A_matrix_values_mm)
            f.create_dataset("ki_x", data=ki_x_array)
            f.create_dataset("ki_y", data=ki_y_array)
            f.create_dataset("ki_z", data=ki_z_array)
            f.create_dataset("kj_x", data=kj_x_array)
            f.create_dataset("kj_y", data=kj_y_array)
            f.create_dataset("kj_z", data=kj_z_array)

    def calculate_volumes_lattice(self, path: Path) -> None:
        """Pre-compute volumes for faster computation of statistics
        later on"""
        is_edge_dict = self.mode_grid.propagating_modes_is_edge_dict
        mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict

        # Calculate the central volume that is re-used a lot
        central_mode_vertices = mode_vertices_dict[0]
        central_volume = (
            geometry_utils.get_six_dimensional_intersection_volume(
                central_mode_vertices,
                central_mode_vertices,
                central_mode_vertices,
                central_mode_vertices,
            )
        )
        volumes = {(0, 0): central_volume}

        for count, first in enumerate(
            self.logger.progress_bar(self.mode_grid.propagating_indices)
        ):
            for second in self.mode_grid.propagating_indices[count:]:
                # Most cases are equal to the central weight. We don't
                # need to do anything for these cases
                if not (is_edge_dict[first] or is_edge_dict[second]):
                    volumes[(first, second)] = central_volume
                    continue

                # At least one of i and j is an edge mode
                mode_i_vertices = mode_vertices_dict[first]
                mode_j_vertices = mode_vertices_dict[second]
                volume = (
                    geometry_utils.get_six_dimensional_intersection_volume(
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_i_vertices,
                        mode_j_vertices,
                    )
                )
                volumes[(first, second)] = volume

        with open(path, "wb") as f:
            pickle.dump(volumes, f)

    def calculate_volumes(self, path: Path) -> None:
        """Pre-compute volumes for faster computation of statistics
        later on"""
        mode_vertices_dict = self.mode_grid.propagating_modes_vertices_dict

        volumes = {}

        for count, first in enumerate(
            self.logger.progress_bar(self.mode_grid.propagating_indices)
        ):
            for second in self.mode_grid.propagating_indices[count:]:
                mode_i_vertices = mode_vertices_dict[first]
                mode_j_vertices = mode_vertices_dict[second]
                volume = (
                    geometry_utils.get_six_dimensional_intersection_volume(
                        mode_i_vertices,
                        mode_j_vertices,
                        mode_i_vertices,
                        mode_j_vertices,
                    )
                )
                volumes[(first, second)] = volume

        with open(path, "wb") as f:
            pickle.dump(volumes, f)

    def get_covariance_results_lattice_generator(
        self,
        indices: list[tuple[int, int, int, int]],
        mode_indices: list[int],
        block_key: str,
        a_matrix_values_path: Path,
        volumes_path: Path,
    ) -> Generator:
        """Get all the covariance results for lattice based mode_grids
        where memory effect is desired. This uses several appoximations to
        make the computation time faesible."""

        k = self.medium_parameters.k
        L = self.medium_parameters.L
        const_factor = self.medium_parameters.cov_const_factor
        num_modes = self.mode_grid.num_propagating
        weights_dict = self.mode_grid.propagating_modes_weights_dict

        # Load appropriate pre-computed A matrix values
        a_matrix_values_key_map = {"t": "A_pp", "r": "A_pm", "r2": "A_mp"}
        with h5py.File(a_matrix_values_path, "r") as f:
            A_matrix_values = f[a_matrix_values_key_map[block_key]][:]
            ki_z_array = f["ki_z"][:]
            kj_z_array = f["kj_z"][:]

        # Load the weights
        with open(volumes_path, "rb") as f:
            volumes = pickle.load(f)

        # Begin main quadruple index loop
        reciprocity_correction = int((num_modes - 1) // 2)
        for indices in self.logger.progress_bar(indices):
            i, j, u, v = indices
            i_sequence = mode_indices.index(i)
            j_sequence = mode_indices.index(j)
            u_sequence = mode_indices.index(u)
            v_sequence = mode_indices.index(v)
            # Determine domain volume. If it's an auto-correlation, it might
            # be an edge mode, which will have smaller area.
            if i == u and j == v:
                volume = volumes.get((i, j)) or volumes.get((j, i))
            else:
                volume = volumes[(0, 0)]

            # Calculate integral
            ij_val = num_modes * i_sequence + j_sequence
            uv_val = num_modes * u_sequence + v_sequence
            ki_z = ki_z_array[ij_val]
            kj_z = kj_z_array[ij_val]
            ku_z = ki_z_array[uv_val]
            kv_z = kj_z_array[uv_val]
            wi = weights_dict[i]
            wj = weights_dict[j]
            wu = weights_dict[u]
            wv = weights_dict[v]
            sec_factor = 1.0 / np.sqrt(np.abs(ki_z * kj_z * ku_z * kv_z))
            weight_factor = 1.0 / np.sqrt(wi * wj * wu * wv)
            A_ij = A_matrix_values[ij_val]
            A_uv = A_matrix_values[uv_val]
            sinc_factor = special_functions.sinc(
                k * L * (ki_z - kj_z - ku_z + kv_z)
            )
            covariance = (
                volume
                * const_factor
                * sec_factor
                * sinc_factor
                * weight_factor
                * np.outer(A_ij, A_uv.conj()).ravel()
            )[np.newaxis, :]
            new_result = IntegrationResult(
                "covariance",
                ("pp,pp", f"{block_key},{block_key}"),
                [(i, j, u, v)],
                covariance,
            )
            yield new_result

    # def _get_covariance_integrand(
    #     self,
    #     wave_block_one: str,
    #     wave_block_two: str,
    #     block_one: str,
    #     block_two: str,
    #     is_eight_dimensions: bool = False,
    #     is_pseudo_covariance: bool = False,
    # ) -> MathematicalFunction:
    #     """Get the integrand for covariance calculations. Note that this does
    #     not include mode weights. We choose to add those in at the end."""

    #     covariance_a_matrix = (
    #         self.medium_statistics.get_covariance_a_matrix()
    #         if not is_pseudo_covariance
    #         else self.medium_statistics.get_pseudo_covariance_a_matrix()
    #     )
    #     k = self.medium_parameters.k
    #     L = self.medium_parameters.L

    #     ki_z_factor = -1.0 if block_one in {"r2", "t2"} else 1.0
    #     kj_z_factor = -1.0 if block_one in {"r", "t2"} else 1.0
    #     ku_z_factor = -1.0 if block_two in {"r2", "t2"} else 1.0
    #     kv_z_factor = -1.0 if block_two in {"r", "t2"} else 1.0
    #     pseudo_sign = -1.0 if is_pseudo_covariance else 1.0

    #     def covariance_integrand(
    #         integration_domain: np.ndarray | cp.ndarray,
    #     ) -> np.ndarray | cp.ndarray:
    #         """The integrand should be of shape N x 6, where N is the number
    #         of points that need to be evaluated. The final dimension is
    #         ki_x, ki_y, kj_x, kj_y, ku_x, ku_y"""
    #         xp = cp.get_array_module(integration_domain)

    #         # Work out wavevectors
    #         if is_eight_dimensions:
    #             ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = (
    #                 integration_domain.T
    #             )
    #         else:
    #             ki_x, ki_y, kj_x, kj_y, ku_x, ku_y = integration_domain.T
    #             kv_x = pseudo_sign * (-ki_x + kj_x) + ku_x
    #             kv_y = pseudo_sign * (-ki_y + kj_y) + ku_y
    #         ki_z = ki_z_factor * xp.sqrt(1.0 - ki_x**2 - ki_y**2)
    #         kj_z = kj_z_factor * xp.sqrt(1.0 - kj_x**2 - kj_y**2)
    #         ku_z = ku_z_factor * xp.sqrt(1.0 - ku_x**2 - ku_y**2)
    #         kv_z = kv_z_factor * xp.sqrt(1.0 - kv_x**2 - kv_y**2)

    #         sinc_factor = special_functions.sinc(
    #             k * L * (ki_z - kj_z - pseudo_sign * (ku_z - kv_z))
    #         )
    #         sec_factor = 1.0 / xp.sqrt(xp.abs(ki_z * kj_z * ku_z * kv_z))

    #         output = (
    #             covariance_a_matrix(
    #                 ki_x,
    #                 ki_y,
    #                 ki_z,
    #                 kj_x,
    #                 kj_y,
    #                 kj_z,
    #                 ku_x,
    #                 ku_y,
    #                 ku_z,
    #                 kv_x,
    #                 kv_y,
    #                 kv_z,
    #             )
    #             * sinc_factor[:, xp.newaxis]
    #             * sec_factor[:, xp.newaxis]
    #         )

    #         return output

    #     return covariance_integrand

    # def get_covariance_results_generator(
    #     self,
    #     indices: list[tuple[int, int, int, int]],
    #     block_key: str,
    #     volumes_path: Path,
    #     integration_task_config,
    #     integration_method: str = "midpoint",
    #     covariance_cubature_scheme: Any | None = None,
    #     is_eight_dimensions: bool = True,
    # ) -> Generator:
    #     xp = cp if integration_task_config.use_gpu else np

    #     # Factor common to all covariance integrals
    #     const_factor = self.medium_parameters.cov_const_factor

    #     # Load the weights
    #     with open(volumes_path, "rb") as f:
    #         volumes = pickle.load(f)

    #     # Prepare task dictionaries
    #     covariance_task_dict = {}
    #     pseudo_covariance_task_dict = {}

    #     for wave_block in WAVE_BLOCKS:
    #         wave_block_one, wave_block_two = wave_block.split(",")

    #         covariance_task_dict[wave_block] = {}
    #         pseudo_covariance_task_dict[wave_block] = {}

    #         for block in BLOCKS:
    #             block_one, block_two = block.split(",")
    #             block_location = (wave_block, block)

    #             # Set up covariance and pseudo_covariance tasks separately
    #             covariance_integrand = get_covariance_integrand(
    #                 medium_statistics,
    #                 medium_parameters,
    #                 wave_block_one,
    #                 wave_block_two,
    #                 block_one,
    #                 block_two,
    #                 is_eight_dimensions,
    #                 is_pseudo_covariance=False,
    #             )
    #             covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=covariance_integrand,
    #                 statistic_type="covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #                 is_eight_dimensions=is_eight_dimensions,
    #             )
    #             covariance_task_dict[wave_block][block] = covariance_task

    #             # Pseudo_covariance
    #             pseudo_covariance_integrand = get_covariance_integrand(
    #                 medium_statistics,
    #                 medium_parameters,
    #                 wave_block_one,
    #                 wave_block_two,
    #                 block_one,
    #                 block_two,
    #                 is_eight_dimensions,
    #                 is_pseudo_covariance=True,
    #             )
    #             pseudo_covariance_task = build_integration_task(
    #                 integration_method=integration_method,
    #                 integrand=pseudo_covariance_integrand,
    #                 statistic_type="pseudo_covariance",
    #                 block_location=block_location,
    #                 sub_block_locations=[],
    #                 const_factor=const_factor,
    #                 use_cupy=integration_task_config.use_gpu,
    #                 cubature_scheme=covariance_cubature_scheme,
    #                 use_dirac_density=False,
    #                 is_eight_dimensions=is_eight_dimensions,
    #             )
    #             pseudo_covariance_task_dict[wave_block][
    #                 block
    #             ] = pseudo_covariance_task

    #     # Main loop
    #     mode_vertices_dict = mode_grid.propagating_modes_vertices_dict
    #     mean_mode_vertices_dict = (
    #         mode_grid.propagating_modes_mean_vertices_dict
    #     )

    #     # Work out the volume common to the majority of the memory effect
    #     # type correlations
    #     repeating_mode_vertices = mode_vertices_dict.get(0)
    #     cartesian_product = geometry_utils.iterated_cartesian_product(
    #         [
    #             repeating_mode_vertices,
    #             repeating_mode_vertices,
    #             repeating_mode_vertices,
    #             repeating_mode_vertices,
    #         ]
    #     )
    #     reduced_intersection = geometry_utils.get_intersection_vertices(
    #         cartesian_product
    #     )[:, columns_to_keep]
    #     reduced_hull = scipy.spatial.ConvexHull(
    #         reduced_intersection, qhull_options="QJ"
    #     )
    #     repeating_volume = reduced_hull.volume
    #     my_counter = 0
    #     for indices in logger.progress_bar(master_indices):
    #         my_counter += 1
    #         # Work out the template's integration domain
    #         # This involves the higher dimensional geometry
    #         i, j, u, v = indices

    #         centroid = np.concatenate(
    #             [
    #                 mean_mode_vertices_dict.get(i),
    #                 mean_mode_vertices_dict.get(j),
    #                 mean_mode_vertices_dict.get(u),
    #                 mean_mode_vertices_dict.get(v),
    #             ]
    #         )

    #         if i == u and j == v:
    #             # Do it manually for autocorrelations
    #             # Most of the time it's the repeating thing, but sometimes it's not
    #             # e.g. when there are edge modes involved

    #             mode_i_vertices = mode_vertices_dict.get(i)
    #             mode_j_vertices = mode_vertices_dict.get(j)
    #             mode_u_vertices = mode_vertices_dict.get(u)
    #             mode_v_vertices = mode_vertices_dict.get(v)

    #             # Get the integration domain
    #             # This part does the geometry with the 8D region being intersected
    #             # by hyperplanes
    #             cartesian_product = geometry_utils.iterated_cartesian_product(
    #                 [
    #                     mode_i_vertices,
    #                     mode_j_vertices,
    #                     mode_u_vertices,
    #                     mode_v_vertices,
    #                 ]
    #             )
    #             reduced_intersection = (
    #                 geometry_utils.get_intersection_vertices(
    #                     cartesian_product
    #                 )[:, columns_to_keep]
    #             )
    #             reduced_hull = scipy.spatial.ConvexHull(
    #                 reduced_intersection, qhull_options="QJ"
    #             )
    #             # centroid = (
    #             #     xp.mean(cartesian_product, axis=0)
    #             #     if is_eight_dimensions
    #             #     else xp.mean(reduced_intersection, axis=0)
    #             # )
    #             volume = reduced_hull.volume
    #         else:
    #             volume = repeating_volume

    #         # Set up arrays with derived integration domains
    #         if integration_method == "cubature":
    #             centroid_expanded = xp.tile(
    #                 centroid, (reduced_hull.simplices.shape[0], 1, 1)
    #             )
    #             new_simplex_array = xp.concatenate(
    #                 [
    #                     xp.asarray(
    #                         reduced_hull.points[reduced_hull.simplices]
    #                     ),
    #                     centroid_expanded,
    #                 ],
    #                 axis=1,
    #             )
    #         elif integration_method == "midpoint":
    #             new_midpoint_array = xp.array([centroid])
    #             new_volume_array = xp.array([volume])

    #         # Add computed geometric quantities to task dictionaries
    #         for wave_block in WAVE_BLOCKS:
    #             wave_block_one, wave_block_two = wave_block.split(",")
    #             for block in BLOCKS:
    #                 block_one, block_two = block.split(",")

    #                 # Check if the mean needs to be calculated for this
    #                 # particular wave_block, block pair
    #                 if indices not in covariance_indices.get(
    #                     wave_block, {}
    #                 ).get(block, set()):
    #                     continue

    #                 # Add domain to integral task
    #                 if integration_method == "cubature":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].simplex_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_simplex_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].simplex_array = xp.vstack(
    #                         (
    #                             pseudo_covariance_task_dict[wave_block][
    #                                 block
    #                             ].simplex_array,
    #                             new_simplex_array,
    #                         )
    #                     )

    #                 elif integration_method == "midpoint":
    #                     old_stack_length = len(
    #                         covariance_task_dict[wave_block][
    #                             block
    #                         ].midpoint_array
    #                     )
    #                     new_stack_length = old_stack_length + len(
    #                         new_midpoint_array
    #                     )

    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].midpoint_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].midpoint_array,
    #                             new_midpoint_array,
    #                         )
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ].volume_array = xp.vstack(
    #                         (
    #                             covariance_task_dict[wave_block][
    #                                 block
    #                             ].volume_array,
    #                             new_volume_array,
    #                         )
    #                     )

    #                 # Add sub block locations.
    #                 # Remember that u and v are negated for pseudo_covariance
    #                 # for reciprocal mode grids
    #                 new_slice = slice(old_stack_length, new_stack_length)

    #                 new_covariance_indices = indices
    #                 new_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_covariance_indices,
    #                 )
    #                 covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_covariance_sub_block_location
    #                 )

    #                 new_pseudo_covariance_indices = (i, j, -u, -v)
    #                 new_pseudo_covariance_sub_block_location = (
    #                     new_slice,
    #                     new_pseudo_covariance_indices,
    #                 )
    #                 pseudo_covariance_task_dict[wave_block][
    #                     block
    #                 ].sub_block_locations.append(
    #                     new_pseudo_covariance_sub_block_location
    #                 )

    #                 # Execute tasks if RAM usage is getting too high.
    #                 # Re-initialize relevant integration task
    #                 current_ram_usage = system_utils.get_current_ram_usage()
    #                 # if current_ram_usage > integration_task_config.ram_limit:
    #                 if True:
    #                     new_covariance_result = covariance_task_dict[
    #                         wave_block
    #                     ][block].execute_task()
    #                     main_result_list.append_result(new_covariance_result)

    #                     block_location = (wave_block, block)

    #                     covariance_integrand = get_covariance_integrand(
    #                         medium_statistics,
    #                         medium_parameters,
    #                         wave_block_one,
    #                         wave_block_two,
    #                         block_one,
    #                         block_two,
    #                         is_eight_dimensions,
    #                         is_pseudo_covariance=False,
    #                     )

    #                     covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=covariance_integrand,
    #                         statistic_type="covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                         is_eight_dimensions=is_eight_dimensions,
    #                     )
    #                     covariance_task_dict[wave_block][
    #                         block
    #                     ] = covariance_task

    #                     new_pseudo_covariance_result = (
    #                         pseudo_covariance_task_dict[wave_block][
    #                             block
    #                         ].execute_task()
    #                     )
    #                     main_result_list.append_result(
    #                         new_pseudo_covariance_result
    #                     )

    #                     block_location = (wave_block, block)

    #                     pseudo_covariance_integrand = get_covariance_integrand(
    #                         medium_statistics,
    #                         medium_parameters,
    #                         wave_block_one,
    #                         wave_block_two,
    #                         block_one,
    #                         block_two,
    #                         is_eight_dimensions,
    #                         is_pseudo_covariance=True,
    #                     )

    #                     pseudo_covariance_task = build_integration_task(
    #                         integration_method=integration_method,
    #                         integrand=pseudo_covariance_integrand,
    #                         statistic_type="pseudo_covariance",
    #                         block_location=block_location,
    #                         sub_block_locations=[],
    #                         const_factor=const_factor,
    #                         use_cupy=integration_task_config.use_gpu,
    #                         cubature_scheme=covariance_cubature_scheme,
    #                         use_dirac_density=False,
    #                         is_eight_dimensions=is_eight_dimensions,
    #                     )
    #                     pseudo_covariance_task_dict[wave_block][
    #                         block
    #                     ] = pseudo_covariance_task

    #     # Execute remaining tasks
    #     for wave_block in WAVE_BLOCKS:
    #         for block in BLOCKS:
    #             new_covariance_result = covariance_task_dict[wave_block][
    #                 block
    #             ].execute_task()
    #             main_result_list.append_result(new_covariance_result)

    #             new_pseudo_covariance_result = pseudo_covariance_task_dict[
    #                 wave_block
    #             ][block].execute_task()
    #             main_result_list.append_result(new_pseudo_covariance_result)

    #     return main_result_list


def build_integration_task(
    integration_method: str,
    integrand: MathematicalFunction,
    statistic_type: str,
    block_location: tuple[str, str],
    sub_block_locations: list[tuple[slice, tuple[int, int]]],
    const_factor: Numeric,
    use_cupy: bool,
    cubature_scheme: Any = None,
    use_dirac_density: bool = False,
    is_eight_dimensions: bool = True,
) -> IntegrationTask:
    if integration_method == "cubature":
        return CubatureIntegrationTask(
            integrand=integrand,
            statistic_type=statistic_type,
            block_location=block_location,
            sub_block_locations=sub_block_locations,
            const_factor=const_factor,
            use_cupy=use_cupy,
            cubature_scheme=cubature_scheme,
            use_dirac_density=use_dirac_density,
        )
    elif integration_method == "midpoint":
        return MidpointIntegrationTask(
            integrand=integrand,
            statistic_type=statistic_type,
            block_location=block_location,
            sub_block_locations=sub_block_locations,
            const_factor=const_factor,
            use_cupy=use_cupy,
            is_eight_dimensions=is_eight_dimensions,
        )
    else:
        raise ValueError(f"Unknown integration method '{integration_method}'")
