"""Class for logging operations associated with finding the input statistics"""

import contextlib
import datetime
import time
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import tqdm


class NullLogger:
    """Empty logger that is used when logging is undesired"""

    def __init__(self) -> None:
        self.log = contextlib.nullcontext

    def show_report(self, *args: Any) -> None:
        return None

    @staticmethod
    def progress_bar(iterable: Any) -> Any:
        return iterable


class InputStatisticsLogger:
    """Base class for printing console messages and timing different
    operations. Should not be used directly."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.messages: dict[str, str] = {}
        self.progress_bar = tqdm.tqdm

    # -------------------------------------------------------------------------
    # Timing and logging methods
    # -------------------------------------------------------------------------

    @contextmanager
    def log(self, operation: str, **kwargs) -> Generator[None, None, None]:
        """Main logging method that handles printing and method timing."""

        message_template = self.messages.get(operation, "")
        message = message_template.format(**kwargs)
        
        print(f"{self._get_date_time()} {message}", flush=True)

        # Time code for performance
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.times[operation] = end - start

        # Print end message
        print(f"{self._get_date_time()} Done", flush=True)

    @staticmethod
    def _get_date_time() -> str:
        """Return the current day and time"""

        return datetime.datetime.now().strftime("[%m/%d %H:%M:%S]")

    # -------------------------------------------------------------------------
    #  Report methods
    # -------------------------------------------------------------------------

    def show_report(self, *args: Any) -> None:
        """Main report method. Prints a report to the terminal."""

        self._show_base_report()
        self._show_subclass_report(*args)

    def _show_base_report(self) -> None:
        """Prints the report common to all subclasses"""

        print(self.get_time_report(), flush=True)

    @abstractmethod
    def _show_subclass_report(self, *args: Any) -> None:
        """Prints report specific to subclasses"""
        pass

    def get_time_report(self) -> str:
        """Get a report of times for all operations"""

        return "Times:\n" + "".join(
            [
                f"{message}: {self.get_operation_time(operation)}\n"
                for operation, message in self.messages.items()
            ]
        )

    def get_operation_time(self, operation: str) -> float:
        """Retrieve the time for a particular opearation"""

        return self.times.get(operation, 0.0)


class IndexFinderLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "independent_elements": "Finding independent elements...",
            "mean": "Finding indices for non-zero mean sub-blocks...",
            "covariance": "Finding indices for non-zero covariance sub-blocks...",
            "pseudo_covariance": "Finding indices for non-zero pseudo-covariance sub-blocks...",
        }

    # -------------------------------------------------------------------------
    # Report methods
    # -------------------------------------------------------------------------

    def _show_subclass_report(
        self,
        independent_elements: dict[str, dict[str, set[tuple[int, int]]]],
        indices: dict[str, dict[str, dict[str, list[tuple[int, int]]]]],
    ) -> None:
        print(
            self.get_indices_report(independent_elements, indices), flush=True
        )

    @staticmethod
    def get_indices_report(
        independent_elements: dict[str, dict[str, set[tuple[int, int]]]],
        indices: dict[str, dict[str, dict[str, list[tuple[int, int]]]]],
    ) -> str:
        out = (
            f"Number of independent elements:\n"
            f"t: {len(independent_elements['pp']['t'])}\n"
            f"r: {len(independent_elements['pp']['r'])}\n"
            f"t2: {len(independent_elements['pp']['t2'])}\n"
            f"r2: {len(independent_elements['pp']['r2'])}\n\n"
            f"Number of mean indices:\n"
            f"t: {len(indices['mean']['pp']['t'])}\n"
            f"r: {len(indices['mean']['pp']['r'])}\n"
            f"t2: {len(indices['mean']['pp']['t2'])}\n"
            f"r2: {len(indices['mean']['pp']['r2'])}\n\n"
            f"Number of covariance indices:\n"
            f"t,t: {len(indices['covariance']['pp,pp']['t,t'])}\n"
            f"t,r: {len(indices['covariance']['pp,pp']['t,r'])}\n"
            f"t,t2: {len(indices['covariance']['pp,pp']['t,t2'])}\n"
            f"t,r2: {len(indices['covariance']['pp,pp']['t,r2'])}\n"
            f"r,r: {len(indices['covariance']['pp,pp']['r,r'])}\n"
            f"r,t2: {len(indices['covariance']['pp,pp']['r,t2'])}\n"
            f"r,r2: {len(indices['covariance']['pp,pp']['r,r2'])}\n"
            f"t2.r2: {len(indices['covariance']['pp,pp']['t2,t2'])}\n"
            f"t2,r2: {len(indices['covariance']['pp,pp']['t2,r2'])}\n"
            f"r2,r2: {len(indices['covariance']['pp,pp']['r2,r2'])}\n"
        )
        return out


class ShapeClassifierLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "singles": "Classify singles",
            "quadruples": "Classify quadruples",
            "templates": "Calculate template integration domains",
            "others": "Calculate other integration domains",
        }

    # -------------------------------------------------------------------------
    # Report methods
    # -------------------------------------------------------------------------

    def _show_subclass_report(
        self,
        num_single_templates: int,
        num_quadruple_templates: int,
        num_quadruples: int,
    ) -> None:
        print(
            self.get_shapes_report(
                num_single_templates, num_quadruple_templates, num_quadruples
            ),
            flush=True,
        )

    @staticmethod
    def get_shapes_report(
        num_single_templates: int,
        num_quadruple_templates: int,
        num_quadruples: int,
    ) -> str:
        out = (
            f"Number of single templates: {num_single_templates}\n"
            f"Number of quadruple templates: {num_quadruple_templates}\n"
            f"Number of quadruples: {num_quadruples}\n"
            f"Percentage: {(num_quadruple_templates + num_quadruples)/num_quadruples * 100}\n"
        )
        return out


class IntegrationTaskPreparerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "mean": "Calculating mean S matrix statistics...",
            "covariance": "Calculating S matrix element covariances...",
            "pseudo_covariance": "Calculating S matrix element pseudo-covariances...",
        }

    def show_report(self) -> None:
        print(self.get_time_report(), flush=True)


class InputStatisticsManagerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "load_statistics": "Loading statistics from memory...",
            "load_partial_statistics": "The mean, covariance and pseudo-covariance have already been compiled. Loading from memory...",
            "load_indices": "Indices have already been calculated. Loading from memory...",
            "load_integration_results": "Integration results have already been calculated. Loading from memory...",
            "load_real_covariance": "Real covariance matrix has already been compiled. Loading from memory...",
            "calculate_a_matrix": "Pre-computing A matrix values for later use...",
            "calculate_volumes": "Pre-computing volumes for later use...",
            "tasks": "Execute tasks",
            "mean": "Compiling the mean scattering matrix...",
            "covariance": "Compiling the covariance matrix...",
            "pseudo_covariance": "Compiling the pseudo-covariance matrix...",
            "real_covariance": "Compiling the covariance matrix for real and imaginary parts...",
            "cholesky": "Computing the Cholesky decomposition...",
            "covariance_block": "Calculating covariance statistics for block {block}...",
            "covariance_partial": "Starting batch {count}/{total}..."
        }

    def show_report(self) -> None:
        print(self.time_report(), flush=True)
