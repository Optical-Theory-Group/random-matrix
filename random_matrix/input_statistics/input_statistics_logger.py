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
    def log(self, operation: str) -> Generator[None, None, None]:
        """Main logging method that handles printing and method timing."""

        # Print start message
        print(
            f"{self._get_date_time()} {self.messages.get(operation, '')}",
            flush=True,
        )

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
            "independent_elements": "Find independent elements",
            "mean": "Calculate mean indices",
            "covariance": "Calculate covariance indices",
            "pseudo_covariance": "Calculate pseudo_covariance indices",
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
    ) -> None:
        print(
            self.get_shapes_report(
                num_single_templates, num_quadruple_templates
            ),
            flush=True,
        )

    @staticmethod
    def get_shapes_report(
        num_single_templates: int,
        num_quadruple_templates: int,
    ) -> str:
        out = (
            f"Number of single templates: {num_single_templates}\n"
            f"Number of quadruple templates: {num_quadruple_templates}\n"
        )
        return out


class IntegrationTaskPreparerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "mean": "Prepare mean tasks",
            "covariance": "Prepare covariance tasks",
            "pseudo_covariance": "Prepare pseudo covariance tasks",
        }

    def show_report(self) -> None:
        print(self.get_time_report(), flush=True)


class InputStatisticsManagerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "tasks": "Execute tasks",
            "mean": "Get mean matrix",
            "covariance": "Get covariance matrix",
            "psuedo_covariance": "Get pseudo covariance matrix",
        }

    def show_report(self) -> None:
        print(self.time_report(), flush=True)
