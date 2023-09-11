"""Class for logging operations associated with finding the input statistics"""

import time
from contextlib import contextmanager


class InputStatisticsLogger:
    def __init__(self) -> None:
        self.times = {}

    @contextmanager
    def log(self, operation: str) -> None:
        # Print message to console
        print(self.messages.get(operation, ""), flush=True)

        # Time code for performance
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.times[operation] = end - start
        print("Done", flush=True)


class IndexFinderLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "independent_elements": "Find independent elements",
            "mean": "Calculate mean indices",
            "covariance": "Calculate covariance indices",
            "pseudo_covariance": "Calculate pseudo_covariance indices",
        }

    def time_report(self):
        out = (
            f"Find independent elements: "
            f"{self.times.get('independent_elements', -1)}\n"
            f"Find mean indices: "
            f"{self.times.get('mean', -1)}\n"
            f"Find covariance indices: "
            f"{self.times.get('covariance', -1)}\n"
            f"Find pseudo covariance indices: "
            f"{self.times.get('pseudo_covariance', -1)}\n"
        )
        return out

    def indices_report(self, independent_elements, indices):
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

    def show_report(self, independent_elements, indices) -> None:
        print(self.time_report(), flush=True)
        print(self.indices_report(independent_elements, indices), flush=True)


class ShapeClassifierLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "singles": "Classify singles",
            "quadruples": "Classify quadruples",
            "templates": "Calculate template integration domains",
            "others": "Calculate other integration domains",
        }

    def time_report(self):
        out = (
            f"{self.messages['singles']}: "
            f"{self.times.get('singles', -1)}\n"
            f"{self.messages['quadruples']}: "
            f"{self.times.get('quadruples', -1)}\n"
            f"{self.messages['templates']}: "
            f"{self.times.get('templates', -1)}\n"
            f"{self.messages['others']}: "
            f"{self.times.get('others', -1)}\n"
        )
        return out

    def shapes_report(self, single_templates, quadruple_templates):
        out = (
            f"Number of single templates: {single_templates}\n"
            f"Number of quadruple templates: {quadruple_templates}\n"
        )
        return out

    def show_report(self, single_templates, quadruple_templates) -> None:
        print(self.time_report(), flush=True)
        print(
            self.shapes_report(single_templates, quadruple_templates),
            flush=True,
        )


class IntegrationTaskPreparerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "mean": "Prepare mean tasks",
            "covariance": "Prepare covariance tasks",
            "pseudo_covariance": "Prepare pseudo covariance tasks",
        }

    def time_report(self):
        out = (
            f"{self.messages['mean']}: "
            f"{self.times.get('mean', -1)}\n"
            f"{self.messages['covariance']}: "
            f"{self.times.get('covariance', -1)}\n"
            f"{self.messages['pseudo_covariance']}: "
            f"{self.times.get('pseudo_covariance', -1)}\n"
        )
        return out

    def show_report(self) -> None:
        print(self.time_report(), flush=True)


class InputStatisticsManagerLogger(InputStatisticsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages = {
            "tasks": "Execute tasks",
            "mean": "Get mean matrix",
            "covariance": "Get covariance matrix",
            "psuedo_covariance": "Get pseudo covariance matrix",
        }

    def time_report(self):
        out = (
            f"{self.messages['tasks']}: "
            f"{self.times.get('tasks', -1)}\n"
            f"{self.messages['mean']}: "
            f"{self.times.get('mean', -1)}\n"
            f"{self.messages['covariance']}: "
            f"{self.times.get('covariance', -1)}\n"
            f"{self.messages['psuedo_covariance']}: "
            f"{self.times.get('psuedo_covariance', -1)}\n"
        )
        return out

    def show_report(self) -> None:
        print(self.time_report(), flush=True)
