from dataclasses import dataclass
from pathlib import Path

BLOCK_KEYS = ["t", "r", "r2"]

DEFAULT_DATA_PATH_ENDING = "data"
DEFAULT_POOLS_PATH_ENDING = "pools"
DEFAULT_CASCADES_PATH_ENDING = "cascades"

INPUT_STATISTICS_PATH_ENDINGS = {
    "metadata": "metadata.json",
    "mode_grid": "mode_grid.pkl",
    "mode_grid_figure": "mode_grid.svg",
    "mode_grid_figure_with_indices": "mode_grid_with_indices.svg",
    "medium_parameters": "medium_parameters.pkl",
    "medium_statistics": "medium_statistics.pkl",
    "independent_elements": "independent_elements.pkl",
    "indices": "indices.pkl",
    "integration_result_list": "integration_result_list.pkl",
    "integration_task_config": "integration_task_config.pkl",
    "mean_S": "mean_S.npy",
    "cholesky": "cholesky.pkl",
    "covariance": "covariance.npz",
    "pseudo_covariance": "pseudo_covariance.npz",
    "real_covariance": "real_covariance.pkl",
    "a_matrix_values": "a_matrix.h5",
    "volumes": "volumes.pkl",
    "covariance_blocks": {
        block_key: f"covariance_{block_key}.npz" for block_key in BLOCK_KEYS
    },
    "covariance_blocks_partial": {
        block_key: f"covariance_{block_key}" for block_key in BLOCK_KEYS
    },
    "cholesky_blocks": {
        block_key: f"cholesky_{block_key}.npz" for block_key in BLOCK_KEYS
    },
}
MATRIX_POOLS_PATH_ENDINGS = {"pools": "pools.h5"}


@dataclass(frozen=True)
class InputStatisticsPaths:
    """Path names for all stored data relevant to the incident statistics that
    feed into random matrix generation"""

    simulation_name: str
    base_path: Path | None = None

    def __post_init__(self):
        base = (
            Path(self.base_path)
            if self.base_path
            else Path.cwd() / DEFAULT_DATA_PATH_ENDING
        )
        sim = base / self.simulation_name

        # Create folders for saving all simulation data
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {base}")

        if not sim.exists():
            sim.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {sim}")

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "simulation", sim)

        for key, ending in INPUT_STATISTICS_PATH_ENDINGS.items():
            if isinstance(ending, str):
                object.__setattr__(self, key, sim / ending)
            elif isinstance(ending, dict):
                object.__setattr__(
                    self, key, {k: sim / v for k, v in ending.items()}
                )

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def get_covariance_blocks_partial_path(
        self, block_key: str, count: int
    ) -> Path:
        return Path(f"{self.covariance_blocks_partial[block_key]}_{count}.npz")


@dataclass(frozen=True)
class MatrixPoolsPaths:
    """Path names for all stored data relevant to the matrix pools and
    subsequent data collection"""

    simulation_name: str
    base_path: Path | None = None

    def __post_init__(self):
        base = (
            Path(self.base_path)
            if self.base_path
            else Path.cwd() / DEFAULT_DATA_PATH_ENDING
        )
        sim = base / self.simulation_name

        pools = sim / DEFAULT_POOLS_PATH_ENDING
        if not pools.exists():
            pools.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {pools}")

        cascades = sim / DEFAULT_CASCADES_PATH_ENDING
        if not cascades.exists():
            cascades.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {cascades}")

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "simulation", sim)
        object.__setattr__(self, DEFAULT_POOLS_PATH_ENDING, pools)
        object.__setattr__(self, DEFAULT_CASCADES_PATH_ENDING, cascades)

        for key, ending in MATRIX_POOLS_PATH_ENDINGS.items():
            if isinstance(ending, str):
                object.__setattr__(self, key, pools / ending)
            elif isinstance(ending, dict):
                object.__setattr__(
                    self, key, {k: sim / v for k, v in ending.items()}
                )

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def get_cascade_h5_path(self, cascade_name: str) -> Path:
        return Path(self.cascades / f"{cascade_name}.h5")
