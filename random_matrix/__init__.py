import os
import warnings
from pathlib import Path

from .config.tools import (
    get_data_directory,
    set_data_directory,
    print_data_directory,
)

DATA_DIRECTORY = get_data_directory()

# Check if the directory exists and handle cases accordingly
if not DATA_DIRECTORY.exists():
    parent_dir = DATA_DIRECTORY.parent
    if parent_dir.exists():
        warnings.warn(
            f"Data directory '{DATA_DIRECTORY}' does not exist. Creating it "
            f"now."
        )
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(
            f"Data directory '{DATA_DIRECTORY}' does not exist and cannot be "
            f"created because the given parent directory does not exist. "
            f"Please edit 'data_directory' in "
            f"'random_matrix/config/config.json' or create the parent "
            f"directory manually."
        )
