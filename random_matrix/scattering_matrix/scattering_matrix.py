import numpy as np
from typing import Optional
from random_matrix.modes.mode_grid import ModeGrid
import matplotlib.pyplot as plt


class ScatteringMatrix:
    def __init__(self, mode_grid: ModeGrid):
        self.mode_grid = mode_grid

    def get_ss(self, array: np.ndarray) -> np.ndarray:
        return array[::2, ::2]

    def get_sp(self, array: np.ndarray) -> np.ndarray:
        return array[::2, 1::2]

    def get_ps(self, array: np.ndarray) -> np.ndarray:
        return array[1::2, ::2]

    def get_pp(self, array: np.ndarray) -> np.ndarray:
        return array[1::2, 1::2]

    def get_column(self, array: np.ndarray, )
