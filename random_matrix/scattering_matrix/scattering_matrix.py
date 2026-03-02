import numpy as np
from typing import Optional
from random_matrix.modes.mode_grid import ModeGrid
import matplotlib.pyplot as plt
from random_matrix.utils import matrix_utils


class ScatteringMatrix:
    def __init__(self, mode_grid: ModeGrid):
        self.mode_grid = mode_grid

    def get_block(self, array: np.ndarray, block: str) -> np.ndarray:
        return matrix_utils.get_block(array, block)

    def get_ss(self, array: np.ndarray) -> np.ndarray:
        return array[::2, ::2]

    def get_sp(self, array: np.ndarray) -> np.ndarray:
        return array[::2, 1::2]

    def get_ps(self, array: np.ndarray) -> np.ndarray:
        return array[1::2, ::2]

    def get_pp(self, array: np.ndarray) -> np.ndarray:
        return array[1::2, 1::2]

    def get_column(self, array: np.ndarray, mode_index: int) -> np.ndarray:
        """It is assumed that the matrix has already been reduced to one of
        the four scattering blocks (r,t,t2,r2) AND one polarization component
        (s, p)"""
        column_index = self.mode_grid.propagating_indices.index(mode_index)
        return array[:, column_index]

    def get_row(self, array: np.ndarray, mode_index: int) -> np.ndarray:
        """It is assumed that the matrix has already been reduced to one of
        the four scattering blocks (r,t,t2,r2) AND one polarization component
        (s, p)"""
        row_index = self.mode_grid.propagating_indices.index(mode_index)
        return array[row_index, :]

    def get_column_intensity(
        self,
        array: np.ndarray,
        incident_index: int,
        incident_polarization: str = "s",
    ) -> np.ndarray:
        """It is assumed the matrix given is one of the blocks, i.e. r,t,t2,r2"""
        if incident_polarization == "s":
            block_p = self.get_ps(array)
            block_s = self.get_ss(array)
        else:
            block_p = self.get_pp(array)
            block_s = self.get_sp(array)

        col_p = self.get_column(block_p, incident_index)
        col_s = self.get_column(block_s, incident_index)
        return np.abs(col_p) ** 2 + np.abs(col_s) ** 2
