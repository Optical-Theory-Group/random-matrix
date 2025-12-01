import numpy as np
from typing import Optional
from random_matrix.modes.mode_grid import ModeGrid
import matplotlib.pyplot as plt

class FieldVector:
    def __init__(self, mode_grid: ModeGrid):
        self.mode_grid = mode_grid

    def by_index(self, array: np.ndarray, index: int) -> np.ndarray:
        """Return the two component (s,p) vector from the mode index"""
        return self.get_jones_dict(array)[index]

    def get_jones_dict(self, array: np.ndarray):
        jones_dict = {}
        for i, mode in enumerate(self.mode_grid.propagating_modes_list):
            jones_dict[mode.index] = array[2 * i : 2 * (i + 1)]
        return jones_dict

    def _get_component_dict(
        self, array: np.ndarray, polarization: str
    ) -> dict[str, np.ndarray]:
        """Get dictionary of field components where the keys are the mode
        indices"""
        output = {}
        func_dict = {"s": self.get_s, "p": self.get_p}
        values = func_dict[polarization](array)
        for index, value in zip(self.mode_grid.propagating_indices, values):
            output[index] = value
        return output

    def get_s_dict(self, array: np.ndarray) -> dict[str, np.ndarray]:
        return self._get_component_dict(array, "s")

    def get_p_dict(self, array: np.ndarray) -> dict[str, np.ndarray]:
        return self._get_component_dict(array, "p")

    def get_s(self, array: np.ndarray) -> np.ndarray:
        return array[::2]

    def get_p(self, array: np.ndarray) -> np.ndarray:
        return array[1::2]

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot(self, array: np.array, **plot_kwargs) -> None:
        """Plot the values contained in array on top of the mode grid"""
        vertices_list = self.mode_grid.propagating_modes_vertices

        norm = plt.Normalize(
            vmin=np.min(array), vmax=np.max(array)
        )
        cmap = plt.cm.jet

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_aspect("equal")

        bg_color = cmap(norm(np.min(array)))
        ax.set_facecolor(bg_color)

        for i, vertices in enumerate(vertices_list):
            color = cmap(norm(array[i]))
            polygon = plt.Polygon(vertices, color=color, alpha=0.9)
            ax.add_patch(polygon)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)

        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
