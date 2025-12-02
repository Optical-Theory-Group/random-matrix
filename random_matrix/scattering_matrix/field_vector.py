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

    def plot_polygon(self, array: np.array, **plot_kwargs) -> None:
        """Plot the values contained in array on top of the mode grid using
        a polygon based approach"""
        vertices_list = self.mode_grid.propagating_modes_vertices

        norm = plt.Normalize(vmin=np.min(array), vmax=np.max(array))
        cmap = plt.cm.jet

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
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

    def get_imshow_array(self, array: np.ndarray) -> None:
        # Construct 2D array for correlation computations
        rows = []
        cols = []
        values = []
        spacing = 0.07

        for mode in self.mode_grid.propagating_modes_list:
            if mode.is_edge:
                continue

            center = mode.center
            multiples = np.rint(center / spacing).astype(int)
            rows.append(-multiples[1])
            cols.append(multiples[0])
            values.append(
                array[self.mode_grid.propagating_indices.index(mode.index)]
            )

        rows_shifted = [row + abs(min(rows)) for row in rows]
        cols_shifted = [col + abs(min(cols)) for col in cols]
        size = max(rows_shifted) + 1
        print(size)
        values_array = np.zeros((size, size), dtype=np.float64)
        for r, c, v in zip(rows_shifted, cols_shifted, values):
            values_array[r, c] = v
        return values_array
    
    def plot_imshow(self, array: np.array, **plot_kwargs) -> None:
        """Plot the values contained in array on top of the mode grid using
        a polygon based approach"""
        values_array = self.get_imshow_array(array)
        norm = plt.Normalize(
            vmin=np.min(values_array), vmax=np.max(values_array)
        )
        cmap = plt.cm.jet
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = plt.subplots()
        im = ax.imshow(values_array, norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax)

    def plot_imshows(self, arrays: list[np.array], **plot_kwargs) -> None:
        """Plot the values contained in array on top of the mode grid using
        a polygon based approach"""
        # Construct 2D array for correlation computations
        rows = []
        cols = []
        values = []
        spacing = 0.07

        for mode in self.mode_grid.propagating_modes_list:
            if mode.is_edge:
                continue

            center = mode.center
            multiples = np.rint(center / spacing).astype(int)
            rows.append(-multiples[1])
            cols.append(multiples[0])
            values.append(
                array[self.mode_grid.propagating_indices.index(mode.index)]
            )

        rows_shifted = [row + abs(min(rows)) for row in rows]
        cols_shifted = [col + abs(min(cols)) for col in cols]
        size = max(rows_shifted) + 1
        print(size)
        values_array = np.zeros((size, size), dtype=np.float64)
        for r, c, v in zip(rows_shifted, cols_shifted, values):
            values_array[r, c] = v

        norm = plt.Normalize(
            vmin=np.min(values_array), vmax=np.max(values_array)
        )
        cmap = plt.cm.jet
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = plt.subplots()
        im = ax.imshow(values_array, norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax)
