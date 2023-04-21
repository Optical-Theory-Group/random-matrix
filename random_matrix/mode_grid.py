import numpy as np
import matplotlib.pyplot as plt
from .mode import Mode
from .utils.geometry_utils import cartesian_to_polar, polar_to_cartesian
from .utils.plotting_utils import set_up_k_space_plot
from .utils.array_utils import get_pairs

class ModeGrid:
    def __init__(self, grid_data: dict = None) -> None:
        self.modes = []
        
        # Parse mode_boundary_data
        t_offset: float = grid_data.get("t_offset", 0.0)
        grid_type: str = grid_data.get("grid_type", "custom")

        self.t_offset = t_offset
        self.grid_type = grid_type

        match grid_data:
            case {"grid_type": "polar", "r_vals": r_vals, "t_vals": t_vals}:
                self._handle_polar_case(r_vals=r_vals, t_vals=t_vals, t_offset=t_offset)

            case {"grid_type": "cartesian", "dx": dx, "dy": dy}:
                self._handle_cartesian_case(dx=dx, dy=dy, t_offset=t_offset)

            case {"grid_type": "custom", "points_array": points_array}:
                pass

            case _:
                raise ValueError("Incorrect mode_boundary_data formatting.")

    def __str__(self):
        output = f"Grid type: {self.grid_type},\nReciprocal: {self.is_reciprocal},\nNumber of modes: {len(self.modes)},\nTheta offset: {self.t_offset}"
        return output

    def _handle_polar_case(self, r_vals: np.ndarray, t_vals: np.ndarray, t_offset: float):        
        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not np.any(np.isclose(r_vals, 0.0)):
            r_vals = np.concatenate(([0.0], r_vals))
        if not np.any(np.isclose(r_vals, 1.0)):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not np.any(np.isclose(t_vals, 0.0)):
            t_vals = np.concatenate(([0.0], t_vals))
        if not np.any(np.isclose(t_vals, 2*np.pi)):
            t_vals = np.concatenate((t_vals, [2*np.pi]))

        # Check if grid is reciprocal
        if len(t_vals) % 2 == 0:
            self.is_reciprocal = False
        else:
            t_pairs = get_pairs(t_vals)
            t_differences = np.mod(t_pairs[:,1] - t_pairs[:,0], 2*np.pi)
            half_way_index = int(len(t_differences)/2)     
            first_half = t_differences[:half_way_index]
            second_half = t_differences[half_way_index:]
            self.is_reciprocal = np.all(np.isclose(first_half, second_half))

        # Handle central mode
        central_mode_radius = r_vals[1]
        points_cartesian = np.array([[0.0,0.0], [central_mode_radius, 0.0]])
        self.modes.append(Mode(index=0, mode_boundary=points_cartesian, is_polar=True))

        # Non-central modes
        r_vals = r_vals[1:]
        r_val_pairs = get_pairs(r_vals)
        t_val_pairs = get_pairs(t_vals)

        mode_index = 0
        reciprocal_mode_index = 0

        for (t_index, t_val_pair) in enumerate(t_val_pairs):
            t_val_pair += t_offset
            for r_val_pair in r_val_pairs:
                # Check if points are reciprocal inverse to an already existing mode
                # If so, use its negative index
                if self.is_reciprocal and t_index >= half_way_index:
                    reciprocal_mode_index -= 1
                    new_mode_index = reciprocal_mode_index
                else:
                    mode_index += 1
                    new_mode_index = mode_index
                
                r_grid, t_grid = np.meshgrid(r_val_pair, t_val_pair)
                points_polar = np.column_stack((r_grid.ravel(), t_grid.ravel()))
                points_cartesian = polar_to_cartesian(points_polar)
                self.modes.append(Mode(index=new_mode_index, mode_boundary=points_cartesian, is_polar=True))

    def _handle_cartesian_case(self, x_lim=1.0, y_lim=1.0):
        # Set up x-y lattice
        x_vals = np.arange(0.0, x_lim+dx, dx)
        x_vals = np.concatenate((-x_vals[1:][::-1], x_vals))
        y_vals = np.arange(0.0, y_lim+dy, dy)
        y_vals = -np.concatenate((-y_vals[1:][::-1], y_vals))
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        r_grid = np.sqrt(x_grid**2 + y_grid**2)
        interior_mask = r_grid <= 1
        pass

    def _handle_custom_case(self):
        pass

    def get_mode_by_index(self, index: int) -> Mode:
        for mode in self.modes:
            if mode.index == index:
                return mode
        return None

    def plot(self, show_indices: bool = False) -> None:
        # Draw axes and k-space boundary
        ax = set_up_k_space_plot()
        for mode in self.modes:
            mode.plot(ax=ax, is_solo=False, show_guidelines=False, mode_color="tab:red", show_index=show_indices)