import numpy as np
import matplotlib.pyplot as plt
from random_matrix.mode import Mode
from random_matrix.utils.geometry_utils import cartesian_to_polar, polar_to_cartesian
from random_matrix.utils.plotting_utils import set_up_k_space_plot

class ModeGrid:
    def __init__(self, mode_boundary_data: dict = None) -> None:
        self.modes = []
        
        # Parse mode_boundary_data
        t_offset = mode_boundary_data.get("t_offset", 0)

        match mode_boundary_data:
            case {"mode_shape_type": "polar", "r_vals": r_vals, "t_vals": t_vals}:
                self._handle_polar_case(r_vals=r_vals, t_vals=t_vals, t_offset=t_offset)

            case {"mode_shape_type": "cartesian", "dx": dx, "dy": dy}:
                self._handle_cartesian_case(dx=dx, dy=dy, t_offset=t_offset)

            case {"mode_shape_type": "custom", "points_array": points_array}:
                pass

            case _:
                raise ValueError("Incorrect mode_boundary_data formatting.")

    def _handle_polar_case(self, r_vals, t_vals, t_offset):
        mode_shape_type = "polar"
        
        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not np.any(np.isclose(r_vals, 0.0)):
            r_vals = np.concatenate(([0.0], r_vals))
        if not np.any(np.isclose(r_vals, 1.0)):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Sort out central mode
        central_mode_radius = r_vals[1]
        points_cartesian = np.array([[0.0,0.0], [central_mode_radius, 0.0]])
        self.modes.append(Mode(index=0, mode_boundary=points_cartesian, mode_shape_type=mode_shape_type))

        # Non-central modes
        # Add t-offset
        t_vals += t_offset

        r_vals = r_vals[1:]
        r_val_pairs = np.column_stack((r_vals[:-1], r_vals[1:]))
        t_val_pairs = np.column_stack((t_vals[:-1], t_vals[1:]))

        for r_val_pair in r_val_pairs:
            for t_val_pair in t_val_pairs:
                r_grid, t_grid = np.meshgrid(r_val_pair, t_val_pair)
                points_polar = np.column_stack((r_grid.ravel(), t_grid.ravel()))
                points_cartesian = polar_to_cartesian(points_polar)
                self.modes.append(Mode(index=0, mode_boundary=points_cartesian, mode_shape_type=mode_shape_type))

    def _handle_cartesian_case(self):
        pass

    def _handle_custom_case(self):
        pass

    def plot(self):
        # Draw axes and k-space boundary
        ax = set_up_k_space_plot()
        for mode in self.modes:
            mode.plot(ax=ax, is_solo=False, show_guidelines=False, mode_color="tab:red")