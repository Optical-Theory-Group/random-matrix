"""This module defines a "ModeGrid" class for use in scattering calculations.

ModeGrid acts as a generator and container for Mode objects.
"""

from typing import Any

import numpy as np

from random_matrix.mode import Mode
from random_matrix.utils.array_utils import get_pairs
from random_matrix.utils.geometry_utils import (
    get_polygon_circle_intersection_points,
    order_points,
    polar_to_cartesian,
    rotate_points,
)
from random_matrix.utils.plotting_utils import set_up_k_space_plot


class ModeGrid:
    """
    A class used to represent a grid of modes.

    Attributes
    ----------
    modes_propagating : dict
        A dictionary containing all of the propagating modes. Note that the
        keys are the mode indices.
    modes_evanescent : dict
        A dictionary containing all of the evanescent modes. Note that the
        keys are the mode indices.
    t_offset : float
        An angle used to rotate the entire grid.
    grid_type : str
        Used to generate special grids. Possible values are:

        "cartesian", "polar", "custom".

    grid_wave_type: str
        The types of modes that are to be generated in the grid. Possible
        values are:

        "propagating", "evanescent", "all".

    Methods
    -------
    """

    def __init__(self, grid_data: dict[str, Any] | None = None) -> None:
        """
        Initialises grid of modes.

        Parameters
        ----------
        grid_data : dict
            A dictionary containing data that is used to generate the grid.

        Returns
        -------
        None
        """

        self.modes_propagating: dict[str, Mode] = {}
        self.modes_evanescent: dict[str, Mode] = {}

        if grid_data is None:
            raise ValueError("grid_data not provided")

        # Parse global parameters in grid_data
        t_offset = grid_data.get("t_offset", 0.0)
        grid_type = grid_data.get("grid_type", "custom")
        grid_wave_type = grid_data.get("grid_wave_type", "propagating")
        self.t_offset = t_offset
        self.grid_type = grid_type
        self.grid_wave_type = grid_wave_type

        # Parse specialised parameters in grid_data
        match grid_data:
            case {"grid_type": "polar", "r_vals": r_vals, "t_vals": t_vals}:
                self.t_vals = t_vals
                self.r_vals = r_vals
                self._handle_polar_case()

            case {"grid_type": "cartesian", "dx": dx, "dy": dy}:
                x_lim = grid_data.get("x_lim", 1.0 + dx)
                y_lim = grid_data.get("y_lim", 1.0 + dy)
                self.dx = dx
                self.dy = dy
                self.x_lim = x_lim
                self.y_lim = y_lim
                self._handle_cartesian_case()

            case _:
                raise ValueError(
                    "Incorrect mode_boundary_data formatting."
                    "Please refer to the documentation"
                )

    def __str__(self) -> str:
        """
        Gives a helpful summary of the grid.

        Parameters
        ----------
        None

        Returns
        -------
        output : str
            A string summarising the grid.
        """

        output = (
            f"Grid type: {self.grid_type},\n"
            f"Reciprocal: {self.is_reciprocal},\n"
            f"Number of propagating modes: {len(self.modes_propagating)},\n"
            f"Number of evanescent modes: {len(self.modes_evanescent)},\n"
            f"Theta offset: {self.t_offset}"
        )
        return output

    def get_mode_by_index(
        self, index: str | int, mode_wave_type: str = "propagating"
    ) -> Mode | None:
        """
        Gets a mode from the ModeGrid dictionaries according to an input
        index and wave type.

        Parameters
        ----------
        index : str, int
            The index of the desired mode.
        mode_wave_type : str
            The type of mode that is to be collected. This determines which
            dictionary the mode will be taken from.

        Returns
        -------
        mode : Mode
            The mode with desired index.
        """

        # Stringify index for consistency
        index = str(index)
        if mode_wave_type == "propagating":
            mode = self.modes_propagating.get(index, None)
        else:
            mode = self.modes_evanescent.get(index, None)
        return mode

    def add_mode(
        self, mode: Mode, mode_wave_type: str = "propagating"
    ) -> None:
        """
        Adds a new mode to the internal ModeGrid dictionaries for storage.

        Parameters
        ----------
        mode : Mode
            The new mode that is to be added
        mode_wave_type : str
            The type of mode that is to be added. This determines which
            dictionary the mode will be stored in.

        Returns
        -------
        None
        """

        index = str(mode.index)
        if mode_wave_type == "propagating":
            self.modes_propagating[index] = mode
        else:
            self.modes_evanescent[index] = mode
        pass

    def _handle_polar_case(self) -> None:
        """
        Sets up a Polar grid of modes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        t_vals = self.t_vals
        r_vals = self.r_vals

        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not np.any(np.isclose(r_vals, 0.0)):
            r_vals = np.concatenate(([0.0], r_vals))
        if not np.any(np.isclose(r_vals, 1.0)):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not np.any(np.isclose(t_vals, 0.0)):
            t_vals = np.concatenate(([0.0], t_vals))
        if not np.any(np.isclose(t_vals, 2 * np.pi)):
            t_vals = np.concatenate((t_vals, [2 * np.pi]))

        # Check if grid is reciprocal
        if len(t_vals) % 2 == 0:
            self.is_reciprocal = False
        else:
            t_pairs = get_pairs(t_vals)
            t_differences = np.mod(t_pairs[:, 1] - t_pairs[:, 0], 2 * np.pi)
            half_way_index = int(len(t_differences) / 2)
            first_half = t_differences[:half_way_index]
            second_half = t_differences[half_way_index:]
            self.is_reciprocal = np.all(
                np.isclose(first_half, second_half)
            )  # type: ignore

        # Handle central mode
        central_mode_radius = r_vals[1]
        points_cartesian = np.array([[0.0, 0.0], [central_mode_radius, 0.0]])
        self.add_mode(
            Mode(index=0, mode_boundary=points_cartesian, is_polar=True)
        )

        # Non-central modes
        r_vals = r_vals[1:]
        r_val_pairs = get_pairs(r_vals)
        t_val_pairs = get_pairs(t_vals)

        mode_index = 0
        reciprocal_mode_index = 0

        for t_index, t_val_pair in enumerate(t_val_pairs):
            t_val_pair += self.t_offset
            for r_val_pair in r_val_pairs:
                # Check if points are reciprocal inverse to an already existing
                # mode. If so, use its negative index
                if self.is_reciprocal and t_index >= half_way_index:
                    reciprocal_mode_index -= 1
                    new_mode_index = reciprocal_mode_index
                else:
                    mode_index += 1
                    new_mode_index = mode_index

                r_grid, t_grid = np.meshgrid(r_val_pair, t_val_pair)
                points_polar = np.column_stack(
                    (r_grid.ravel(), t_grid.ravel())
                )
                points_cartesian = polar_to_cartesian(points_polar)
                self.add_mode(
                    Mode(
                        index=new_mode_index,
                        mode_boundary=points_cartesian,
                        is_polar=True,
                    )
                )

    def _handle_cartesian_case(self) -> None:
        """
        Sets up a Cartesian grid of modes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        mode_list_propagating = []
        mode_list_evanescent = []

        dx = self.dx
        dy = self.dy
        x_lim = self.x_lim
        y_lim = self.y_lim

        # Cartesian grids satisfy reciprocity automatically
        self.is_reciprocal = True

        # Set up x-y lattice of rectangular box boundaries
        x_vals = np.arange(dx / 2, 2 * x_lim, dx)
        x_vals = np.concatenate((-x_vals[::-1], x_vals))
        y_vals = np.arange(dy / 2, 2 * y_lim, dy)
        y_vals = -np.concatenate((-y_vals[::-1], y_vals))

        # Filter grids up to x_lim and y_lim
        x_vals = x_vals[np.abs(x_vals) <= x_lim]
        y_vals = y_vals[np.abs(y_vals) <= y_lim]

        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        num_x = len(x_vals)
        num_y = len(y_vals)

        for i in range(num_x - 1):
            for j in range(num_y - 1):
                # Find points that form a box within the lattice
                box_x_vals = x_vals[i : i + 2]
                box_y_vals = y_vals[j : j + 2]
                box_x_grid, box_y_grid = np.meshgrid(box_x_vals, box_y_vals)
                box_points = np.column_stack(
                    (box_x_grid.ravel(), box_y_grid.ravel())
                )

                # Order box_points cyclically
                box_points = order_points(box_points)

                # Check if the box points are all inside or outside of the
                # lattice
                box_r_vals = np.linalg.norm(box_points, axis=1)
                if np.all(box_r_vals >= 1.0):
                    box_wave_type = "evanescent"
                elif np.all(box_r_vals <= 1.0):
                    box_wave_type = "propagating"
                else:
                    box_wave_type = "mixed"

                # Handle cases separately
                # First determine if new mode is to be added depending on the
                # values of box_wave_type and grid_wave_type
                add_new_mode = (
                    self.grid_wave_type == "all"
                    or box_wave_type == "mixed"
                    or box_wave_type == self.grid_wave_type
                )

                # Move to next box if add_new_move evaluates to false
                if not add_new_mode:
                    continue

                # For a mixed box, handle weird edge cases and add
                # modes that incorporate a portion of the circular boundary
                if box_wave_type == "mixed":
                    # Find intersection of circle with box
                    # There must be at least 2 intersection points
                    circle_points = get_polygon_circle_intersection_points(
                        box_points
                    )

                    # Catch bug
                    if circle_points is None:
                        raise ValueError(
                            "Box considered 'mixed' but does not intersect "
                            "the circle."
                        )

                    # If there are more than 2 points, order intersection
                    # points and take the first and last
                    if len(circle_points) > 2:
                        circle_points = order_points(circle_points)
                        circle_points = np.array(
                            [circle_points[0], circle_points[-1]]
                        )

                    # Set up the interior and exterior modes
                    interior_points = box_points[box_r_vals <= 1.0]
                    exterior_points = box_points[box_r_vals >= 1.0]

                    interior_mode_points = np.append(
                        interior_points, circle_points, axis=0
                    )
                    exterior_mode_points = np.append(
                        exterior_points, circle_points, axis=0
                    )

                    # Rotate points
                    interior_mode_points = rotate_points(
                        points=interior_mode_points,
                        rotation_angle=self.t_offset,
                    )
                    exterior_mode_points = rotate_points(
                        points=exterior_mode_points,
                        rotation_angle=self.t_offset,
                    )

                    interior_mode = Mode(
                        mode_boundary=interior_mode_points, is_polar=False
                    )
                    exterior_mode = Mode(
                        mode_boundary=exterior_mode_points, is_polar=False
                    )

                    # Note that both modes get added in the "all" case
                    if self.grid_wave_type in ["propagating", "all"]:
                        mode_list_propagating.append(interior_mode)
                    if self.grid_wave_type in ["evanescent", "all"]:
                        mode_list_evanescent.append(exterior_mode)

                else:
                    # If the box wave type is not mixed, we just add the mode
                    # constructed from the original box points
                    box_points = rotate_points(
                        points=box_points, rotation_angle=self.t_offset
                    )
                    if box_wave_type == "propagating":
                        mode_list_propagating.append(
                            Mode(mode_boundary=box_points, is_polar=False)
                        )
                    else:
                        mode_list_evanescent.append(
                            Mode(mode_boundary=box_points, is_polar=False)
                        )

        # Put modes in dictionaries with correct keys
        # For propagating case the central mode has index 0
        if self.grid_wave_type in ["propagating", "all"]:
            num_modes_propagating = len(mode_list_propagating)
            max_index_propagating = int((num_modes_propagating - 1) / 2)
            self.max_index_propagating = max_index_propagating
            indices = np.arange(
                -max_index_propagating, max_index_propagating + 1
            )
            for index, mode in zip(indices, mode_list_propagating):
                mode.index = index
                self.add_mode(mode, "propagating")

        # Same but for evanescent waves
        # Node that indexing is slightly different as there is no evanescent
        # mode with index 0
        if self.grid_wave_type in ["evanescent", "all"]:
            num_modes_evanescent = len(mode_list_evanescent)
            max_index_evanescent = int(num_modes_evanescent / 2)
            self.max_index_evanescent = max_index_evanescent
            indices = np.arange(1, max_index_evanescent + 1)
            indices = np.concatenate((-indices[::-1], indices))
            for index, mode in zip(indices, mode_list_evanescent):
                mode.index = index
                self.add_mode(mode, "evanescent")

    def _handle_custom_case(self) -> None:
        pass

    def plot(
        self, show_indices: bool = False, show_triangulation: bool = False
    ) -> None:
        """
        Draws the grid of modes.

        Parameters
        ----------
        show_indices : bool
            Will show indices as text on top of modes if True.
        show_triangulation : bool
            Will show the triangulation of all modes in the grid if True.

        Returns
        -------
        None
        """

        # Draw axes and k-space boundary
        ax = set_up_k_space_plot()
        if len(self.modes_propagating) > 0:
            for key, mode in self.modes_propagating.items():
                mode.plot(
                    ax=ax,
                    is_solo=False,
                    show_guidelines=False,
                    mode_color="tab:red",
                    show_index=show_indices,
                    show_triangulation=show_triangulation,
                )
        if len(self.modes_evanescent) > 0:
            for key, mode in self.modes_evanescent.items():
                mode.plot(
                    ax=ax,
                    is_solo=False,
                    show_guidelines=False,
                    mode_color="tab:red",
                    show_index=show_indices,
                    show_triangulation=show_triangulation,
                )
