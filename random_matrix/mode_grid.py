"""This module defines a "ModeGrid" class for use in scattering calculations.

ModeGrid acts as a generator and container for Mode objects.
"""

from typing import Any, Self

import numpy as np

from random_matrix.types.array_types import Vector, Matrix
from random_matrix.mode import Mode
from random_matrix.utils.array_utils import (
    get_pairs,
    vals_to_box,
    is_in_array,
    remove_duplicate_points,
    are_equal,
)
from random_matrix.utils.geometry_utils import (
    get_polygon_circle_intersection_points,
    order_points,
    polar_to_cartesian,
    rotate_points,
    get_angularly_separated_edge_points,
)
from random_matrix.utils.plotting_utils import set_up_k_space_plot


class ModeGrid:
    """
    A class used to represent a grid of modes. To construct a grid, please use
    the

    from_dx_dy
    from_xy_vals
    from_dr_dt
    from_rt_vals

    class methods.

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

    def __init__(
        self, grid_data: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """
        Initialises grid of modes. Please use class methods outlined above to
        construct grids.

        Parameters
        ----------
        grid_data : dict
            A dictionary containing data that is used to generate the grid.
            This should contain the following three key-value pairs:

            "t_offset"
                Angle that rotates the entire grid.
            "grid_type"
                Type of constructer to be used. Options are
                "polar", "cartesian"
            "grid_wave_type"
                Type of modes to be included in the grid. Options are
                "propagating", "evanescent"

        Returns
        -------
        None
        """

        if grid_data is None:
            raise ValueError("grid_data not provided")

        # Parse global grid parameters contained in grid_data
        t_offset = grid_data.get("t_offset", 0.0)
        grid_type = grid_data.get("grid_type", "custom")
        grid_wave_type = grid_data.get("grid_wave_type", "propagating")
        self.t_offset = t_offset
        self.grid_type = grid_type
        self.grid_wave_type = grid_wave_type

        # Parse specialised parameters in grid_data
        match grid_type:
            case "cartesian":
                # Cartesian grids with no translational offset from the origin
                # are necessarily reciprocal
                x_vals = kwargs.get("x_vals", None)
                y_vals = kwargs.get("y_vals", None)

                (
                    mode_list_propagating,
                    mode_list_evanescent,
                ) = self._handle_cartesian_case(x_vals, y_vals)

            case "polar":
                # Polar grids may not be reciprocal. It depends on the theta
                # values.
                r_vals = kwargs.get("r_vals", None)
                t_vals = kwargs.get("t_vals", None)
                include_central_mode = grid_data.get(
                    "include_central_mode", True
                )

                (
                    mode_list_propagating,
                    mode_list_evanescent,
                ) = self._handle_polar_case(
                    r_vals, t_vals, include_central_mode
                )

            case "random":
                pass
            case "custom":
                pass
            case _:
                raise ValueError(
                    "Incorrect grid_data formatting."
                    "Please refer to the documentation in __init__."
                )

        combined_modes = mode_list_propagating + mode_list_evanescent

        # Check if resultant grid is reciprocal or not
        is_reciprocal_grid = self.is_reciprocal_mode_list(combined_modes)
        self.is_reciprocal = is_reciprocal_grid

        # Check if the propagating portion of the modes contains a central
        # mode or not
        contains_central_mode = self.contains_central_mode(
            mode_list_propagating
        )

        # Put modes in dictionaries with correct keys
        self.modes_propagating: dict[str, Mode] = {}
        self.modes_evanescent: dict[str, Mode] = {}

        if self.grid_wave_type in ["propagating", "all"]:
            self._set_up_mode_dictionary(
                mode_list=mode_list_propagating,
                list_wave_type="propagating",
                is_reciprocal=is_reciprocal_grid,
                contains_central_mode=contains_central_mode,
            )

        # Evanescent modes cannot contain a central mode by definition
        if self.grid_wave_type in ["evanescent", "all"]:
            self._set_up_mode_dictionary(
                mode_list=mode_list_evanescent,
                list_wave_type="evanescent",
                is_reciprocal=is_reciprocal_grid,
                contains_central_mode=False,
            )

    @classmethod
    def from_dx_dy(
        cls,
        dx: float,
        dy: float,
        grid_data: dict[str, Any],
        x_lim: float = 1.2,
        y_lim: float = 1.2,
    ) -> Self:
        """
        Initialises Cartesian grid from dx and dy.

        Parameters
        ----------
        dx : float
            Spacing in the kx direction.
        dy : float
            Spacing in the ky direction.
        grid_data : dict
            As in __init__
        x_lim : float
            Upper limit for the x values in the grid. The grid will be
            filtered so that any x values generated from the provided dx
            such that x > x_lim will be discarded.
        y_lim : float
            Same as x_lim, but for y

        Returns
        -------
        ModeGrid
            The constructed grid.
        """

        # Set up x-y lattice of rectangular box boundaries
        x_vals = np.arange(dx / 2, 2 * x_lim, dx)
        x_vals = np.concatenate((-x_vals[::-1], x_vals))
        y_vals = np.arange(dy / 2, 2 * y_lim, dy)
        y_vals = -np.concatenate((-y_vals[::-1], y_vals))

        # Filter grids up to x_lim and y_lim
        x_vals = x_vals[np.abs(x_vals) <= x_lim]
        y_vals = y_vals[np.abs(y_vals) <= y_lim]

        return cls.from_xy_vals(
            x_vals=x_vals, y_vals=y_vals, grid_data=grid_data
        )

    @classmethod
    def from_xy_vals(
        cls,
        x_vals: Vector[np.float32],
        y_vals: Vector[np.float32],
        grid_data: dict[str, Any],
    ) -> Self:
        """
        Initialises Cartesian grid from grids of x and y values.

        Parameters
        ----------
        x_vals : np.ndarray
            A mesh grid containing the x_coordinates of boundaries of
            rectangles in the lattice.
        y_vals : np.ndarray
            Same, but for y coordinates.
        grid_data : dict
            As in __init__

        Returns
        -------
        ModeGrid
            The constructed grid.
        """
        # Check that arrays have been properly given
        cls._validate_input_vals(first_vals=x_vals, second_vals=y_vals)

        # Force grid_type to be cartesian
        grid_data["grid_type"] = "cartesian"

        # Remove duplicates and ensure arrays are properly sorted
        x_vals = remove_duplicate_points(x_vals)
        y_vals = remove_duplicate_points(y_vals)
        x_vals = np.sort(x_vals)
        y_vals = np.sort(y_vals)

        return cls(grid_data=grid_data, x_vals=x_vals, y_vals=y_vals)

    @classmethod
    def from_dr_dt(
        cls,
        dr: float,
        dt: float,
        grid_data: dict[str, Any],
        r_lim: float = 1.0,
    ) -> Self:
        # Set up r and t value arrays
        r_vals = np.arange(0.0, 2 * r_lim, dr)
        t_vals = np.arange(0.0, 2 * np.pi, dt)

        # Filter r_vals to be up to limit
        r_vals = r_vals[np.abs(r_vals) <= r_lim]

        return cls.from_rt_vals(
            r_vals=r_vals, t_vals=t_vals, grid_data=grid_data
        )

    @classmethod
    def from_rt_vals(
        cls,
        r_vals: Vector[np.float32],
        t_vals: Vector[np.float32],
        grid_data: dict[str, Any],
    ) -> Self:
        # Check that arrays have been properly given
        cls._validate_input_vals(first_vals=r_vals, second_vals=t_vals)

        # Force grid_type to be polar
        grid_data["grid_type"] = "polar"

        # Reduce t values moduli 2PI for consistency
        t_vals = np.mod(t_vals, 2 * np.pi)

        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not is_in_array(0.0, r_vals):
            r_vals = np.concatenate(([0.0], r_vals))
        if not is_in_array(1.0, r_vals):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not is_in_array(0.0, t_vals):
            t_vals = np.concatenate(([0.0], t_vals))
        if not is_in_array(2 * np.pi, t_vals):
            t_vals = np.concatenate((t_vals, [2 * np.pi]))

        # Remove duplicates and sort
        r_vals = remove_duplicate_points(r_vals)
        t_vals = remove_duplicate_points(t_vals)
        r_vals = np.sort(r_vals)
        t_vals = np.sort(t_vals)

        return cls(grid_data=grid_data, r_vals=r_vals, t_vals=t_vals)

    @staticmethod
    def _validate_input_vals(
        first_vals: Vector[np.float32],
        second_vals: Vector[np.float32],
    ) -> None:
        # Generic checks
        if first_vals is None or second_vals is None:
            raise ValueError("Input vals have not been provided.")

        if not (
            isinstance(first_vals, np.ndarray)
            and isinstance(second_vals, np.ndarray)
        ):
            raise ValueError(
                "For automatic grid generation, input vals must be numpy "
                "arrays."
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

    @staticmethod
    def is_reciprocal_mode_list(mode_list: list[Mode]) -> np.bool_:
        # Get inverted list of modes. dtype is object because different modes
        # may be mode from different numbers of points
        points_array = np.concatenate(
            np.array([mode.points for mode in mode_list], dtype=object)
        )
        inverted_points_array = -np.copy(points_array)
        is_reciprocal = are_equal(
            first_array=points_array, second_array=inverted_points_array
        )
        return is_reciprocal

    @staticmethod
    def contains_central_mode(mode_list: list[Mode]) -> np.bool_:
        for mode in mode_list:
            if mode.is_central_mode:
                return np.bool_(True)
        return np.bool_(False)

    def _handle_polar_case(
        self,
        r_vals: Matrix[np.float32],
        t_vals: Matrix[np.float32],
        include_central_mode: bool = True,
    ) -> tuple[list[Mode], list[Mode]]:
        """
        Sets up a Polar grid of modes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

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

    def _handle_cartesian_case(
        self, x_vals: Matrix[np.float32], y_vals: Matrix[np.float32]
    ) -> tuple[list[Mode], list[Mode]]:
        """
        Sets up a Cartesian grid of modes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Lists for storing modes
        # Will later be converted into dictionaries
        mode_list_propagating = []
        mode_list_evanescent = []

        num_x = len(x_vals)
        num_y = len(y_vals)

        for i in range(num_x - 1):
            for j in range(num_y - 1):
                # Find points that form a box within the lattice
                box_x_vals = x_vals[i : i + 2]
                box_y_vals = y_vals[j : j + 2]
                box_points = vals_to_box(
                    first_vals=box_x_vals, second_vals=box_y_vals
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
                        circle_points = get_angularly_separated_edge_points(
                            circle_points
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

        return mode_list_propagating, mode_list_evanescent

    def _set_up_mode_dictionary(
        self,
        mode_list: list[Mode],
        list_wave_type: str,
        is_reciprocal: np.bool_,
        contains_central_mode: np.bool_,
    ) -> None:
        """
        Sets up dictionaries of modes with correct indices. Note that input
        is assumed to be from a cartesian or polar grid generation

        Parameters
        ----------
        mode_list : list[Mode]
            List of modes from cartesian or polar generators.
        list_wave_type : str
            Type of mode in list. Either "propagating" or "evanescent".

        Returns
        -------
        None
        """
        num_modes = len(mode_list)

        if self.is_reciprocal:
            # In reciprocal case, modes take negative indices
            # Whether or not the 0 index is used, however, depends on whether
            # a central mode is present.
            if contains_central_mode:
                max_index = int((num_modes - 1) / 2)
                indices = np.arange(-max_index, max_index + 1)
            else:
                max_index = int(num_modes / 2)
                self.max_index_evanescent = max_index
                indices = np.arange(1, max_index + 1)
                indices = np.concatenate((-indices[::-1], indices))

        else:
            # If not reciprocal, just number the modes sequentially.
            # Negative indices are meaningless.
            indices = np.arange(0, num_modes)

        # Add modes to dictionary
        for index, mode in zip(indices, mode_list):
            mode.index = index
            self.add_mode(mode, list_wave_type)

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
