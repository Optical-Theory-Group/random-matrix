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
    is_equal_array,
    sort_by_reference_list,
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
        is_polar_grid = grid_data.get("is_polar_grid", False)
        grid_wave_type = grid_data.get("grid_wave_type", "propagating")
        self.t_offset = t_offset
        self.is_polar_grid = is_polar_grid
        self.grid_wave_type = grid_wave_type

        # Parse specialised parameters in grid_data
        if is_polar_grid:
            # Polar grids may not be reciprocal. It depends on the theta
            # values.
            r_vals = kwargs.get("r_vals", None)
            t_vals = kwargs.get("t_vals", None)
            include_central_mode = grid_data.get("include_central_mode", True)

            (
                mode_list_propagating,
                mode_list_evanescent,
            ) = self._handle_polar_case(r_vals, t_vals, include_central_mode)

        else:
            # Cartesian grids with no translational offset from the origin
            # are necessarily reciprocal
            mode_boundary_list = kwargs.get("mode_boundary_list", None)

            (
                mode_list_propagating,
                mode_list_evanescent,
            ) = self._handle_general_case(mode_boundary_list)

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
        grid_data["is_polar_grid"] = False

        # Remove duplicates and ensure arrays are properly sorted
        x_vals = remove_duplicate_points(x_vals)
        y_vals = remove_duplicate_points(y_vals)
        x_vals = np.sort(x_vals)
        y_vals = np.sort(y_vals)

        # list in which the boundaries of all the modes will all be stored
        mode_boundary_list = cls.generate_rectangles(
            x_vals=x_vals, y_vals=y_vals
        )

        return cls(grid_data=grid_data, mode_boundary_list=mode_boundary_list)

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
        grid_data["is_polar_grid"] = True

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

    @classmethod
    def from_tiling(cls, grid_data, tiling_shape="triangle", side_length=0.4):
        grid_data["is_polar_grid"] = False

        match tiling_shape:
            case "triangle":
                mode_boundary_list = cls.generate_triangles(
                    side_length=side_length, x_lim=1.5, y_lim=1.5
                )
            case "hexagon":
                print("hi")
                mode_boundary_list = cls.generate_hexagons(
                    side_length=side_length, x_lim=1.5, y_lim=1.5
                )

        return cls(grid_data=grid_data, mode_boundary_list=mode_boundary_list)

    def generate_polar_regions(self, r_vals, y_vals, r_lim):
        pass

    @staticmethod
    def generate_rectangles(x_vals, y_vals):
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
                yield box_points

    @staticmethod
    def generate_triangles(side_length, x_lim, y_lim):
        # The repeating unit for our triangular lattice is
        #
        #   _____
        #   \  / \
        #    \/___\  <--- center at mid-point on left
        #    /\   /
        #   /__\_/
        #
        #
        # The "center" is at the midpoint of the base of the pair of
        # triangles in the middle row of sides.

        # s is the side length and h is the vertical height of a triangle
        s = side_length
        h = s * np.sqrt(3) / 2

        # Set up cartesian lattice for centers
        dx = s
        dy = 2 * h

        x_vals = np.arange(0.0, x_lim + dx, dx)
        x_vals = np.concatenate((-x_vals[1:][::-1], x_vals))
        y_vals = np.arange(0.0, y_lim + dy, dy)
        y_vals = np.concatenate((-y_vals[1:][::-1], y_vals))

        for x in x_vals:
            for y in y_vals:
                yield np.array(
                    [[x - s / 2, y + h], [x, y], [x + s / 2, y + h]]
                )
                yield np.array([[x + s / 2, y + h], [x, y], [x + s, y]])
                yield np.array(
                    [[x - s / 2, y - h], [x, y], [x + s / 2, y - h]]
                )
                yield np.array([[x, y], [x + s / 2, y - h], [x + s, y]])

    @staticmethod
    def generate_hexagons(side_length, x_lim, y_lim):
        # The repeating unit for our triangular lattice is
        #
        #    ____
        #   /    \
        #  /      \____  <--- center at mid-point of first hexagon
        #  \      /    \
        #   \____/      \
        #        \      /
        #         \____/
        #
        # The "center" is at the midpoint of the left hexagon

        # s is the side length and h is the vertical height of a hexagon
        s = side_length
        h = s * np.sqrt(3) / 2

        # Set up cartesian lattice for centers
        dx = 3 * s
        dy = 2 * h

        x_vals = np.arange(0.0, x_lim + dx, dx)
        x_vals = np.concatenate((-x_vals[1:][::-1], x_vals))
        y_vals = np.arange(0.0, y_lim + dy, dy)
        y_vals = np.concatenate((-y_vals[1:][::-1], y_vals))

        for x in x_vals:
            for y in y_vals:
                yield np.array(
                    [
                        [x - s, y],
                        [x - s / 2, y + h],
                        [x + s / 2, y + h],
                        [x + s, y],
                        [x + s / 2, y - h],
                        [x - s / 2, y - h],
                    ]
                )
                yield np.array(
                    [
                        [x + s / 2, y - h],
                        [x + s, y],
                        [x + 2 * s, y],
                        [x + 5 / 2 * s, y - h],
                        [x + 2 * s, y - 2 * h],
                        [x + s, y - 2 * h],
                    ]
                )

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
            f"Polar grid: {self.is_polar_grid},\n"
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

        points_array = np.empty((0, 2))
        for mode in mode_list:
            points_array = np.vstack((points_array, mode.points))

        inverted_points_array = -np.copy(points_array)

        is_reciprocal = is_equal_array(
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
        r_vals: Vector[np.float32],
        t_vals: Vector[np.float32],
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

        # Lists for storing modes
        # Will later be converted into dictionaries
        mode_list_propagating = []
        mode_list_evanescent = []

        # Handle central mode if included
        if include_central_mode:
            central_mode_radius = r_vals[1]
            points_cartesian = np.array(
                [
                    [central_mode_radius, 0.0],
                    [0.0, central_mode_radius],
                    [0.0, -central_mode_radius],
                    [-central_mode_radius, 0.0],
                ]
            )

            # Rotate points
            points_cartesian = rotate_points(
                points=points_cartesian,
                rotation_angle=self.t_offset,
            )

            mode_list_propagating.append(
                Mode(mode_boundary=points_cartesian, is_polar=True)
            )

        else:
            # If central mode is not included, we make a fan of 3-point
            # polar modes about the origin
            r = r_vals[1]

            pairs = get_pairs(t_vals)
            for first_t, second_t in pairs:
                points_polar = np.array(
                    [[0.0, 0.0], [r, first_t], [r, second_t]]
                )
                points_cartesian = polar_to_cartesian(points_polar)

                # Rotate points
                points_cartesian = rotate_points(
                    points=points_cartesian,
                    rotation_angle=self.t_offset,
                )

                mode_list_propagating.append(
                    Mode(mode_boundary=points_cartesian, is_polar=True)
                )

        r_vals = r_vals[1:]

        num_r = len(r_vals)
        num_t = len(t_vals)

        is_evanescent = False

        for i in range(num_r - 1):
            # Trigger is_evanescent flag once r goes past 1.0
            if np.isclose(r_vals[i], 1.0) and not is_evanescent:
                is_evanescent = True

            for j in range(num_t - 1):
                # Find points that form a "polar box" within the lattice
                box_r_vals = r_vals[i : i + 2]
                box_t_vals = t_vals[j : j + 2]
                points_polar = vals_to_box(
                    first_vals=box_r_vals, second_vals=box_t_vals
                )

                points_cartesian = polar_to_cartesian(points_polar)
                # Rotate points
                points_cartesian = rotate_points(
                    points=points_cartesian,
                    rotation_angle=self.t_offset,
                )
                if is_evanescent:
                    mode_list_evanescent.append(
                        Mode(mode_boundary=points_cartesian, is_polar=True)
                    )

                else:
                    mode_list_propagating.append(
                        Mode(mode_boundary=points_cartesian, is_polar=True)
                    )

        return mode_list_propagating, mode_list_evanescent

    def _handle_general_case(
        self, mode_boundary_list: list[Matrix[np.float32]]
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

        for boundary_points in mode_boundary_list:
            # Check if the box points are all inside or outside of the
            # lattice
            boundary_r_vals = np.linalg.norm(boundary_points, axis=1)
            if np.all(boundary_r_vals >= 1.0):
                mode_wave_type = "evanescent"
            elif np.all(boundary_r_vals <= 1.0):
                mode_wave_type = "propagating"
            else:
                mode_wave_type = "mixed"

            # Handle cases separately
            # First determine if new mode is to be added depending on the
            # values of box_wave_type and grid_wave_type
            add_new_mode = (
                self.grid_wave_type == "all"
                or mode_wave_type == "mixed"
                or mode_wave_type == self.grid_wave_type
            )

            # Move to next box if add_new_move evaluates to false
            if not add_new_mode:
                continue

            # For a mixed box, handle weird edge cases and add
            # modes that incorporate a portion of the circular boundary
            if mode_wave_type == "mixed":
                # Find intersection of circle with box
                # There must be at least 2 intersection points
                circle_points = get_polygon_circle_intersection_points(
                    boundary_points
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
                interior_points = boundary_points[boundary_r_vals <= 1.0]
                exterior_points = boundary_points[boundary_r_vals >= 1.0]

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
                boundary_points = rotate_points(
                    points=boundary_points, rotation_angle=self.t_offset
                )
                if mode_wave_type == "propagating":
                    mode_list_propagating.append(
                        Mode(mode_boundary=boundary_points, is_polar=False)
                    )
                else:
                    mode_list_evanescent.append(
                        Mode(mode_boundary=boundary_points, is_polar=False)
                    )

        return mode_list_propagating, mode_list_evanescent

    def _set_up_mode_dictionary(
        self,
        mode_list: list[Mode],
        list_wave_type: str,
        is_reciprocal: bool | np.bool_,
        contains_central_mode: bool | np.bool_,
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

        if self.is_reciprocal:
            indices = self._get_reciprocal_indices(mode_list)
        else:
            # If not reciprocal, just number the modes sequentially.
            # Negative indices are meaningless.
            num_modes = len(mode_list)
            indices = np.arange(1, num_modes + 1)

        # Add modes to dictionary
        for index, mode in zip(indices, mode_list):
            mode.index = index
            self.add_mode(mode, list_wave_type)

    def _get_reciprocal_indices(self, mode_list: list[Mode]) -> list[int]:
        # Reciprocal indices will be the indices we use later in the
        # dictionary. already_handled_modes will keep track of which
        # modes have been dealt with. In fact, this list will also tell
        # us what modes the reciprocal indices line up with
        reciprocal_indices = []
        already_handled_modes = []

        running_index = 1

        for mode_index, mode in enumerate(mode_list):
            # Check if we have already dealt with this mode
            if mode_index in already_handled_modes:
                continue

            # This mode is new. Find the index of its reciprocal partner
            partner_index = self._get_reciprocal_partner_index(
                mode=mode, mode_list=mode_list
            )

            if mode_index == partner_index:
                # In this case, the mode must be the central mode
                reciprocal_indices.append(0)
                already_handled_modes.append(mode_index)
            else:
                # General case
                reciprocal_indices.append(running_index)
                reciprocal_indices.append(-running_index)
                already_handled_modes.append(mode_index)
                already_handled_modes.append(partner_index)

                running_index += 1

        reciprocal_indices = sort_by_reference_list(
            reciprocal_indices, already_handled_modes
        )
        return reciprocal_indices

    def _get_reciprocal_partner_index(
        self, mode: Mode, mode_list: list[Mode]
    ) -> int | None:
        if not self.is_reciprocal:
            raise ValueError(
                "Attempted to find reciprocal partners in "
                "non-reciprocal grid."
            )

        points = mode.points
        for index, other_mode in enumerate(mode_list):
            other_points = other_mode.points

            # Check that points are the same shape
            # If not, move to the next one.
            if np.shape(points) != np.shape(other_points):
                continue

            if is_equal_array(points, -other_points):
                return index

        return None

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
