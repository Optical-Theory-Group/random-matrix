"""This module defines a "ModeGrid" class for use in scattering calculations.

ModeGrid acts as a generator and container for Mode objects.
"""

from typing import Any, Iterable, Self

import numpy as np
import numpy.typing as npt
import skspatial.objects

from random_matrix.mode import Mode
from random_matrix.utils import array_utils, geometry_utils, plotting_utils
from random_matrix.utils.typevars import Numeric


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
        self,
        grid_params: dict[str, Any],
        r_lim: float = 1.0,
        mode_boundary_list: Iterable[npt.NDArray[Numeric]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialises grid of modes. Please use class methods outlined above to
        construct grids.

        Parameters
        ----------

            mode_boundary_list : numpy.array (generator)
                A list containing the bounary points for each mode in the grid.
                Generated from the ModeGridGenerator class.
            grid_params : dict
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

        # Ensure that mode_boundary_list has been provided
        if mode_boundary_list is None:
            raise ValueError("mode_boundary_list not provided.")

        # Make sure that r_lim is greater than 1.0
        if r_lim < 1.0:
            raise ValueError(
                f"Given r_lim value of {r_lim} must be at least 1.0."
            )
        self.r_lim = r_lim

        # Parse global grid parameters contained in grid_params
        is_polar_grid = grid_params.get("is_polar_grid", False)
        grid_wave_type = grid_params.get("grid_wave_type", "propagating")
        self.is_polar_grid = is_polar_grid
        self.grid_wave_type = grid_wave_type

        # Handle polar case separately
        if is_polar_grid:
            # Polar grids may not be reciprocal. It depends on the theta
            # values.
            r_vals = kwargs.get("r_vals", None)
            t_vals = kwargs.get("t_vals", None)
            include_central_mode = grid_params.get(
                "include_central_mode", True
            )

            (
                mode_list_propagating,
                mode_list_evanescent,
            ) = self._handle_polar_case(r_vals, t_vals, include_central_mode)

        else:
            # Cartesian grids with no translational offset from the origin
            # are necessarily reciprocal
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
        grid_params: dict[str, Any],
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
        grid_params : dict
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
            x_vals=x_vals, y_vals=y_vals, grid_params=grid_params
        )

    @classmethod
    def from_xy_vals(
        cls,
        x_vals: npt.NDArray[Numeric],
        y_vals: npt.NDArray[Numeric],
        grid_params: dict[str, Any],
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
        grid_params : dict
            As in __init__

        Returns
        -------
        ModeGrid
            The constructed grid.
        """
        # Check that arrays have been properly given
        cls._validate_input_vals(first_vals=x_vals, second_vals=y_vals)

        # Force grid_type to be cartesian
        grid_params["is_polar_grid"] = False

        # Remove duplicates and ensure arrays are properly sorted
        x_vals = array_utils.remove_duplicate_points(x_vals)
        y_vals = array_utils.remove_duplicate_points(y_vals)
        x_vals = np.sort(x_vals)
        y_vals = np.sort(y_vals)

        # list in which the boundaries of all the modes will all be stored
        mode_boundary_list = cls.generate_rectangles(
            x_vals=x_vals, y_vals=y_vals
        )

        return cls(
            grid_params=grid_params, mode_boundary_list=mode_boundary_list
        )

    @classmethod
    def from_dr_dt(
        cls,
        dr: float,
        dt: float,
        grid_params: dict[str, Any],
        r_lim: float = 1.0,
    ) -> Self:
        # Set up r and t value arrays
        r_vals = np.arange(0.0, 2 * r_lim, dr)
        t_vals = np.arange(0.0, 2 * np.pi, dt)

        # Filter r_vals to be up to limit
        r_vals = r_vals[np.abs(r_vals) <= r_lim]

        return cls.from_rt_vals(
            r_vals=r_vals, t_vals=t_vals, grid_params=grid_params
        )

    @classmethod
    def from_rt_vals(
        cls,
        r_vals: npt.NDArray[Numeric],
        t_vals: npt.NDArray[Numeric],
        grid_params: dict[str, Any],
    ) -> Self:
        # Check that arrays have been properly given
        cls._validate_input_vals(first_vals=r_vals, second_vals=t_vals)

        # Force grid_type to be polar
        grid_params["is_polar_grid"] = True

        # Reduce t values moduli 2PI for consistency
        t_vals = np.mod(t_vals, 2 * np.pi)

        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not array_utils.is_in_array(0.0, r_vals):
            r_vals = np.concatenate(([0.0], r_vals))
        if not array_utils.is_in_array(1.0, r_vals):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not array_utils.is_in_array(0.0, t_vals):
            t_vals = np.concatenate(([0.0], t_vals))
        if not array_utils.is_in_array(2 * np.pi, t_vals):
            t_vals = np.concatenate((t_vals, [2 * np.pi]))

        # Remove duplicates and sort
        r_vals = array_utils.remove_duplicate_points(r_vals)
        t_vals = array_utils.remove_duplicate_points(t_vals)
        r_vals = np.sort(r_vals)
        t_vals = np.sort(t_vals)

        return cls(grid_params=grid_params, r_vals=r_vals, t_vals=t_vals)

    @staticmethod
    def _validate_input_vals(
        first_vals: npt.NDArray[Numeric],
        second_vals: npt.NDArray[Numeric],
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
            f"Maximum |kappa|: {self.r_lim}.\n"
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

        # Stringify index for consistency. Means that the user can give an int
        # if they want
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

        is_reciprocal = array_utils.is_equal_array(
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
        r_vals: npt.NDArray[Numeric],
        t_vals: npt.NDArray[Numeric],
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
            points_cartesian = geometry_utils.rotate_points(
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

            pairs = array_utils.get_pairs(t_vals)
            for first_t, second_t in pairs:
                points_polar = np.array(
                    [[0.0, 0.0], [r, first_t], [r, second_t]]
                )
                points_cartesian = geometry_utils.polar_to_cartesian(
                    points_polar
                )

                # Rotate points
                points_cartesian = geometry_utils.rotate_points(
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
                points_polar = array_utils.vals_to_box(
                    first_vals=box_r_vals, second_vals=box_t_vals
                )

                points_cartesian = geometry_utils.polar_to_cartesian(
                    points_polar
                )
                # Rotate points
                points_cartesian = geometry_utils.rotate_points(
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
        self, mode_boundary_list: Iterable[npt.NDArray[Numeric]]
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

        r_lim = self.r_lim

        # Lists for storing modes
        # Will later be converted into dictionaries
        mode_list_propagating = []
        mode_list_evanescent = []

        for boundary_points in mode_boundary_list:
            boundary_r_vals = np.linalg.norm(boundary_points, axis=1)

            # These points exclude those lying on the circles
            interior_points = boundary_points[boundary_r_vals < 1.0]

            exterior_points = boundary_points[boundary_r_vals > 1.0]
            temp_r_vals = np.linalg.norm(exterior_points, axis=1)
            exterior_points = exterior_points[temp_r_vals < r_lim]

            # Find out where the mode crosses the circles of radii
            # r = 1.0 and r = r_lim
            (
                mode_wave_type,
                inner_circle_crossings,
                outer_circle_crossings,
            ) = self._get_circle_crossings(boundary_points)

            # Clean up circle crossings to remove unnecessary points
            if (
                inner_circle_crossings is not None
                and len(inner_circle_crossings) > 2
            ):
                inner_circle_crossings = (
                    geometry_utils.get_angularly_separated_edge_points(
                        inner_circle_crossings
                    )
                )
            if (
                outer_circle_crossings is not None
                and len(outer_circle_crossings) > 2
            ):
                outer_circle_crossings = (
                    geometry_utils.get_angularly_separated_edge_points(
                        outer_circle_crossings
                    )
                )

            # There are four possible cases
            is_interior_mode = False
            is_exterior_mode = False

            match (mode_wave_type):
                case {
                    "crosses_inner_circle": True,
                    "crosses_outer_circle": True,
                }:
                    # In this case we make two modes and include the exterior
                    # circle points
                    interior_mode_points = np.append(
                        interior_points, inner_circle_crossings, axis=0
                    )
                    exterior_mode_points = np.append(
                        np.append(
                            exterior_points, inner_circle_crossings, axis=0
                        ),
                        outer_circle_crossings,
                        axis=0,
                    )
                    is_interior_mode = True
                    is_exterior_mode = True

                case {
                    "crosses_inner_circle": True,
                    "crosses_outer_circle": False,
                }:
                    # In this case we make two modes, but don't include the
                    # exterior circle points
                    interior_mode_points = np.append(
                        interior_points, inner_circle_crossings, axis=0
                    )
                    exterior_mode_points = np.append(
                        exterior_points, inner_circle_crossings, axis=0
                    )
                    is_interior_mode = True
                    is_exterior_mode = True

                case {
                    "crosses_inner_circle": False,
                    "crosses_outer_circle": True,
                }:
                    # In this case we make one evanescent mode, including
                    # the outer circle points
                    exterior_mode_points = np.append(
                        exterior_points, outer_circle_crossings, axis=0
                    )
                    is_exterior_mode = True

                case {
                    "crosses_inner_circle": False,
                    "crosses_outer_circle": False,
                }:
                    # In this case we only make at most one mode, but it
                    # depends on where the shape is. This can be determined by
                    # any of the r_vals
                    if boundary_r_vals[0] < 1.0:
                        interior_mode_points = interior_points
                        is_interior_mode = True

                    elif (
                        boundary_r_vals[0] > 1.0 and boundary_r_vals[0] < r_lim
                    ):
                        exterior_mode_points = exterior_points
                        is_exterior_mode = True
                    else:
                        # Mode is completely outside of r_lim
                        continue

            # Make the mode and add it to the lists
            # Note that both modes get added in the "all" case
            if is_interior_mode and self.grid_wave_type in [
                "propagating",
                "all",
            ]:
                interior_mode = Mode(
                    mode_boundary=interior_mode_points, is_polar=False
                )
                mode_list_propagating.append(interior_mode)

            if is_exterior_mode and self.grid_wave_type in [
                "evanescent",
                "all",
            ]:
                exterior_mode = Mode(
                    mode_boundary=exterior_mode_points, is_polar=False
                )
                mode_list_evanescent.append(exterior_mode)

        return mode_list_propagating, mode_list_evanescent

    def _get_circle_crossings(self, boundary_points):
        r_lim = self.r_lim

        inner_circle_crossings = (
            geometry_utils.get_polygon_circle_intersection_points(
                boundary_points, skspatial.objects.Circle([0.0, 0.0], 1.0)
            )
        )

        outer_circle_crossings = (
            geometry_utils.get_polygon_circle_intersection_points(
                boundary_points, skspatial.objects.Circle([0.0, 0.0], r_lim)
            )
        )

        mode_wave_type = {
            "crosses_inner_circle": False,
            "crosses_outer_circle": False,
        }

        if inner_circle_crossings is not None:
            mode_wave_type["crosses_inner_circle"] = True
        if outer_circle_crossings is not None:
            mode_wave_type["crosses_outer_circle"] = True

        return mode_wave_type, inner_circle_crossings, outer_circle_crossings

    def _get_mode_wave_type(self, boundary_points):
        mode_wave_type = {
            "crosses_inner_circle": False,
            "crosses_outer_circle": False,
            "pure_wave_mode_type": None,
        }
        r_lim = self.r_lim
        boundary_r_vals = np.linalg.norm(boundary_points, axis=1)

        # There are 5 possible cases for the r values in boundary_r_vals
        # 1: r < 1.0            in the interior of the unit circle
        # 2: r = 1.0            on the unit circle
        # 3: 1.0 < r < r_lim    in the propagating-evanescent annulus
        # 4: r = r_lim          on the circle of radius r_lim
        # 5: r > r_lim          in the exterior of the outer circle

        boundary_point_cases = []
        for r_val in boundary_r_vals:
            if r_val < 1.0:
                boundary_point_cases.append("1")
            elif np.isclose(r_val, 1.0):
                boundary_point_cases.append("2")
            elif r_val > 1.0 and r_val < r_lim:
                boundary_point_cases.append("3")
            elif np.isclose(r_val, r_lim):
                boundary_point_cases.append("4")
            else:
                boundary_point_cases.append("5")

        # Check which circles the mode has crossed
        # We have crossed the inner circle if "1" is present and any of
        # "3", "4" or "5" are present

        crosses_inner_circle = "1" in boundary_point_cases and set(
            ["3", "4", "5"]
        ).intersection(set(boundary_r_vals))
        # Similarly, we have crossed the outer circle if "5" is present and any of
        # "1", "2" or "3" are present

        crosses_outer_circle = "5" in boundary_point_cases and set(
            ["1", "2", "3"]
        ).intersection(set(boundary_r_vals))

        if crosses_inner_circle:
            mode_wave_type["crosses_inner_circle"] = True
        if crosses_outer_circle:
            mode_wave_type["crosses_outer_circle"] = True

        # If we have crossed neither circle, we must be a pure mode
        is_pure_propagating = (
            not crosses_inner_circle and not crosses_outer_circle
        ) and (set(boundary_point_cases).intersection(set(["1", "2"])))

        is_pure_evanescent = (
            not crosses_inner_circle and not crosses_outer_circle
        ) and (set(boundary_point_cases).intersection(set(["2", "3", "4"])))

        if is_pure_propagating:
            mode_wave_type["pure_wave_mode_type"] = "propagating"
        if is_pure_evanescent:
            mode_wave_type["pure_wave_mode_type"] = "evanescent"

        return mode_wave_type

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

        reciprocal_indices = array_utils.sort_by_reference_list(
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

            if array_utils.is_equal_array(points, -other_points):
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
        ax = plotting_utils.set_up_k_space_plot()
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
