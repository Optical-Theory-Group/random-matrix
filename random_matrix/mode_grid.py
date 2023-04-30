"""This module defines a "ModeGrid" class for use in scattering calculations.

ModeGrid acts as a generator and container for Mode objects.
"""

from typing import Any, Iterator, Self

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
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

    # --------------------------------------------------------------------------
    # Constructor method
    # --------------------------------------------------------------------------

    def __init__(
        self,
        r_lim: float,
        mode_list: Iterator[Mode] | None = None,
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

        if mode_list is None:
            raise ValueError("Mode list not provided.")
        
        self.r_lim = r_lim

        mode_list_propagating = []
        mode_list_evanescent = []

        for mode in mode_list:
            if mode.wave_type == "propagating":
                mode_list_propagating.append(mode)
            else:
                mode_list_evanescent.append(mode)

        num_modes_propagating = len(mode_list_propagating)
        num_modes_evanescent = len(mode_list_evanescent)
        
        if num_modes_propagating > 0 and num_modes_evanescent > 0:
            self.grid_wave_type = "all"
        elif num_modes_propagating > 0:
            self.grid_wave_type = "propagating"
        else:
            self.grid_wave_type = "evanescent"

        # Check if resultant grid is reciprocal or not
        combined_modes = mode_list_propagating + mode_list_evanescent
        is_reciprocal_grid = self._is_reciprocal_mode_list(combined_modes)
        self.is_reciprocal = is_reciprocal_grid

        # Check if the propagating portion of the modes contains a central
        # mode or not
        contains_central_mode = self._contains_central_mode(
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

    # --------------------------------------------------------------------------
    # Input validation and processing
    # --------------------------------------------------------------------------

    def _is_reciprocal_mode_list(self, mode_list: list[Mode]) -> bool:
        for mode_index, mode in enumerate(mode_list):
            partner_index = self._get_reciprocal_partner_index(
                mode=mode, mode_list=mode_list
            )
            # If we find the partner, keep going
            if partner_index is not None:
                continue

            # If we haven't found a partner, it can't be a reciprocal grid
            return False

        # If we reach here, every mode must have found a partner
        return True

    @staticmethod
    def _contains_central_mode(mode_list: list[Mode]) -> bool:
        for mode in mode_list:
            if mode.is_central:
                return True
        return False

    def _get_reciprocal_partner_index(
        self, mode: Mode, mode_list: list[Mode]
    ) -> int | None:
        mode_boundary = mode.boundary
        for partner_index, partner in enumerate(mode_list):
            partner_boundary = partner.boundary
            inverted_partner_boundary = -partner_boundary

            # Check that points are the same shape
            # If not, move to the next one.
            if np.shape(mode_boundary) != np.shape(inverted_partner_boundary):
                continue

            is_correct_partner = array_utils.is_equal_array(
                mode_boundary, inverted_partner_boundary
            )

            # Check equality
            if array_utils.is_equal_array(
                mode_boundary, inverted_partner_boundary
            ):
                return partner_index

        return None

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
            self._add_mode(mode, list_wave_type)

    def _add_mode(
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

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Object representations
    # --------------------------------------------------------------------------

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
            f"Reciprocal: {self.is_reciprocal},\n"
            f"Number of propagating modes: {len(self.modes_propagating)},\n"
            f"Number of evanescent modes: {len(self.modes_evanescent)},\n"
            f"Maximum |kappa|: {self.r_lim}\n"
        )
        return output

    def plot(
        self,
        show_indices: bool = False,
        show_triangulation: bool = False,
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

        # Set up k space plot
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # x and y axes
        plotting_utils.draw_ray(
            ax, r_min=-1, theta=0, linestyle="-", color="black", alpha=0.5
        )
        plotting_utils.draw_ray(
            ax,
            r_min=-1,
            theta=np.pi / 2,
            linestyle="-",
            color="black",
            alpha=0.5,
        )

        propagating_mode_color = "red"
        evanescent_mode_color = "blue"
        triangulation_color = "tab:blue"

        if len(self.modes_propagating) > 0:
            for key, mode in self.modes_propagating.items():
                mode.plot(
                    ax=ax,
                    boundary_color=propagating_mode_color,
                    triangulation_color=triangulation_color,
                    index_color=propagating_mode_color,
                    show_index=show_indices,
                    show_triangulation=show_triangulation,
                )

        if len(self.modes_evanescent) > 0:
            for key, mode in self.modes_evanescent.items():
                mode.plot(
                    ax=ax,
                    boundary_color=evanescent_mode_color,
                    triangulation_color=triangulation_color,
                    index_color=evanescent_mode_color,
                    show_index=show_indices,
                    show_triangulation=show_triangulation,
                )

        plotting_utils.draw_circle(ax)
        plotting_utils.draw_circle(ax, r=self.r_lim)
        