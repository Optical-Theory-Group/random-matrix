"""ModeGrid class for use in scattering calculations.

ModeGrid serves as a container for Mode objects.
"""

from dataclasses import InitVar, dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from random_matrix.modes.mode import Mode
from random_matrix.utils import array_utils, plotting_utils
from random_matrix.utils.types import Numeric


@dataclass(slots=True)
class ModeGrid:
    """A class used to represent a grid of modes.

    To construct a grid, please do not call this class directly, but use
    functions from the mode_grid_generator module.

    Attributes
    ----------
        r_lim : float
            radial extend of the grid of modes
        mode_list : list[Mode]
            List of modes that will be stored in the grid. Note that this list
            is only used for initialising objects. After an objet has been
            created, modes are stored the "modes" attribute.
        is_reciprocal : bool
            Tracks whether or not the set of modes satisfies reciprocity.
        modes : dict
            Main dictionary in which the modes are stored. Keys are tuples
            of the form (index, wave_type).

            N.B.
            To access modes, it is advised that you use the by_index method.
        num_propagating : int
            Number of propagating modes. Computed when needed.
        num_evanescent : int
            Number of evanescent modes. Computed when needed.

    Methods
    -------
        by_index
            Obtain a mode from the grid by refernencing its index. Since
            indices for propagating and evanescent waves are independent, one
            should also specify the type of mode that is desired, i.e.
            propagating or evanescent. Note that "p" and "e" can also be passed
            as the second argument instead of "propagating" and "evanescent".
        plot
            Plots the grid of modes.

    """

    r_lim: float

    mode_list: InitVar[list[Mode]] = None

    is_reciprocal: bool = field(init=False)
    modes: dict[tuple[str, str], Mode] = field(init=False)

    rec_mat: Numeric = field(init=False)

    # --------------------------------------------------------------------------
    # Constructor method
    # --------------------------------------------------------------------------

    def __post_init__(self, mode_list: list[Mode]) -> None:
        """Validates input data and sets up the mode grid"""

        self._validate_input(mode_list)

        self.is_reciprocal = self._get_is_reciprocal(mode_list)
        self.modes = self._get_mode_dictionary(mode_list, self.is_reciprocal)

        # Matrices associated with symmetries
        self.rec_mat = self._get_rec_mat()

    # --------------------------------------------------------------------------
    # Input validation and processing
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_input(mode_list: list[Mode]) -> None:
        if mode_list is None:
            raise ValueError("Mode list not provided.")
        if len(mode_list) == 0:
            raise ValueError("Your mode_list is an empty list.")

    @staticmethod
    def _get_reciprocal_partner_index(
        mode: Mode, mode_list: list[Mode]
    ) -> int:
        """Find the index of a mode's reciprocal partner in mode_list

        Parameters
        ----------
        mode : Mode
            The mode whose partner is to be found.
        mode_list : list[Mode]
            The whole list of modes.

        Returns
        -------
        int
            Index of the partner within mode_list
        """

        vertices = mode.vertices
        for partner_index, partner in enumerate(mode_list):
            partner_vertices = partner.vertices
            inverted_partner_vertices = -partner_vertices

            # Check that points are the same shape
            # If not, move to the next one.
            if np.shape(vertices) != np.shape(inverted_partner_vertices):
                continue

            # Check equality
            if array_utils.is_equal_array(vertices, inverted_partner_vertices):
                return partner_index

        return -1

    @classmethod
    def _get_is_reciprocal(cls, mode_list: list[Mode]) -> bool:
        """Determine whether or not the list of modes satisfies reciprocity.

        Parameters
        ----------
        mode_list : list[Mode]
            The list of modes.

        Returns
        -------
        bool
            Is the grid reciprocal?
        """

        for mode_index, mode in enumerate(mode_list):
            partner_index = cls._get_reciprocal_partner_index(
                mode=mode, mode_list=mode_list
            )

            # If we find the partner, keep going. This will be the case if the
            # returned index is a positive integer
            if partner_index >= 0:
                continue

            # If we haven't found a partner, it can't be a reciprocal grid
            return False

        # If we reach here, every mode must have found a partner
        return True

    @classmethod
    def _get_reciprocal_indices(cls, mode_list: list[Mode]) -> list[int]:
        """Get a list of indices for saving the modes in the modes dictionary.

        If an index is negative, e.g. -5, then modes 5 and -5 are reciprocal
        partners.

        Parameters
        ----------
        mode_list : list[Mode]
            The list of modes.

        Returns
        -------
        list[int]
            The list of indices
        """

        # Reciprocal indices will be the indices we use later in the
        # dictionary. already_handled_modes will keep track of which
        # modes have been dealt with. In fact, this list will also tell
        # us what modes the reciprocal indices line up with
        if len(mode_list) == 0:
            return []

        reciprocal_indices: list[Any] = []
        already_handled_modes: list[Any] = []
        running_index = 1

        for mode_index, mode in enumerate(mode_list):
            # Check if we have already dealt with this mode
            if mode_index in already_handled_modes:
                continue

            # This mode is new. Find the index of its reciprocal partner
            partner_index = cls._get_reciprocal_partner_index(
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

    @classmethod
    def _get_mode_dictionary(
        cls,
        mode_list: list[Mode],
        is_reciprocal: bool,
    ) -> dict[tuple[str, str], Mode]:
        """Get a dictionary of modes with correct indices and wave_types

        Parameters
        ----------
        mode_list : list[Mode]
            List of modes
        is_reciprocal : bool
            Is the grid reciprocal?

        Returns
        -------
        dict
            The dictionary that will become the "modes" attribute.
        """

        modes = {}

        (
            mode_list_propagating,
            mode_list_evanescent,
        ) = cls._get_separated_mode_lists(mode_list)

        if is_reciprocal:
            indices_propagating = cls._get_reciprocal_indices(
                mode_list_propagating,
            )
            indices_evanescent = cls._get_reciprocal_indices(
                mode_list_evanescent,
            )

        else:
            indices_propagating = range(1, len(mode_list_propagating) + 1)
            indices_evanescent = range(1, len(mode_list_evanescent) + 1)

        for index, mode in zip(indices_propagating, mode_list_propagating):
            mode.index = index
            modes[(str(index), "propagating")] = mode
        for index, mode in zip(indices_evanescent, mode_list_evanescent):
            mode.index = index
            modes[(str(index), "evanescent")] = mode

        return modes

    @staticmethod
    def _get_separated_mode_lists(
        mode_list: list[Mode],
    ) -> tuple[list[Mode], list[Mode]]:
        """Separate a list of modes into lists of propagating and evanescent
        components.

        Parameters
        ----------
        mode_list : list[Mode]
            List of all modes

        Returns
        -------
        mode_list_propagating, mode_list_evanescent
            Lists containing only propagating and evanescent modes.
        """

        mode_list_propagating = [
            mode for mode in mode_list if mode.wave_type == "propagating"
        ]
        mode_list_evanescent = [
            mode for mode in mode_list if mode.wave_type == "evanescent"
        ]
        return mode_list_propagating, mode_list_evanescent

    def _get_rec_mat(self) -> Numeric:
        """The scattering matrix satisfies

        S = rec_mat @ S^T @ rec_mat

        """

        I = np.eye(2)
        J = np.eye(self.num_propagating)
        J = np.flip(J, axis=0)
        sig = np.array([[1.0, 0.0], [0.0, -1.0]])
        rec_mat = np.kron(I, np.kron(J, sig))
        return rec_mat

    # -------------------------------------------------------------------------
    # Property methods
    # -------------------------------------------------------------------------

    @property
    def num_propagating(self) -> int:
        propagating_modes = [
            mode
            for mode in self.modes.values()
            if mode.wave_type == "propagating"
        ]
        return len(propagating_modes)

    @property
    def num_evanescent(self) -> int:
        evanescent_modes = [
            mode
            for mode in self.modes.values()
            if mode.wave_type == "evanescent"
        ]
        return len(evanescent_modes)

    @property
    def propagating_indices(self) -> list[int]:
        propagating_indices = [
            int(entry[0])
            for entry in list(self.modes.keys())
            if entry[1] == "propagating"
        ]
        propagating_indices.sort()
        return propagating_indices

    @property
    def evanescent_indices(self) -> list[int]:
        evanescent_indices = [
            int(entry[0])
            for entry in list(self.modes.keys())
            if entry[1] == "evanescent"
        ]
        evanescent_indices.sort()
        return evanescent_indices

    @property
    def propagating_modes_list(self) -> list[Mode]:
        return [self.by_index(index) for index in self.propagating_indices]

    @property
    def evanescent_modes_list(self) -> list[Mode]:
        return [self.by_index(index) for index in self.evanescent_indices]

    @property
    def mode_list(self) -> list[Mode]:
        return self.propagating_modes_list + self.evanescent_modes_list

    @property
    def max_index(self) -> int:
        return (
            int((self.num_propagating - 1) / 2)
            if self.is_reciprocal
            else self.num_propagating - 1
        )

    @property
    def central_index(self) -> int:
        return 0 if self.is_reciprocal else int((self.num_propagating - 1) / 2)

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    def by_index(
        self, index: str | int, mode_wave_type: str = "propagating"
    ) -> Mode | None:
        """Get a mode from "modes" dictionary

        Parameters
        ----------
        index : str, int
            The index of the desired mode.
        mode_wave_type : str
            The type of mode that is to be collected. Options are

            "propagating" (or "p")
            "evanescent" (or "e")

        Returns
        -------
        mode : Mode
            The desired mode.
        """

        # Stringify index for consistency. Means that the user can give an int
        # if they want
        if not isinstance(index, (int, str)):
            raise ValueError("index must be an integer or a string")

        index = str(index)

        accepted_propagating_strings = {"p", "propagating"}
        accepted_evanescent_strings = {"e", "evanescent"}

        if mode_wave_type in accepted_propagating_strings:
            mode = self.modes.get((index, "propagating"), None)
        elif mode_wave_type in accepted_evanescent_strings:
            mode = self.modes.get((index, "evanescent"), None)
        else:
            raise ValueError(
                "Unknown mode_wave_type. Please specify either "
                "propagating or evanescent."
            )

        return mode

    # --------------------------------------------------------------------------
    # Object representations
    # --------------------------------------------------------------------------

    def __str__(self) -> str:
        output = (
            f"Reciprocal: {self.is_reciprocal},\n"
            f"Number of propagating modes: {self.num_propagating},\n"
            f"Number of evanescent modes: {self.num_evanescent},\n"
            f"Maximum |kappa|: {self.r_lim}\n"
        )
        return output

    def plot(
        self,
        show_indices: bool = False,
        savefig: None | str = None,
    ) ->:
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
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        # x and y axes
        # plotting_utils.draw_ray(
        #     ax, r_min=-1, theta=0, linestyle="-", color="black", alpha=0.5
        # )
        # plotting_utils.draw_ray(
        #     ax,
        #     r_min=-1,
        #     theta=np.pi / 2,
        #     linestyle="-",
        #     color="black",
        #     alpha=0.5,
        # )

        for mode in self.modes.values():
            mode.plot(
                ax=ax,
                is_solo=False,
                show_index=show_indices,
            )

        plotting_utils.draw_circle(ax)
        plotting_utils.draw_circle(ax, r=self.r_lim)
        if savefig is not None:
            fig.savefig(savefig, format="svg")
        plt.closefig()