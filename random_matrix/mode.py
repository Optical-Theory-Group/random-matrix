"""This module defines a "Mode" class for use in scattering calculations.

In this context, a mode is defined as a non-zero, finite region of (k_x, k_y)
space. A mode thus represents a bundle of wavevectors that light can scatter
from or into.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.spatial

from random_matrix.utils import array_utils, geometry_utils, plotting_utils
from random_matrix.utils.typevars import Numeric


class Mode:
    """
    A class used to represent a mode

    Attributes
    ----------
    index : int
        The index of the mode within its container ModeGrid.
    is_polar : bool
        A boolean that is True if the mode is polar. Polar modes have curved
        edges, while others are polygonal in shape.
    points : np.ndarray
        A numpy array of size (N,2) where N is the number of points defining
        the boundary of the mode.
    circle_points : np.ndarray | None
        An array of size (2,2) containing a pair of points within the boundary
        points that also lie on the unit circle. If no such points exist,
        this will be None.
    mode_wave_type: str
        A string describing if the mode is "propagating" or "evanescent"

    Methods
    -------
    """

    circle_rtol = 1e-8
    generic_rtol = 1e-8

    # --------------------------------------------------------------------------
    # Constructor methods
    # --------------------------------------------------------------------------

    def __init__(
        self,
        index: int = 0,
        mode_boundary_dict: dict[
            str, npt.NDArray[Numeric] | list[npt.NDArray[Numeric]]
        ]
        | None = None,
    ) -> None:
        """
        Initialises mode.

        Parameters
        ----------
        index : int
            The mode's index within its containing ModeGrid.
        mode_boudnary : np.ndarray, ConvexHull
            The boundary of the mode, defined by an array of 2D points.
            A ConvexHull object will also be accepted.
        is_polar : bool
            A boolean that tells the initialiser whether or not the mode is
            polar. If it is, it has curved edges and is treated differently.
        is_central_mode : bool
            True if the mode is centrosymmetric about the origin.

        Returns
        -------
        None
        """

        # Validate and clean up mode_boundary
        mode_boundary_dict = self._validate_boundary_dict(
            mode_boundary_dict=mode_boundary_dict
        )

        mode_boundary = mode_boundary_dict["mode_boundary"]
        arc_points_list = mode_boundary_dict["arc_points_list"]

        self.index = index
        self.boundary = mode_boundary
        self.arc_points_list = arc_points_list

        # Check if mode is a centro-symmetric or not
        inverted_boundary = -mode_boundary  # type: ignore
        self.is_central = array_utils.is_equal_array(
            mode_boundary, inverted_boundary  # type: ignore
        )

        self.wave_type = self._get_mode_wave_type(mode_boundary)  # type: ignore

        # triangulation = scipy.spatial.Delaunay(self.boundary)
        # self.triangulation = self.boundary[triangulation.simplices]

        # Weight
        self.weight = self._get_weight()

    # --------------------------------------------------------------------------
    # Input validation and processing
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_boundary_dict(
        mode_boundary_dict: dict[
            str, npt.NDArray[Numeric] | list[npt.NDArray[Numeric]]
        ]
        | None,
    ) -> dict[str, npt.NDArray[Numeric] | list[npt.NDArray[Numeric]]]:
        if mode_boundary_dict is None:
            raise ValueError("No data provided to construct mode.")

        mode_boundary = mode_boundary_dict["mode_boundary"]
        arc_points_list = mode_boundary_dict["arc_points_list"]

        # Check that either a numpy array of points has been
        # provided and ensure that dimensions are correct
        if isinstance(mode_boundary, np.ndarray):
            if mode_boundary.ndim != 2 or mode_boundary.shape[1] != 2:
                raise ValueError(
                    f"mode_boundary must be a 2D array of (x,y) points.\n"
                    f"Your array has shape {np.shape(mode_boundary)}.\n"
                )
        else:
            raise ValueError(
                f"mode_boundary must be a numpy array.\n"
                f"Yours is a {type(mode_boundary)}.\n"
            )

        # Remove duplicate points and order them cyclically
        mode_boundary = array_utils.remove_duplicate_points(mode_boundary)
        mode_boundary = geometry_utils.order_points(mode_boundary)
        mode_boundary_dict["mode_boundary"] = mode_boundary

        num_points = len(mode_boundary)
        num_arc_points = len(arc_points_list)
        # Check that the number of points in mode_boundary is correct

        if num_points == 2 and num_arc_points == 0:
            raise ValueError(
                "A mode containing only two points must be "
                "have arc points too. You have 0."
            )
        if num_points < 2:
            raise ValueError(
                "At least two boundary points are required to "
                "define a mode."
            )

        return mode_boundary_dict

    @staticmethod
    def _get_mode_wave_type(mode_boundary: npt.NDArray[Numeric]) -> str:
        r_vals = np.linalg.norm(mode_boundary, axis=1)
        circle_points_excluded = r_vals[~np.isclose(r_vals, 1.0)]
        all_interior_points = np.all(circle_points_excluded < 1.0)
        all_exterior_points = np.all(circle_points_excluded > 1.0)

        if all_interior_points:
            return "propagating"
        elif all_exterior_points:
            return "evanescent"
        else:
            raise ValueError(
                "Your mode has points both in the propagating and\n"
                "evanescent regions of k-space. This is not allowed.\n"
            )

    def _get_weight(self) -> float:
        boundary_points = self.boundary
        num_points = len(boundary_points)
        arc_points_list = self.arc_points_list
        num_arcs = len(arc_points_list)

        base_polygon_area = 0.0

        if num_points > 2:
            base_polygon_area += geometry_utils.get_convex_polygon_area(
                boundary_points
            )

        # Check if there are any arc points. If not, we're done
        if num_arcs == 0:
            return base_polygon_area

        # Sort inner and outer crossings
        inner_crossing_list = []
        outer_crossing_list = []
        for arc_points in arc_points_list:
            r_val = np.linalg.norm(arc_points[0])
            if np.isclose(r_val, 1.0):
                inner_crossing_list.append(arc_points)
            else:
                outer_crossing_list.append(arc_points)

        # Areas where polygons meet the circle
        inner_edge_area = 0.0
        outer_edge_area = 0.0

        for inner_crossing in inner_crossing_list:
            inner_edge_area += geometry_utils.get_edge_area(
                points=inner_crossing, radius=1.0
            )
        for outer_crossing in outer_crossing_list:
            outer_edge_area += geometry_utils.get_edge_area(
                points=outer_crossing,
                radius=np.linalg.norm(outer_crossing[0]),  # type: ignore
            )

        extra_area = 0.0

        if self.wave_type == "propagating":
            extra_area += inner_edge_area
        elif self.wave_type == "evanescent":
            extra_area += outer_edge_area
            extra_area -= inner_edge_area

        weight = base_polygon_area + extra_area

        return weight  # type: ignore

    # --------------------------------------------------------------------------
    # Object representations
    # --------------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Gives a helpful summary of the mode.

        Parameters
        ----------
        None

        Returns
        -------
        output : str
            A string summarising the mode.
        """

        output = (
            f"Boundary vertices: \n"
            f"{self.boundary},\n"
            f"Arc points: \n"
            f"{self.arc_points_list}\n"
            # f"Weight: {self.weight}\n"
            f"Wave Type: {self.wave_type},\n"
        )
        return output

    def plot(
        self,
        ax: plt.Axes,
        boundary_color: str,
        triangulation_color: str,
        index_color: str,
        show_index: bool = False,
        show_triangulation: bool = False,
    ) -> None:
        """
        Draws the mode.

        Parameters
        ----------
        ax : plt.Axes
            The axis on which the mode will be drawn.
        is_solo : bool
            Creates a new axis object if True. Allows for plotting of indivudal
            modes. Keep false if plotting from within a ModeGrid.
        show_guidelines : bool
            If True, shows rays and circles that illustrate the positioning of
            a polar mode. Does nothing for general modes.
        mode_color : str
            The color of the mode.
        show_index : bool
            Shows the mode's index at its center if True.
        show_triangulation : bool
            Also draws the mode's triangulation if True

        Returns
        -------
        None
        """

        self._plot_general(
            ax=ax,
            boundary_color=boundary_color,
            triangulation_color=triangulation_color,
            index_color=index_color,
            show_index=show_index,
            show_triangulation=show_triangulation,
        )

    def _plot_polar(
        self,
        ax: plt.Axes,
        boundary_color: str,
        index_color: str,
        show_index: bool = False,
    ) -> None:
        """
        Extension of plot for polar modes.

        Parameters
        ----------
        ax : plt.Axes
            The axis on which the mode will be drawn.
        show_guidelines : bool
            If True, shows rays and circles that illustrate the positioning of
            a polar mode. Does nothing for general modes.
        mode_color : str
            The color of the mode.
        show_index : bool
            Shows the mode's index at its center if True.

        Returns
        -------
        None
        """

        if self.is_central:
            # mode is a small circle centred at the origin
            small_circle_radius = self.r_max
            plotting_utils.draw_circle(
                ax, r=small_circle_radius, color=boundary_color
            )
        else:
            # Ensure acute angle sector is taken
            t_1 = self.t_min
            t_2 = self.t_max
            if self.t_max - self.t_min > np.pi:
                t_1 = self.t_max - 2 * np.pi
                t_2 = self.t_min

            # Two side rays
            plotting_utils.draw_ray(
                ax,
                theta=self.t_min,
                r_min=self.r_min,
                r_max=self.r_max,
                linestyle="-",
                color=boundary_color,
            )
            plotting_utils.draw_ray(
                ax,
                theta=self.t_max,
                r_min=self.r_min,
                r_max=self.r_max,
                linestyle="-",
                color=boundary_color,
            )
            # Circular parts
            plotting_utils.draw_circle(
                ax,
                t_min=t_1,
                t_max=t_2,
                r=self.r_min,
                linestyle="-",
                color=boundary_color,
            )
            plotting_utils.draw_circle(
                ax,
                t_min=t_1,
                t_max=t_2,
                r=self.r_max,
                linestyle="-",
                color=boundary_color,
            )

        if show_index:
            central_coordinates = self.get_center()  # type: ignore
            x, y = central_coordinates
            plt.text(
                x,
                y,
                str(self.index),
                ha="center",
                va="center",
                color=index_color,
            )

    def _plot_general(
        self,
        ax: plt.Axes,
        boundary_color: str,
        index_color: str,
        triangulation_color: str,
        show_index: bool = False,
        show_triangulation: bool = False,
    ) -> None:
        """
        Extension of plot for non-polar modes.

        Parameters
        ----------
        ax : plt.Axes
            The axis on which the mode will be drawn.
        show_guidelines : bool
            If True, shows rays and circles that illustrate the positioning of
            a polar mode. Does nothing for general modes.
        mode_color : str
            The color of the mode.
        show_index : bool
            Shows the mode's index at its center if True.

        Returns
        -------
        None
        """
        if self.wave_type == "propagating":
            boundary_color = "red"
        else:
            boundary_color = "blue"

        if show_triangulation:
            for triangle in self.triangulation:
                plotting_utils.draw_interior_triangle(
                    ax,
                    triangle,
                    color=triangulation_color,
                    polygon_points=self.boundary,
                )

        plotting_utils.draw_convex_polygon(
            ax,
            self.boundary,
            color=boundary_color,
            arc_points_list=self.arc_points_list,
        )

        if show_index:
            central_coordinates = self.get_center()  # type: ignore
            x, y = central_coordinates
            plt.text(
                x,
                y,
                str(self.index),
                ha="center",
                va="center",
                color=boundary_color,
            )

    # -------------------------------------------------------------------------
    # Miscellaneous
    # -------------------------------------------------------------------------

    def get_center(self) -> npt.NDArray[Numeric]:
        """
        Find the central coordinates of the mode. Used in plotting.

        Parameters
        ----------
        None

        Returns
        -------
        central_coordinates : np.ndarray
            An array of length 2 containing the coordinates of the centre of
            the mode.
        """

        boundary = self.boundary

        # Centro-symmetric mode. The center is just the origin.
        if self.is_central:
            center = np.array([0.0, 0.0])
        else:
            center = np.mean(self.boundary, axis=0)  # type:ignore
        return center
