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
        mode_boundary: npt.NDArray[Numeric] | None = None,
        inner_circle_crossings: npt.NDArray[Numeric] | None = None,
        outer_circle_crossings: npt.NDArray[Numeric] | None = None,
        r_lim: float = 1.5,
        is_polar: bool = False,
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
        mode_boundary = self._validate_boundary_data(
            mode_boundary=mode_boundary, is_polar=is_polar
        )

        self.index = index
        self.boundary = mode_boundary
        self.inner_circle_crossings = inner_circle_crossings
        self.outer_circle_crossings = outer_circle_crossings
        self.is_polar = is_polar
        self.r_lim = r_lim

        # Check if mode is a centro-symmetric or not
        inverted_boundary = -mode_boundary
        self.is_central = array_utils.is_equal_array(
            mode_boundary, inverted_boundary
        )

        self.wave_type = self._get_mode_wave_type(mode_boundary)

        # Handle special cases separately
        if is_polar:
            polar_points = geometry_utils.cartesian_to_polar(mode_boundary)
            r_vals = polar_points[:, 0]
            t_vals = polar_points[:, 1]

            # Discard t=0 for origin point in wedge case
            if len(mode_boundary) == 3:
                t_vals = t_vals[~np.isclose(r_vals, 0.0)]

            r_min, r_max = np.min(r_vals), np.max(r_vals)
            t_min, t_max = np.min(t_vals), np.max(t_vals)
            self.t_min = t_min
            self.t_max = t_max
            self.r_min = r_min
            self.r_max = r_max

            dt = geometry_utils.get_small_angular_difference(t_min, t_max)
            self.dt = dt

        # Non-polar case
        else:
            triangulation = scipy.spatial.Delaunay(self.boundary)
            self.triangulation = self.boundary[triangulation.simplices]

        # Weight
        self.weight = self._get_weight(is_polar=is_polar)

    # --------------------------------------------------------------------------
    # Input validation and processing
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_boundary_data(
        mode_boundary: npt.NDArray[Numeric] | None, is_polar: bool
    ) -> npt.NDArray[Numeric]:
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

        num_points = len(mode_boundary)
        # Check that the number of points in mode_boundary is correct
        # Polar case
        if is_polar:
            points_polar = geometry_utils.cartesian_to_polar(mode_boundary)
            r_vals = points_polar[:, 0]
            t_vals = points_polar[:, 1]
            r_min, r_max = np.min(r_vals), np.max(r_vals)
            t_min, t_max = np.min(t_vals), np.max(t_vals)

            # Wedge case
            if num_points == 3:
                is_correctly_defined_mode = np.isclose(
                    r_min, 0.0
                ) and not np.isclose(r_max, 0.0)

                if not is_correctly_defined_mode:
                    raise ValueError(
                        f"Polar mode with three points must have\n"
                        f"r_min = 0.0 and r_max > 0.0.\n"
                        f"You have\n"
                        f"r_min = {r_min}\n"
                        f"r_max = {r_max}\n"
                    )

            elif num_points == 4:
                # General polar case
                if not np.isclose(r_min, r_max):
                    # Check that t vals are distinct
                    is_correctly_defined_mode = not np.isclose(t_min, t_max)
                    if not is_correctly_defined_mode:
                        raise ValueError(
                            f"Non-central polar mode with four points must\n"
                            f"have distinct t_min and t_max.\n"
                            f"You have\n"
                            f"t_min = {t_min}\n"
                            f"t_max = {t_max}\n"
                        )

            # Mode is supposedly polar but not made from 3 or 4 points
            else:
                raise ValueError(
                    f"A polar region must be constructed from 3\n"
                    f"(central) or 4 points. You have {num_points}"
                    f"points.\n"
                )

        # Genaral non-polar case
        if not is_polar and len(mode_boundary) < 3:
            raise ValueError(
                f"A non-polar region requires at least 3 points\n"
                f"to be defined. You have {len(mode_boundary)} points.\n"
            )

        return mode_boundary

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

    def _get_weight(self, is_polar: bool) -> float:
        if is_polar:
            num_points = len(self.boundary)
            dt = self.dt
            r_min = self.r_min
            r_max = self.r_max

            if num_points == 3:
                # Wedge case
                weight = 0.5 * dt * r_max**2

            elif num_points == 4:
                # Central circle case
                if np.isclose(r_min, r_max):
                    weight = np.pi * r_max**2
                # Truncated sector (general) case
                else:
                    weight = 0.5 * dt * (r_max**2 - r_min**2)

        else:
            base_polygon_area = geometry_utils.get_convex_polygon_area(
                self.boundary
            )

            # Areas where polygons meet the circle
            inner_edge_area = 0.0
            outer_edge_area = 0.0

            if self.inner_circle_crossings is not None:
                inner_edge_area = geometry_utils.get_edge_area(
                    points=self.inner_circle_crossings, radius=1.0
                )
            if self.outer_circle_crossings is not None:
                outer_edge_area = geometry_utils.get_edge_area(
                    points=self.outer_circle_crossings, radius=self.r_lim
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
            f"Inner circle crossings: \n"
            f"{self.inner_circle_crossings},\n"
            f"Outer circle crossings: \n"
            f"{self.outer_circle_crossings},\n"
            f"Weight: {self.weight}"
            f"Polar: {self.is_polar},\n"
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

        if self.is_polar:
            self._plot_polar(
                ax=ax,
                boundary_color=boundary_color,
                index_color=index_color,
                show_index=show_index,
            )
        else:
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
            inner_circle_crossings=self.inner_circle_crossings,
            outer_circle_crossings=self.outer_circle_crossings,
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

        # Polar case
        elif self.is_polar:
            r_min, r_max = self.r_min, self.r_max

            # Firt get the mean t value. This requires handling funny
            # cases where the angle wraps around 2 PI
            t_min, t_max = self.t_min, self.t_max
            dt = self.dt
            if np.isclose(t_max, t_min + dt):
                t_mean = 0.5 * (t_min + t_max)
            else:
                t_mean = t_max + dt / 2
            # There are 2 cases for r, either the mode is a wedge near the
            # origin, or it's a truncated sector.

            # Wedge case
            if len(boundary) == 3:
                r_mean = 2 / 3 * r_max
            elif len(boundary) == 4:
                r_mean = 0.5 * (r_min + r_max)

            center_polar = np.array([r_mean, t_mean])
            center = geometry_utils.polar_to_cartesian(center_polar)
        else:
            center = np.mean(self.boundary, axis=0)  # type:ignore
        return center
