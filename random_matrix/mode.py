"""This module defines a "Mode" class for use in scattering calculations.

In this context, a mode is defined as a non-zero, finite region of (k_x, k_y)
space. A mode thus represents a bundle of wavevectors that light can scatter
from or into.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from random_matrix.types.array_types import Matrix, Vector
from random_matrix.utils.array_utils import remove_duplicate_points
from random_matrix.utils.geometry_utils import (
    cartesian_to_polar,
    get_edge_area,
    get_convex_polygon_area,
    get_small_angular_difference,
    is_rectangle,
    order_points,
    polar_to_cartesian,
)
from random_matrix.utils.plotting_utils import (
    draw_circle,
    draw_convex_polygon,
    draw_interior_triangle,
    draw_ray,
    set_up_k_space_plot,
)


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

    def __init__(
        self,
        index: int = 0,
        mode_boundary: Matrix[np.float32] | ConvexHull = None,
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

        Returns
        -------
        None
        """

        # Check that index is an integer
        if not isinstance(index, int):
            raise ValueError("index must be given as an integer")
        self.index = index

        # Check that either a convex_hull or numpy array of points has been
        # provided and ensure that dimensions are correct
        if isinstance(mode_boundary, ConvexHull):
            points = mode_boundary.points
        elif isinstance(mode_boundary, np.ndarray):
            if mode_boundary.ndim != 2 or mode_boundary.shape[1] != 2:
                raise ValueError(
                    "mode_boundary must be a 2D array of (x,y) points."
                )
            points = mode_boundary
        else:
            raise ValueError(
                "mode_boundary must be given as either a "
                "ConvexHull object or a numpy array of points."
            )

        # Remove duplicate points
        points = remove_duplicate_points(points)

        # Check that the number of points is correct
        self.is_polar = is_polar
        if is_polar and len(points) not in [2, 4]:
            raise ValueError(
                "A polar region must be constructed from 2 "
                "(central) or 4 points."
            )
        if not is_polar and len(points) < 3:
            raise ValueError(
                "A non-polar region requires at least 3 points "
                "to be defined."
            )

        # Order the points unless special polar case
        # (makes future calculations simpler)
        if len(points) >= 3:
            points = order_points(points)
        self.points = points

        # Check points lying on the circle
        r_vals = cartesian_to_polar(points)[:, 0]
        circle_points = points[np.isclose(r_vals, 1.0)]
        num_circle_points = len(circle_points)
        if num_circle_points > 2:
            raise ValueError(
                f"A mode must not have more than two points lying on the "
                f"circle. You have {num_circle_points}."
            )
        self.circle_points = circle_points if num_circle_points == 2 else None

        # Check that a mode does not have points both inside and outside of
        # the circle
        circle_points_excluded = r_vals[~np.isclose(r_vals, 1.0)]
        all_interior_points = np.all(circle_points_excluded <= 1.0)
        all_exterior_points = np.all(circle_points_excluded >= 1.0)
        if all_interior_points:
            self.mode_wave_type = "propagating"
        elif all_exterior_points:
            self.mode_wave_type = "evanescent"
        else:
            raise ValueError(
                "Excluding boundary points, all points must be "
                "either inside or outside the circle."
            )

        # Handle special cases separately
        if is_polar:
            self._handle_polar_case()
        else:
            self._handle_general_case()

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
            f"Polar: {self.is_polar},\n"
            f"Wave Type: {self.mode_wave_type},\n"
            f"Points: {self.points},\n"
            f"Weight: {self.weight}"
        )
        return output

    def _handle_polar_case(self) -> None:
        """
        Sets up a polar mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        points_polar = cartesian_to_polar(self.points)
        self.points_polar = points_polar
        num_points = len(points_polar)

        r_min, r_max = np.min(points_polar[:, 0]), np.max(points_polar[:, 0])
        t_min, t_max = np.min(points_polar[:, 1]), np.max(points_polar[:, 1])
        self.r_min = r_min
        self.r_max = r_max

        if num_points == 2:
            # If only two points are given, it must be the central mode
            is_correctly_defined_central_mode = np.isclose(
                r_min, 0.0
            ) and not np.isclose(r_min, r_max)
            if not is_correctly_defined_central_mode:
                raise ValueError(
                    "Central mode incorrectly specified. Please "
                    "include (0,0) and one other point not "
                    "located at the origin."
                )
            small_circle_radius = r_max
            self.weight = np.pi * small_circle_radius**2
            self.is_central_mode = True

        else:
            # If four points are given, the mode should be a non-central mode
            # Check that points properly align
            is_correctly_defined_central_mode = is_rectangle(points_polar)
            if not is_correctly_defined_central_mode:
                raise ValueError(
                    "Non-central mode incorrectly specified. "
                    "Please include 4 points with 2 unique "
                    "values of r and theta."
                )

            # Ensure smaller angle is taken
            sector_angle = get_small_angular_difference(t_min, t_max)

            self.weight = 0.5 * sector_angle * (r_max**2 - r_min**2)
            self.is_central_mode = False
            self.t_min = t_min
            self.t_max = t_max

    def _handle_general_case(self) -> None:
        """
        Sets up a non-polar mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Check if is edge case or not
        self.is_edge = self.circle_points is not None

        # Calculate weight based on whether mode is an edge mode or not
        weight = get_convex_polygon_area(self.points)
        if self.is_edge:
            weight += get_edge_area(self.circle_points)
        self.weight = weight

        # Work out triangulation used for integration
        triangulation = Delaunay(self.points)
        self.triangulation = self.points[triangulation.simplices]

    def plot(
        self,
        ax: plt.Axes = None,
        is_solo: bool = True,
        show_guidelines: bool = True,
        mode_color: str = "tab:red",
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

        if is_solo:
            ax = set_up_k_space_plot()

        if self.is_polar:
            self._plot_polar(ax, show_guidelines, mode_color, show_index)
        else:
            self._plot_general(ax, mode_color, show_index, show_triangulation)

    def _plot_polar(
        self,
        ax: plt.Axes,
        show_guidelines: bool,
        mode_color: str,
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

        if self.is_central_mode:
            # mode is a small circle centred at the origin
            small_circle_radius = self.r_max
            draw_circle(ax, r=small_circle_radius, color=mode_color)
        else:
            if show_guidelines:
                draw_ray(
                    ax,
                    r_min=-1,
                    theta=self.t_min,
                    linestyle="--",
                    color="tab:blue",
                )
                draw_ray(
                    ax,
                    r_min=-1,
                    theta=self.t_max,
                    linestyle="--",
                    color="tab:blue",
                )
                draw_circle(ax, r=self.r_min, linestyle="--", color="tab:blue")
                draw_circle(ax, r=self.r_max, linestyle="--", color="tab:blue")

            # Ensure acute angle sector is taken
            t_1 = self.t_min
            t_2 = self.t_max
            if self.t_max - self.t_min > np.pi:
                t_1 = self.t_max - 2 * np.pi
                t_2 = self.t_min
            draw_ray(
                ax,
                theta=self.t_min,
                r_min=self.r_min,
                r_max=self.r_max,
                linestyle="-",
                color=mode_color,
            )
            draw_ray(
                ax,
                theta=self.t_max,
                r_min=self.r_min,
                r_max=self.r_max,
                linestyle="-",
                color=mode_color,
            )
            draw_circle(
                ax,
                t_min=t_1,
                t_max=t_2,
                r=self.r_min,
                linestyle="-",
                color=mode_color,
            )
            draw_circle(
                ax,
                t_min=t_1,
                t_max=t_2,
                r=self.r_max,
                linestyle="-",
                color=mode_color,
            )

        if show_index:
            central_coordinates = self.get_mode_center()
            x, y = central_coordinates
            plt.text(x, y, str(self.index), ha="center", va="center")

    def get_mode_center(self) -> Vector[np.float32]:
        """
        Find the central coordinates of the mode.

        Parameters
        ----------
        None

        Returns
        -------
        central_coordinates : np.ndarray
            An array of length 2 containing the coordinates of the centre of
            the mode.
        """

        if self.is_polar and self.is_central_mode:
            central_coordinates = self.points[0]
        elif self.is_polar and not self.is_central_mode:
            r_mean = np.mean(self.points_polar[:, 0])

            # Handle funny t cases where the angle wraps around 2PI
            t_min, t_max = self.t_min, self.t_max
            dt = get_small_angular_difference(t_min, t_max)
            if np.isclose(t_max, t_min + dt):
                t_mean = 0.5 * (t_min + t_max)
            else:
                t_mean = t_max + dt / 2

            central_coordinates_polar = np.array([r_mean, t_mean])
            central_coordinates = polar_to_cartesian(
                central_coordinates_polar
            )[0]
        else:
            central_coordinates = np.mean(self.points, axis=0)
        return central_coordinates  # type: ignore

    def _plot_general(
        self,
        ax: plt.Axes,
        mode_color: str,
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
                draw_interior_triangle(
                    ax, triangle, color="tab:blue", polygon_points=self.points
                )
        draw_convex_polygon(ax, self.points, color="red")
        if show_index:
            index_color = (
                "black" if self.mode_wave_type == "propagating" else "blue"
            )
            central_coordinates = self.get_mode_center()
            x, y = central_coordinates
            plt.text(
                x,
                y,
                str(self.index),
                ha="center",
                va="center",
                color=index_color,
            )
