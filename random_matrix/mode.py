"""
Classes for handling modes contained in the discrete angular spectrum

Author: Niall Francis Byrnes
https://niallbyrnes.com

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.spatial import ConvexHull, Delaunay
from .utils.array_types import Vector, Matrix
from .utils.plotting_utils import (
    draw_circle,
    draw_ray,
    set_up_k_space_plot,
    draw_convex_polygon,
    draw_interior_triangle,
)
from .utils.geometry_utils import (
    cartesian_to_polar,
    polar_to_cartesian,
    is_rectangle,
    get_convex_hull_area,
    points_to_ordered_convex_hull_vertices,
    get_small_angular_difference,
    get_boundary_area,
)
from .utils.array_utils import remove_duplicate_points


class Mode:
    def __init__(
        self,
        index: int = 0,
        mode_boundary: Matrix[np.float64] | ConvexHull = None,
        is_polar: bool = False,
    ) -> None:
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

        # Remove duplicate points and order them
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
            points = points_to_ordered_convex_hull_vertices(points)
        self.points = points

        # Check boundary points
        r_vals = cartesian_to_polar(points)[:, 0]
        boundary_points = points[np.isclose(r_vals, 1.0)]
        num_boundary_points = len(boundary_points)
        if num_boundary_points > 2:
            raise ValueError(
                f"A mode must not have more than two boundary "
                f"points. You have {num_boundary_points}."
            )
        self.boundary_points = (
            boundary_points if num_boundary_points > 0 else None
        )

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
        output = (
            f"Polar: {self.is_polar},\nWave Type: "
            f"{self.mode_wave_type}, "
            f"\nPoints: {self.points},\nWeight: {self.weight}"
        )
        return output

    def _handle_polar_case(self) -> None:
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
        # Check if is edge case or not
        self.is_edge = (
            self.boundary_points is not None and len(self.boundary_points) == 2
        )

        # Calculate weight based on whether mode is an edge mode or not
        weight = get_convex_hull_area(self.points)
        if self.is_edge:
            weight += get_boundary_area(self.boundary_points)
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
        if show_triangulation:
            for triangle in self.triangulation:
                draw_interior_triangle(
                    ax, triangle, color="tab:blue", polygon_points=self.points
                )
        draw_convex_polygon(ax, self.points, color="red")
