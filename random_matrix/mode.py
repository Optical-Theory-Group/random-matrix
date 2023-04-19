"""
Classes for handling modes contained in the discrete angular spectrum

Author: Niall Francis Byrnes
https://niallbyrnes.com

"""

import numpy as np
from copy import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from utils.plotting_utils import draw_circle, draw_ray, set_up_k_space_plot, draw_line, draw_convex_polygon, draw_horizontal_chord, draw_vertical_chord
from utils.geometry_utils import circle, cartesian_to_polar, polar_to_cartesian, is_rectangle, get_convex_hull_area
from utils.array_utils import remove_duplicate_points

class Mode:    
    def __init__(self, index=0, mode_boundary=None, mode_shape_type=None):
        # Check that index is an integer
        if not isinstance(index, int):
            raise ValueError("index must be supplied as an integer")
        self.index = index

        # Check that input string arguments are from allowed values
        valid_shape_types = ["cartesian", "polar", "custom"]
        if mode_shape_type not in valid_shape_types:
            raise ValueError(f"{mode_shape_type} is an invalid shape type. Valid shape types are: {valid_shape_types}")
        self.mode_shape_type = mode_shape_type
    
        # Check that either a convex_hull or numpy array of points has been provided
        # Ensure that dimensions are correct
        if isinstance(mode_boundary, ConvexHull):
            points = mode_boundary.points
        elif isinstance(mode_boundary, np.ndarray):
            if mode_boundary.ndim != 2 or mode_boundary.shape[1] != 2:
                raise ValueError("mode_boundary must be a 2D array of (x,y) points.")
            points = mode_boundary
        else:
            raise ValueError("mode_boundary must be supplied as either a ConvexHull object or a numpy array of points.")

        # Check that there are at least 3 points (2 for polar modes)
        if len(points) < 3 and mode_shape_type != "polar":
            raise ValueError("A non-polar region requires at least 3 points to be defined.")
        if len(points) < 2 and mode_shape_type == "polar":
            raise ValueError("A polar region requires at least 2 points to be defined.")

        # Remove duplicate points
        duplicates_removed = remove_duplicate_points(points)
        self.points = duplicates_removed

        # Check that points are either in the interior or exterior of the circle
        r_values = cartesian_to_polar(points)[:,0]
        all_interior_points = np.all(r_values <= 1.0)
        all_exterior_points = np.all(r_values >= 1.0)
        if all_interior_points:
            self.mode_wave_type = "propagating"
        elif all_exterior_points:
            self.mode_wave_type = "evanescent"
        else:
            raise ValueError("Excluding boundary points, all points must be either inside or outside the circle.")

        # Handle special cases separately
        match mode_shape_type:
            case "polar":
                self._handle_polar_case()
            case "cartesian":
                self._handle_cartesian_case()
            case _:
                self._handle_custom_case()

    def __str__(self):
        output = f"Mode shape type: {self.mode_shape_type},\nWave Type: {self.mode_wave_type},\nPoints: {self.points},\nWeight: {self.weight}"
        return output

    def _handle_polar_case(self):
        points_cartesian = self.points
        points_polar = cartesian_to_polar(points_cartesian)
        n_points = len(points_polar)

        r_min, r_max = np.min(points_polar[:,0]), np.max(points_polar[:,0])
        t_min, t_max = np.min(points_polar[:,1]), np.max(points_polar[:,1])
        self.r_min = r_min
        self.r_max = r_max

        match n_points:
            case 2:
                # If only two points are given, it must be the central mode        
                is_correctly_defined_central_mode = np.isclose(r_min, 0.0) and not np.isclose(r_min, r_max)
                if not is_correctly_defined_central_mode:
                    raise ValueError("Central mode incorrectly specified. Please include (0,0) and one other point not located at the origin.")
                small_circle_radius = r_max
                self.weight = np.pi*small_circle_radius**2
                self.is_central_mode = True

            case 4:
                # If four points are given, the mode should be a non-central mode
                # Check that points properly align
                is_correctly_defined_central_mode = is_rectangle(points_polar)
                if not is_correctly_defined_central_mode:
                    raise ValueError("Non-central mode incorrectly specified. Please include 4 points with 2 unique values of r and theta.")
                
                # Ensure smaller angle is taken
                sector_angle = t_max - t_min
                if sector_angle > np.pi:
                    sector_angle = 2*np.pi - sector_angle

                self.weight = 0.5*sector_angle*(r_max**2 - r_min**2)
                self.is_central_mode = False
                self.t_min = t_min
                self.t_max = t_max

            case _:
                # Mode has been incorrectly specified
                raise ValueError("Polar mode must be constructed from either 2 or 4 points.")
            
    def _handle_cartesian_case(self):
        points_cartesian = self.points

        # Check that points form a rectangle
        is_correctly_defined_cartesian_mode = is_rectangle(points_cartesian)
        if not is_correctly_defined_cartesian_mode:
            raise ValueError("Points do not form a rectangle.")

        x_min, x_max = np.min(points_cartesian[:,0]), np.max(points_cartesian[:,0])
        y_min, y_max = np.min(points_cartesian[:,1]), np.max(points_cartesian[:,1])
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.weight = get_convex_hull_area(points_cartesian)

    def _handle_custom_case(self):
        pass

    def integrate(self, function):
        match self.mode_shape_type:
            case "polar":
                return self._integrate_polar(function)
            case "cartesian":
                return self._integrate_cartesian(function)
            case "custom":
                return self._integrate_custom(function)

    def _integrate_polar(self, function):
        pass

    def _integrate_cartesian(self, function):
        pass

    def _integrate_custom(self, function):
        pass

    def plot(self, ax=None, is_solo=True, show_guidelines=True, mode_color="tab:red"):
        if is_solo:
            ax = set_up_k_space_plot()

        match self.mode_shape_type:
            case "polar":
                self._plot_polar(ax, show_guidelines, mode_color)
            case "cartesian":
                self._plot_cartesian(ax, show_guidelines, mode_color)
            case "custom":
                self._plot_custom()

    def _plot_polar(self, ax, show_guidelines, mode_color):
        if self.is_central_mode:
            # mode is a small circle centred at the origin
            small_circle_radius = self.r_max
            draw_circle(ax, r=small_circle_radius, color=mode_color)
        else:
            if show_guidelines:
                draw_ray(ax, r_min=-1, theta=self.t_min, linestyle="--", color="tab:blue")
                draw_ray(ax, r_min=-1, theta=self.t_max, linestyle="--", color="tab:blue")
                draw_circle(ax, r=self.r_min, linestyle="--", color="tab:blue")
                draw_circle(ax, r=self.r_max, linestyle="--", color="tab:blue")

            # Ensure acute angle sector is taken
            t1 = self.t_min
            t2 = self.t_max
            if self.t_max - self.t_min > np.pi:
                t1 = self.t_max - 2*np.pi
                t2 = self.t_min
            draw_ray(ax, theta=self.t_min, r_min=self.r_min, r_max=self.r_max, linestyle="-", color=mode_color)
            draw_ray(ax, theta=self.t_max, r_min=self.r_min, r_max=self.r_max, linestyle="-", color=mode_color)
            draw_circle(ax, t_min=t1, t_max=t2, r=self.r_min, linestyle="-", color=mode_color)
            draw_circle(ax, t_min=t1, t_max=t2, r=self.r_max, linestyle="-", color=mode_color)

    def _plot_cartesian(self, ax, show_guidelines, mode_color):
        if show_guidelines:
            # Vertical lines
            draw_vertical_chord(ax, self.x_min, color="tab:blue", linestyle="--")
            draw_vertical_chord(ax, self.x_max, color="tab:blue", linestyle="--")
            draw_horizontal_chord(ax, self.y_min, color="tab:blue", linestyle="--")
            draw_horizontal_chord(ax, self.y_max, color="tab:blue", linestyle="--")

        draw_convex_polygon(ax, self.points, color="red")