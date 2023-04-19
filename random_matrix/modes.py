"""
Classes for handling modes contained in the discrete angular spectrum

Author: Niall Francis Byrnes
https://niallbyrnes.com

"""

import numpy as np
from copy import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from utils import draw_circle, draw_ray, circle, cartesian_to_polar, polar_to_cartesian, remove_duplicate_points, draw_k_space

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
        self.points_polar = points_polar

        all_r_values = points_polar[:,0]
        all_t_values = points_polar[:,1]
        unique_r_values = remove_duplicate_points(all_r_values)
        unique_t_values = remove_duplicate_points(all_t_values)

        r_min = np.min(unique_r_values)
        r_max = np.max(unique_r_values)
        self.r_min = r_min
        self.r_max = r_max
        
        t_min = np.min(unique_t_values)
        t_max = np.max(unique_t_values)
        n_points = len(all_r_values)

        # Deal with central mode and non-central mode cases separately
        match n_points:
            case 2:
                # For a central mode, we expect 2 points
                # One should be the origin, and the other should be any non-origin point
                is_correctly_defined_central_mode = np.isclose(r_min, 0.0) and not np.isclose(r_min, r_max)
                if not is_correctly_defined_central_mode:
                    raise ValueError("Central mode incorrectly specified. Please include (0,0) and one other point not located at the origin.")
                small_circle_radius = r_max
                self.weight = np.pi*small_circle_radius**2
                self.is_central_mode = True

            case 4:
                # For a non-central mode, we expect 4 points
                # These four points must exahust all possibilites of two different r and t values
                is_correctly_defined_non_central_mode = len(unique_r_values) == 2 and len(unique_t_values) == 2
                if not is_correctly_defined_non_central_mode:
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
        pass

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
            ax = draw_k_space()

        match self.mode_shape_type:
            case "polar":
                self._plot_polar(ax, show_guidelines, mode_color)
            case "cartesian":
                self._plot_cartesian()
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

class Modes:
    def __init__(self, mode_boundary_data: dict = None) -> None:
        self.modes = []
        
        # Parse mode_boundary_data
        t_offset = mode_boundary_data.get("t_offset", 0)

        match mode_boundary_data:
            case {"mode_shape_type": "polar", "r_vals": r_vals, "t_vals": t_vals}:
                self._handle_polar_case(r_vals=r_vals, t_vals=t_vals, t_offset=t_offset)

            case {"mode_shape_type": "cartesian", "dx": dx, "dy": dy}:
                self._handle_cartesian_case(dx=dx, dy=dy, t_offset=t_offset)

            case {"mode_shape_type": "custom", "points_array": points_array}:
                pass

            case _:
                raise ValueError("Incorrect mode_boundary_data formatting.")

    def _handle_polar_case(self, r_vals, t_vals, t_offset):
        mode_shape_type = "polar"
        
        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not np.any(np.isclose(r_vals, 0.0)):
            r_vals = np.concatenate(([0.0], r_vals))
        if not np.any(np.isclose(r_vals, 1.0)):
            r_vals = np.concatenate((r_vals, [1.0]))

        # Sort out central mode
        central_mode_radius = r_vals[1]
        points_cartesian = np.array([[0.0,0.0], [central_mode_radius, 0.0]])
        self.modes.append(Mode(index=0, mode_boundary=points_cartesian, mode_shape_type=mode_shape_type))


        
        # Non-central modes
        r_vals = r_vals[1:]
        r_val_pairs = np.column_stack((r_vals[:-1], r_vals[1:]))
        t_val_pairs = np.column_stack((t_vals[:-1], t_vals[1:]))

        for r_val_pair in r_val_pairs:
            for t_val_pair in t_val_pairs:
                r_grid, t_grid = np.meshgrid(r_val_pair, t_val_pair)
                points_polar = np.column_stack((r_grid.ravel(), t_grid.ravel()))
                points_cartesian = polar_to_cartesian(points_polar)
                self.modes.append(Mode(index=0, mode_boundary=points_cartesian, mode_shape_type=mode_shape_type))

    def plot(self):
        # Draw axes and k-space boundary
        ax = draw_k_space()
        for mode in self.modes:
            mode.plot(ax=ax, is_solo=False, show_guidelines=False, mode_color="tab:red")