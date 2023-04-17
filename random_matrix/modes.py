"""
Classes for handling modes contained in the discrete angular spectrum

Author: Niall Francis Byrnes
https://niallbyrnes.com

"""

import numpy as np
from copy import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from utils import draw_circle, draw_ray, circle, cartesian_to_polar, remove_duplicate_points

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
        if isinstance(mode_boundary, ConvexHull):
            points = mode_boundary.points
        elif isinstance(mode_boundary, np.ndarray):
            points = mode_boundary
        else:
            ValueError("mode_boundary must be supplied as either a ConvexHull object or a numpy array of points.")

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
                is_correct_central_mode = np.isclose(r_min, 0.0) and not np.isclose(r_min, r_max)
                if not is_correct_central_mode:
                    raise ValueError("Central mode incorrectly specified. Please include (0,0) and one other point not located at the origin.")
                small_circle_radius = r_max
                self.weight = np.pi*small_circle_radius**2
                self.is_central_mode = True

            case 4:
                # For a non-central mode, we expect 4 points
                # These four points must exahust all possibilites of two different r and t values
                is_correct_non_central_mode = len(unique_r_values) == 2 and len(unique_t_values) == 2
                if not is_correct_non_central_mode:
                    raise ValueError("Non-central mode incorrectly specified. Please include 4 points with 2 unique values of r and theta.")

                # Ensure acute angle is taken
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
            # Draw axes and k-space boundary
            fig, ax = plt.subplots()
            draw_ray(ax, r_min=-1, theta=0, linestyle="-", color="black", alpha=0.4)
            draw_ray(ax, r_min=-1, theta=np.pi/2, linestyle="-", color="black", alpha=0.4)
            ax.set_aspect("equal")
            draw_circle(ax)

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
    def __init__(self, mode_shape_type="cartesian", mode_boundary_data=None):
        self.modes = []
        
        # Parse mode_boundary_data
        t_offset = mode_boundary_data["t_offset"] if mode_boundary_data >= {"t_offset"} else 0

        # Cases for polar modes
        if {"r_vals", "t_vals"} >= mode_boundary_data:
            r_vals = mode_boundary_data["r_vals"]
            t_vals = mode_boundary_data["t_vals"]
            self.handle_polar_case(input_type="vals", r_vals=r_vals, t_vals=t_vals, t_offset=t_offset)
        elif {"n_r", "n_t"} >= mode_boundary_data:
            n_r = mode_boundary_data["n_r"]
            n_t = mode_boundary_data["n_t"]
            self.handle_polar_case(input_type="nums", n_r=n_r, n_t=n_t, t_offset=t_offset)

        # Cases for cartesian modes
        elif {"dx", "dy"} >= mode_boundary_data:
            dx = mode_boundary_data["dx"]
            dy = mode_boundary_data["dy"]
            self.handle_cartesian_case(dx=r_vals, dy=t_vals, t_offset=t_offset)

        # Custom case
        elif {"data"} >= mode_boundary_data:
            pass

        else:
            raise ValueError("Incorrect mode_boundary_data formatting.")

    def handle_polar_case(r_vals, t_vals, t_offset):
        # Add central mode
        central_mode_points = np.array([[0.0,0.0], [r_vals[0],0.0]])
        self.modes.append(Mode(index=0, mode_boundary=central_mode_points, mode_shape_type="polar"))

        pass



    def get_modes(self, **kwargs):
        match self.mode_type:
            case "cartesian": 
                return self.get_modes_cartesian(**kwargs)
            case "polar":
                return self.get_modes_polar(**kwargs)

    def get_modes_cartesian(self, **kwargs):
        # Test for dx, dy versus nx, ny
        if {'dx', 'dy'} <= kwargs.keys():
            return self.get_modes_cartesian_spacing(**kwargs)
        elif {'nx', 'ny'} <= kwargs.keys():
            return self.get_modes_cartesian_number(**kwargs)
        else:
            raise ValueError("Invalid input. Please specify either (dx and dy) or (nx and ny).")

    def get_modes_cartesian_spacing(self, dx=None, dy=None):
        # Validate dx and dy
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ValueError("dx and dy must be numbers.")
        if dx <= 0 or dx >= 1 or dy <= 0 or dy >= 1:
            raise ValueError("dx and dy must lie in the range 0 to 1")
        
        self.dx = dx
        self.dy = dy

        # Find modes in the quarter circle in the first quadrant. Symmetry will get use the rest
        for y_center in np.arange(0.0, 1.0, dy):            
            x_center = 0            
            while x_center**2 + y_center**2 <= 1.0:
                new_mode = CartesianMode(center=np.array([x_center, y_center]), dx=dx, dy=dy)
                self.mode_list.append(new_mode)
                
                # Find reflected modes
                if x_center != 0:
                    new_mode_reflected_x = new_mode.reflect_x()
                    self.mode_list.append(new_mode_reflected_x)                  

                if y_center != 0:
                    new_mode_reflected_y = new_mode.reflect_y()
                    self.mode_list.append(new_mode_reflected_y)                  

                if x_center !=0 and y_center != 0:
                    new_mode_reflected_xy = new_mode.reflect_x().reflect_y()
                    self.mode_list.append(new_mode_reflected_xy)                  

                x_center += dx 

    def get_modes_cartesian_number(self, nx=None, ny=None):
        raise RuntimeError("Feature not supported. Please use dx and dy.")

    def get_modes_polar(self, **kwargs):
        pass
    