"""
Classes for handling modes contained in the discrete angular spectrum

Author: Niall Francis Byrnes
https://niallbyrnes.com

"""

import numpy as np
from copy import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

class Modes:
    def __init__(self, mode_type="cartesian", **kwargs):
        # Validate mode type
        valid_mode_types = ["cartesian", "polar"]
        if mode_type not in valid_mode_types:
            raise ValueError(f"Mode type '{mode_type}' is invalid. Valid mode types are {valid_mode_types}.")
        self.mode_type = mode_type
        self.mode_list = []
        self.get_modes(**kwargs)

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
    
class Mode:

    valid_arguments = {"mode_wave_type": ["propagating", "evanescent"], "mode_shape_type": ["cartesian", "polar", "custom"]}
    
    def __init__(self, index, convex_hull=None, points=None, mode_shape_type="polar", mode_wave_type="propagating"):
        self.index = index

        # Check that input string arguments are from allowed values
        self._validate_string_input('mode_shape_type', mode_shape_type)
        self._validate_string_input('mode_wave_type', mode_wave_type)
        self.mode_shape_type = mode_shape_type
        self.mode_wave_type = mode_wave_type

        # Calculate convex_hull from points if not provided
        # Check types using isinstance
        ###########################################################################
        if convex_hull is None and points is not None:
            convex_hull = ConvexHull(points)        
        elif convex_hull is None and points is None:
            raise ValueError("At least one of 'convex_hull' or 'points' must be provided.")
        self.convex_hull = convex_hull

        # Check that points are appropriate given the mode wave type
        r_values = np.array([cartesian_to_polar(x, y) for x, y in convex_hull.points])[:,0]
        if mode_wave_type == "propagating" and np.any(r_values > 1.0):
            raise ValueError("Mode is propagating, but at least one point within the convex hull lies outside of the unit circle.")
        elif mode_wave_type == "evanescent" and np.any(r_values < 1.0):
            raise ValueError("Mode is evanescent, but at least one point within the convex hull lies inside of the unit circle.")

        # Handle special cases
        match mode_shape_type:
            case "polar":
                self._handle_polar_case()
            case "cartesian":
                self._handle_cartesian_case()
            case _:
                self._handle_custom_case()

    def _validate_string_input(self, argument_name, input_argument):
        if input_argument not in self.valid_arguments[argument_name]:
            raise ValueError(f"Invalid argument '{argument_name}': {input_argument}. Valid arguments are: {', '.join(map(str, self.valid_arguments[argument_name]))}")

    def _handle_polar_case(self):
        points_cartesian = self.convex_hull.points
        points_polar = np.array([cartesian_to_polar(x, y) for x, y in points_cartesian])
        
        # Check that points are appropriate for polar case
        # Handle special case of on-axis mode (circular region)
        r_values, t_values, is_central_mode = self._validate_polar_points(points_polar)
        self.is_central_mode = is_central_mode

        r_min, r_max = np.min(r_values), np.max(r_values)
        t_min, t_max = np.min(t_values), np.max(t_values)
        self.t_min = t_min
        self.t_max = t_max
        self.r_min = r_min
        self.r_max = r_max

        # Assign weight according to mode type
        self.weight = np.pi*r_max**2 if is_central_mode else 0.5*(t_max - t_min)*(r_max**2 - r_min**2)

    def _handle_cartesian_case(self):
        pass

    def _handle_custom_case(self):
        pass

    def _validate_polar_points(self, points_polar):
        if len(points_polar) != 4:
            raise ValueError("A polar mode should be constructed from exactly 4 points.")

        all_r_values = points_polar[:,0]
        all_t_values = points_polar[:,1]

        r_values = np.unique(points_polar[:,0])
        r_min = np.min(r_values)
        r_max = np.max(r_values)

        # If three of the original r values are 0, the points have been formatted
        # to represent the central mode
        # t_values are thus None and is_central_mode is True
        if np.isclose(r_min, 0) and len(r_values) == 2:
            return r_values, None, True

        t_values = np.unique(points_polar[:,1])

        # Check that there are exactly 2 r and t values
        if not (len(r_values) == 2 and len(t_values) == 2):
            raise ValueError("Points do not correctly align on polar grid.")

        return r_values, t_values, False

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

    def plot(self, ax=None, is_solo=True, show_guidelines=True, color="tab:red"):
        if is_solo:
            # Draw axes and k-space boundary
            fig, ax = plt.subplots()
            self._draw_ray(ax, r_min=-1, t_end=0,linestyle="-",color="black",alpha=0.4)
            self._draw_ray(ax, r_min=-1, t_end=np.pi/2,linestyle="-",color="black",alpha=0.4)
            ax.set_aspect("equal")
            self._draw_circle(ax)

        match self.mode_shape_type:
            case "polar":
                self._plot_polar(ax, show_guidelines, color)
            case "cartesian":
                self._plot_cartesian()
            case "custom":
                self._plot_custom()

    def _plot_polar(self, ax, show_guidelines, color):
        if self.is_central_mode:
            # mode is a small circle centred at the origin
            small_circle_radius = self.r_max
            self._draw_circle(ax, r=small_circle_radius, color=color)
        else:
            if show_guidelines:
                self._draw_ray(ax, r_min=-1, t_end=self.t_min, linestyle="--", color="tab:blue")
                self._draw_ray(ax, r_min=-1, t_end=self.t_max, linestyle="--", color="tab:blue")
                self._draw_circle(ax, r=self.r_min, linestyle="--", color="tab:blue")
                self._draw_circle(ax, r=self.r_max, linestyle="--", color="tab:blue")

            self._draw_ray(ax, self.t_min, r_min=self.r_min, r_max=self.r_max, linestyle="-", color=color)
            self._draw_ray(ax, self.t_max, r_min=self.r_min, r_max=self.r_max, linestyle="-", color=color)
            self._draw_circle(ax, t_min=self.t_min, t_max=self.t_max, r=self.r_min, linestyle="-", color=color)
            self._draw_circle(ax, t_min=self.t_min, t_max=self.t_max, r=self.r_max, linestyle="-", color=color)


    def _draw_ray(self, ax, t_end, r_min=0, r_max=1, color="tab:blue", linestyle="--",alpha=1.0):
        x = np.array([r_min*np.cos(t_end), r_max*np.cos(t_end)])
        y = np.array([r_min*np.sin(t_end), r_max*np.sin(t_end)])
        ax.plot(x, y, linestyle=linestyle, color=color,alpha=alpha)
 
    def _draw_circle(self, ax, r=1, t_min=0, t_max=2*np.pi, color="black", linestyle="-"):
        t = np.linspace(t_min, t_max)
        x = r*np.cos(t)
        y = r*np.sin(t)
        ax.plot(x,y,color=color, linestyle=linestyle)

def circle(x, r):
    return np.sqrt(r**2-x**2)

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y, x)
    return r, t

def main():
    points = [(0.5,0),(0,0.5),(0.6,0),(-0.0,0.6)]

    hull = ConvexHull(points)
    mode = Mode(index=0, convex_hull=hull, mode_wave_type="propagating")
    mode.plot()

    r1= 0.3
    r2 = 0.4
    t1 = 1.43
    t2 = 2.11
    points = [(r1*np.cos(t1),r1*np.sin(t1)),(r2*np.cos(t1),r2*np.sin(t1)),(r2*np.cos(t2),r2*np.sin(t2)),(r2*np.cos(t1),r2*np.sin(t1))]

    hull = ConvexHull(points)
    mode = Mode(index=0, convex_hull=hull, mode_wave_type="propagating")
    mode.plot()


if __name__ == "__main__":
    main()