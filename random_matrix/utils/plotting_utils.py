"""Utility functions that help with plotting figures."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from random_matrix.utils import array_utils, geometry_utils


def draw_ray(
    ax: plt.Axes,
    theta: float = 0.0,
    r_min: float = 0.0,
    r_max: float = 1.0,
    color: str = "tab:blue",
    linestyle: str = "--",
    alpha: float = 1.0,
) -> None:
    """Draws a straight line segment on the given Axes object representing a
    ray extending from the origin in the direction of the given angle.

    Parameters:
    -----------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object on which to draw the ray.
        theta : float, optional
            The angle (in radians) at which the ray extends from the origin.
            Default is 0 radians (pointing to the right).
        r_min : float, optional
            The minimum radius at which to start drawing the ray. Default is 0.
        r_max : float, optional
            The maximum radius at which to stop drawing the ray. Default is 1.
        color : str, optional
            The color of the ray. Default is "tab:blue".
        linestyle : str, optional
            The style of the line. Default is "--".
        alpha : float, optional
            The alpha (transparency) value of the ray.
            Default is 1.0 (fully opaque).

    Returns:
    -----------
        None
    """

    x = np.array([r_min * np.cos(theta), r_max * np.cos(theta)])
    y = np.array([r_min * np.sin(theta), r_max * np.sin(theta)])
    ax.plot(x, y, linestyle=linestyle, color=color, alpha=alpha)


def draw_circle(
    ax: plt.Axes,
    r: float = 1,
    t_min: float = 0,
    t_max: float = 2 * np.pi,
    color: str = "black",
    linestyle: str = "-",
) -> None:
    """Draws a circle of the given radius centered at the origin on the
    specified Axes object.

    Parameters:
    -----------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object on which to draw the circle.
        r : float, optional
            The radius of the circle. Default is 1.
        t_min : float, optional
            The minimum angle (in radians) at which to start drawing the
            circle. Default is 0 (starting at the positive x-axis).
        t_max : float, optional
            The maximum angle (in radians) at which to stop drawing the circle.
            Default is 2*pi (completing the full circle).
        color : str, optional
            The color of the circle. Default is "black".
        linestyle : str, optional
            The style of the line used to draw the circle. Default is "-".

    Returns:
    -----------
        None
    """
    t = np.linspace(t_min, t_max)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot(x, y, color=color, linestyle=linestyle)


def draw_line(
    ax: plt.Axes,
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    color: str = "black",
    linestyle: str = "-",
) -> None:
    """Draw a line on the given Matplotlib axis object from `start` to `end`.

    Parameters
    ----------
        ax : matplotlib.axes.Axes
            The Matplotlib axis object to draw the line on.
        start : numpy.ndarray
            The starting point of the line as a 2D vector of the form [x, y].
        end : numpy.ndarray
            The end point of the line as a 2D vector of the form [x, y].
        color : str, optional
            The color of the line. Defaults to "black".
        linestyle : str, optional
            The line style to use. Defaults to "-".

    Returns
    -------
        None
    """
    xs = np.array([start[0], end[0]])
    ys = np.array([start[1], end[1]])
    ax.plot(xs, ys, color=color, linestyle=linestyle)


def draw_vertical_chord(
    ax: plt.Axes,
    x: float,
    radius: float = 1,
    color: str = "black",
    linestyle: str = "-",
) -> None:
    """Draw a vertical chord on a circle given its x-coordinate and the radius
    of the circle.

    Parameters:
    -----------
        ax : matplotlib.axes.Axes
            The axes on which to draw the chord.
        x : float
            The x-coordinate of the chord.
        radius : float, optional
            The radius of the circle. Default is 1.
        color : str, optional
            The color of the chord. Default is "black".
        linestyle : str, optional
            The line style of the chord. Default is "-".

    Returns:
    --------
        None
    """
    y_top = geometry_utils.get_circle_coordinate(x, radius)  # type: ignore
    y_bottom = -y_top
    bottom_point = np.array([x, y_bottom])
    top_point = np.array([x, y_top])
    draw_line(ax, bottom_point, top_point, color=color, linestyle=linestyle)


def draw_horizontal_chord(
    ax: plt.Axes,
    y: float,
    radius: float = 1,
    color: str = "black",
    linestyle: str = "-",
) -> None:
    """Draw a horizontal chord on a circle given its y-coordinate and the
    radius of the circle.

    Parameters:
    -----------
        ax : matplotlib.axes.Axes
            The axes on which to draw the chord.
        y : float
            The y-coordinate of the chord.
        radius : float, optional
            The radius of the circle. Default is 1.
        color : str, optional
            The color of the chord. Default is "black".
        linestyle : str, optional
            The line style of the chord. Default is "-".

    Returns:
    --------
        None
    """

    x_right = geometry_utils.get_circle_coordinate(y, radius)  # type: ignore
    x_left = -x_right
    left_point = np.array([x_left, y])
    right_point = np.array([x_right, y])
    draw_line(ax, left_point, right_point, color=color, linestyle=linestyle)


def draw_interior_triangle(
    ax: plt.Axes,
    triangle: npt.NDArray[np.float64],
    polygon_points: npt.NDArray[np.float64],
    color: str = "tab:blue",
    linestyle: str = "-",
) -> None:
    """Draws a triangle inside a given polygon (excluding edges that coincide
    with the polygon edges).

    Parameters
    ----------
        ax : plt.Axes
            The axes on which the triangle will be drawn.
        triangle : Matrix[np.float32]
            A 3x2 matrix representing the three vertices of the triangle to be
            drawn.
        polygon_points : Matrix[np.float32]
            A Nx2 matrix representing the vertices of the polygon in which the
            triangle is contained.
        color : str, optional
            The color of the line that will be used to draw the triangle
            (default is "tab:blue").
        linestyle : str, optional
            The style of the line that will be used to draw the triangle
            (default is "-").

    Returns
    -------
        None
    """
    pairs = array_utils.get_pairs(triangle, cyclic=True)
    for first_point, second_point in pairs:
        first_index = array_utils.get_point_index(first_point, polygon_points)
        second_index = array_utils.get_point_index(
            second_point, polygon_points
        )
        if first_index is None or second_index is None:
            raise ValueError(
                "Triangle vertices do not coincide with polygon vertices!"
            )
        index_difference = np.abs(first_index - second_index)
        if index_difference != 1:
            draw_line(
                ax,
                start=first_point,
                end=second_point,
                color=color,
                linestyle=linestyle,
            )
