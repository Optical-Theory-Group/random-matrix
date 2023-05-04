"""Mode class for use in scattering calculations.

In this context, a mode is defined as a non-zero, finite region of (k_x, k_y)
space. A mode thus represents a bundle of wavevectors that light can scatter
from or into.
"""

from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.spatial

from random_matrix.utils import array_utils, geometry_utils, plotting_utils


@dataclass(slots=True)
class Mode:
    """A class used to represent a mode.

    Attributes
    ----------
    vertices : np.ndarray
        Array of vertices that lie on the boundary of the mode.
    sides : list[namedtuple]
        List of namedtuples (Side), each of which represents a side of the
        mode's boundary. Each Side contains two attributes, namely
        "points" and "type". "points" is a (2,2) array that contains the two
        end points of the side. "type", which is either "line" or "arc"
        describes whether the side is a straight line between the two points,
        or a circular arc between them.
    index : int
        Keeps track of the mode mode within a ModeGrid object.
    clean_vertices : bool
        If True, duplicates will be removed from vertices. Points will also
        be ordered clockwise for consistency. Note that this attribute is not
        saved after an instance has been created.
    is_central : bool
        True if the mode is centro-symmetric about the origin. Helps when
        setting up the grid.
    wave_type : str
        A string describing if the mode is "propagating" or "evanescent".
    weight : float
        The area of the mode.
    triangulation : np.ndarray
        An array containing the vertices of the delaunay triangulation of the
        mode. Used for integration.

    Methods
    ----------
    plot
        Plots the mode on a given set of axes

    """

    vertices: npt.NDArray[np.float64]
    sides: list[NamedTuple]
    index: int = 0

    clean_vertices: InitVar[bool] = False

    is_central: bool = field(init=False)
    wave_type: str = field(init=False)
    weight: float = field(init=False)
    triangulation: npt.NDArray[np.float64] = field(init=False)

    # --------------------------------------------------------------------------
    # Constructor method
    # --------------------------------------------------------------------------

    def __post_init__(self, clean_vertices: bool) -> None:
        """Validates input data and determines computed attributes"""

        self._validate_input(self.vertices, self.sides)

        if clean_vertices:
            self.vertices = array_utils.remove_duplicate_points(self.vertices)
            self.vertices = geometry_utils.order_points(self.vertices)

        # Check if mode is a centro-symmetric or not
        self.is_central = self._get_is_central(self.vertices)
        self.wave_type = self._get_mode_wave_type(self.vertices)
        self.triangulation = self._get_triangulation(self.vertices)
        self.weight = self._get_weight(
            self.vertices, self.sides, self.wave_type
        )

    # --------------------------------------------------------------------------
    # Input validation and processing
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        vertices: npt.NDArray[np.float64],
        sides: list[NamedTuple],
    ) -> None:
        """Checks vertices and arc_points_list and throws exceptions
        if the data is improperly given"""

        # Check types
        if not isinstance(vertices, np.ndarray):
            raise ValueError(
                f"vertices must be a numpy array. "
                f"You gave a {type(vertices)}"
            )
        else:
            if vertices.ndim != 2 or vertices.shape[1] != 2:
                raise ValueError(
                    f"vertices must be a 2D array of (x,y) points.\n"
                    f"Your array has shape {np.shape(vertices)}.\n"
                )

        # Check that the number of points in vertices is correct
        num_vertices = len(vertices)
        num_arc_sides = len([side for side in sides if side.type == "arc"])

        if num_vertices == 2 and num_arc_sides == 0:
            raise ValueError(
                "A mode containing only two vertices must have arc "
                " points too. You have 0."
            )
        if num_vertices < 2:
            raise ValueError(
                "At least two vertices are required to " "define a mode."
            )

    @staticmethod
    def _get_mode_wave_type(vertices: npt.NDArray[np.float64]) -> str:
        """Determine whether the mode is propagating or evanescent

        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the mode.

        Returns
        -------
        str
            The wave type of the mode
        """
        r_vals = np.linalg.norm(vertices, axis=1)
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

    @staticmethod
    def _get_is_central(vertices: npt.NDArray[np.float64]) -> bool:
        """Get a boolean telling if the mode is centro-symmetric or not

        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the mode.

        Returns
        -------
        is_central : bool
            Is the mode centro-symmetric or not?
        """

        inverted_vertices = -vertices
        is_central = array_utils.is_equal_array(vertices, inverted_vertices)
        return is_central

    @staticmethod
    def _get_weight(
        vertices: npt.NDArray[np.float64],
        sides: list[NamedTuple],
        wave_type: str,
    ) -> float:
        """Get the weight associated with the mode.

        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the mode.
        sides : list[NamedTuple]
            Description of types of sides on boundary.
        wave_type : str
            The type of mode, i.e. "propagating" or "evanescent".

        Returns
        -------
        weight : float
            The area of the mode.
        """

        num_points = len(vertices)
        arc_sides = [side for side in sides if side.type == "arc"]
        num_arcs = len(arc_sides)
        base_polygon_area = 0.0

        if num_points > 2:
            base_polygon_area += geometry_utils.get_convex_polygon_area(
                vertices
            )

        # Check if there are any arc points. If not, we're done
        if num_arcs == 0:
            return base_polygon_area

        inner_crossing_list = []
        outer_crossing_list = []

        vertices_r_vals = set([np.linalg.norm(vertex) for vertex in vertices])
        vertices_r_max = max(vertices_r_vals)
        vertices_r_min = min(vertices_r_vals)

        for side in arc_sides:
            r_val = np.linalg.norm(side.points[0])

            if not np.isclose(r_val, vertices_r_max):
                inner_crossing_list.append(side.points)
            else:
                outer_crossing_list.append(side.points)

        # Areas where polygons meet the circle
        inner_edge_area = 0.0
        outer_edge_area = 0.0
        for inner_crossing in inner_crossing_list:
            inner_edge_area += geometry_utils.get_edge_area(
                points=inner_crossing, radius=vertices_r_min
            )
        for outer_crossing in outer_crossing_list:
            outer_edge_area += geometry_utils.get_edge_area(
                points=outer_crossing, radius=vertices_r_max
            )

        extra_area = 0.0
        extra_area += outer_edge_area
        extra_area -= inner_edge_area
        weight = base_polygon_area + extra_area
        return weight

    @staticmethod
    def _get_triangulation(
        vertices: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Get the delaunay triangulation of the mode.

        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the mode.

        Returns
        -------
        triangulation : np.ndarray
            An array containing the vertices of the triangles in the
            triangulation.
        """

        if len(vertices) == 2:
            return np.empty((0, 3, 2))

        triangulation_obj = scipy.spatial.Delaunay(vertices)
        triangulation: npt.NDArray[np.float64] = vertices[
            triangulation_obj.simplices
        ]
        return triangulation

    # --------------------------------------------------------------------------
    # Object representations
    # --------------------------------------------------------------------------

    def __str__(self) -> str:
        output = (
            f"Index: \n"
            f"{self.index}\n"
            f"Vertices: \n"
            f"{self.vertices},\n"
            f"Sides: \n"
            f"{self.sides}\n"
            f"Weight: {self.weight}\n"
            f"Wave type: {self.wave_type},\n"
            f"Is central: {self.is_central}\n"
        )
        return output

    def plot(
        self,
        ax: plt.Axes = None,
        is_solo: bool = True,
        triangulation_color: str = "tab:blue",
        show_index: Optional[bool] = False,
        show_triangulation: Optional[bool] = False,
    ) -> None:
        """Draw the mode on a given axis.

        Parameters
        ----------
        ax : plt.Axes
            The axis on which the mode will be drawn.
        triangulation_color : str
            Color that the triangles will be drawn in.
        show_index : bool
            Shows the mode's index at its center if True.
        show_triangulation : bool
            Also draws the mode's triangulation if True

        Returns
        -------
        None
        """

        color = "red" if self.wave_type == "propagating" else "blue"

        if is_solo:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")

        for side in self.sides:
            connection = side.points
            is_arc = side.type == "arc"
            if is_arc:
                radius = np.linalg.norm(connection[0])
                thetas = geometry_utils.cartesian_to_polar(connection)[:, 1]
                t_min = np.min(thetas)
                t_max = np.max(thetas)
                if t_max - t_min >= np.pi:
                    t_max = t_max - 2 * np.pi
                    t_min, t_max = t_max, t_min
                plotting_utils.draw_circle(
                    ax,
                    t_min=t_min,
                    t_max=t_max,
                    color=color,
                    r=radius,  # type: ignore
                )
            else:
                plotting_utils.draw_line(
                    ax,
                    start=connection[0],
                    end=connection[1],
                    color=color,
                )

        if show_index:
            central_coordinates = self.center
            x, y = central_coordinates
            plt.text(
                x,
                y,
                str(self.index),
                ha="center",
                va="center",
                color=color,
            )

        if show_triangulation:
            for triangle in self.triangulation:
                plotting_utils.draw_interior_triangle(
                    ax,
                    triangle,
                    color=triangulation_color,
                    polygon_points=self.vertices,
                )

    # -------------------------------------------------------------------------
    # Miscellaneous
    # -------------------------------------------------------------------------

    @property
    def center(self) -> npt.NDArray[np.float64]:
        """Find the central coordinates of the mode. Used in plotting.

        Parameters
        ----------
        None

        Returns
        -------
        central_coordinates : np.ndarray
            An array of length 2 containing the coordinates of the centre of
            the mode.
        """

        if self.is_central:
            center = np.array([0.0, 0.0])
        else:
            center = np.mean(self.vertices, axis=0)
        return center
