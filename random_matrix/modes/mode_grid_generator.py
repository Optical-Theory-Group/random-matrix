"""Generator module for ModeGrid.

A module for generating grids to be used to generate ModeGrid instances.
Specifically, the functions in this module partition the interior of the
circle of radius r_lim > 1.0 into disjoint regions (or modes). These regions
are also cut along the circimference of the circle of radius 1.0, where they
are then split into multiple regions.

To use this module, please call one of the public constructor functions.
These are:

from_tiling        -Periodic tiling pattern
from_dr_dt         -Polar grid from radial and angular spacings
from_rt_vals       -Polar grid from arrays of radial and angular values
from_dx_dy (X)     -Rectangular grid from (x,y) lattice spacings
from_xy_vals (X)   -Rectangular grid from x and y boudary values
from_random        -Random grid from randomly generated points
from_data (X)      -Builds a grid based on polygon vertex data passed by a user


(X) indicates that the function is not yet supported. More information about
each of these methods can be found in their
individual function documentation.

Note that, excluding circular arcs, such as in polar grids or where modes are
cute by circles, only grids consisting of convex polygons are currently
supported. Grids consisting of concave polygons from user data should be used
with great caution.
"""

import collections
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.spatial
import skspatial.objects

from random_matrix.modes.mode import Mode, Side
from random_matrix.modes.mode_grid import ModeGrid
from random_matrix.utils import array_utils, geometry_utils, plotting_utils

# -----------------------------------------------------------------------------
# Public constructor functions
# -----------------------------------------------------------------------------


def from_tiling(
    tiling_type: str,
    side_length: float | tuple[float, float],
    r_lim: float = 1.2,
    rotation_angle: float = 0.0,
    translation_vector: npt.NDArray[np.float64] = np.array([0.0, 0.0]),
    grid_wave_type: str = "propagating",
) -> ModeGrid:
    """Generate ModeGrid from a tiling.

    Creates a ModeGrid object from a periodic planar tiling.

    Parameters
    ----------
        tiling_type : str
            The type of tiling. Possible options are:

            "triangles"
            "rectangles"
            "hexagons"

        side_length : float or (float, float)
            The side length of the polygons in the tiling unit cell. Tiling
            polygons are assumed to be regular. In the case of rectangles,
            however, a tuple describing the height and width of the rectangles
            can be specified. If a float is given, the rectangles will be
            squares.
        r_lim : float
            The radial extent of the tiling pattern.
        rotation_angle : float
            Angle by which the grid is rotated relative to its standard
            orientation (see unit cell generator functions).
        translation_vector: numpy.array
            Vector by which the grid will be translated relative to its
            standard definition.
        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        ModeGrid
            The ModeGrid object.
    """

    base_lattice: Iterator[npt.NDArray[np.float64]] = _generate_base_lattice(
        side_length=side_length,
        tiling_type=tiling_type,
        r_lim=r_lim,
    )

    vertices_list = _generate_tiling_vertices_list(
        side_length=side_length,
        base_lattice=base_lattice,
        tiling_type=tiling_type,
        r_lim=r_lim,
        rotation_angle=rotation_angle,
        translation_vector=translation_vector,
    )

    return _get_mode_grid(
        vertices_list=vertices_list,
        r_lim=r_lim,
        grid_wave_type=grid_wave_type,
    )


def from_dr_dt(
    dr: float,
    dt: float,
    r_lim: float = 1.2,
    include_central_mode: bool = True,
    rotation_angle: float = 0.0,
) -> ModeGrid:
    """Generate polar grid from dr and dt.

    Parameters
    ----------
        dr : float
            Spacing in the radial direction.
        dt : float
            Spacing in the angular direction.
        r_lim : float
            The radial extent of the tiling pattern.
        include_central_mode : bool
            If True, a circle of radius dr will included at the origin.
        rotation_angle : float
            Angle by which the grid is rotated relative to its standard
            definitions.
        translation_vector: numpy.array
            Vector by which the grid will be translated relative to its
            standard definition.
        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        ModeGrid
            The ModeGrid object.
    """

    r_vals = np.arange(0, r_lim, dr)
    r_vals = np.append(r_vals, r_lim)
    t_vals = np.arange(0, 2 * np.pi, dt)

    return from_rt_vals(
        r_vals=r_vals,
        t_vals=t_vals,
        include_central_mode=include_central_mode,
        rotation_angle=rotation_angle,
    )


def from_rt_vals(
    r_vals: npt.NDArray[np.float64],
    t_vals: npt.NDArray[np.float64],
    include_central_mode: bool = True,
    rotation_angle: np.float64 = np.float64(0.0),
) -> ModeGrid:
    """Generate polar grid from arrays of r and t values.

    Parameters
    ----------
        r_vals : numpy.array
            Array of radial values.
        t_vals : numpy.array
            Array of angular values.
        include_central_mode : bool
            If True, a circle of radius dr will included at the origin.
        rotation_angle : float
            Angle by which the grid is rotated relative to its
            definition.
        translation_vector: numpy.array
            Vector by which the grid will be translated relative to its
            standard definition.
        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        ModeGrid
            The ModeGrid object.
    """
    r_lim = np.max([1.0, np.max(r_vals)])

    mode_boundary_dict_list = _get_polar_mode_boundary_dict_list(
        r_vals=r_vals,
        t_vals=t_vals,
        rotation_angle=rotation_angle,
        include_central_mode=include_central_mode,
    )

    # Construct mode objects from dictionary list
    mode_list = list(_get_mode_list(mode_boundary_dict_list))

    return ModeGrid(mode_list=mode_list, r_lim=r_lim)


def from_dx_dy() -> None:
    pass


def from_xy_vals() -> None:
    pass


def from_random(
    num_points: int = 100,
    r_lim: float = 1.2,
    random_type: str = "delaunay",
    grid_wave_type: str = "all",
) -> ModeGrid:
    """Generate ModeGrid from randomly generated modes.

    Parameters
    ----------
        num_points : int
            Number of randomly generated points.
        r_lim : float
            The radial extent of the tiling pattern.
        random_type : str
            The method by which the modes are randomly generated.
            Possible options are:

            "delaunay"

        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        ModeGrid
            The ModeGrid object.
    """

    # Get a list of mode_boundaries
    vertices_list = _generate_random_vertices_list(
        num_points=num_points, r_lim=r_lim, random_type=random_type
    )

    return _get_mode_grid(
        vertices_list=vertices_list,
        r_lim=r_lim,
        grid_wave_type=grid_wave_type,
    )


def from_data() -> None:
    pass


# -------------------------------------------------------------------------
# Private constructor methods
# -------------------------------------------------------------------------


def _get_mode_grid(
    vertices_list: Iterator[npt.NDArray[np.float64]],
    r_lim: float,
    grid_wave_type: str,
) -> ModeGrid:
    """Intermediate function for generating ModeGrid.

    Should not be run directly. This method is used automatically
    by the other constructor methods.

    Parameters
    ----------
        mode_boundary_list : list[numpy.array]
            List of arrays describing the boundaries of the modes.
        r_lim : float
            The radial extent of the tiling pattern.
        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        ModeGrid
            The ModeGrid object.
    """

    # Initialise mode_boundary dictionaries
    mode_boundary_dict_list = (
        {
            "vertices": vertices,
            "arc_points_list": [],
        }
        for vertices in vertices_list
    )

    # Check where modes cut the circles and split into separate modes.
    mode_boundary_dict_list = _cut_and_filter(
        mode_boundary_dict_list=mode_boundary_dict_list,
        r_lim=r_lim,
        grid_wave_type=grid_wave_type,
    )

    # Construct mode objects from dictionary list
    mode_list = list(_get_mode_list(mode_boundary_dict_list))

    return ModeGrid(mode_list=mode_list, r_lim=r_lim)


def _get_mode_list(
    mode_boundary_dict_list: Iterator[dict[str, Any]]
) -> Iterator[Mode]:
    """Intermediate function for generating ModeGrid.

    Should not be run directly. This method is used automatically
    by the other constructor methods.

    Parameters
    ----------
        mode_boundary_dict_list : list[dict[numpy.array]]
            List of dictionaries describing mode boundaries as well as
            arc_points

    Returns
    -------
        Mode
            A Mode object generated from the dictionaries.
    """

    # These namedtuples keep track of whether a certain side is a line or arc
    for mode_boundary_dict in mode_boundary_dict_list:
        vertices = mode_boundary_dict.get("vertices")
        arc_points_list = mode_boundary_dict.get("arc_points_list")
        sides = []

        pairs = array_utils.get_pairs(vertices, cyclic=True)
        for first_point, second_point in pairs:
            connection = np.array([first_point, second_point])

            # Check if connection is equal to any of the arc points
            is_arc = False
            for arc_points in arc_points_list:
                if array_utils.is_equal_array(connection, arc_points):
                    is_arc = True
                    break

            if is_arc:
                new_side = Side(connection, "arc")
            else:
                new_side = Side(connection, "line")

            sides.append(new_side)

        new_mode = Mode(vertices=vertices, sides=sides)
        yield new_mode


# -------------------------------------------------------------------------
# Lattice methods
# -------------------------------------------------------------------------


def _generate_base_lattice(
    side_length: float | tuple[float, float],
    tiling_type: str,
    r_lim: float,
) -> Iterator[npt.NDArray[np.float64]]:
    """Generates base lattice used in grid generation from tiling.

    The base grid gives a lattice of points at which the unit cell
    is repeated.

    Parameters
    ----------
        side_length : float or (float, float)
            The side length of the polygons in the tiling unit cell. Tiling
            polygons are assumed to be regular. In the case of rectangles,
            however, a tuple describing the height and width of the rectangles
            can be specified. If a float is given, the rectangles will be
            squares.
        tiling_type : str
            The type of tiling. Possible options are:

            "triangles"
            "rectangles"
            "hexagons"

        r_lim : float
            The radial extent of the tiling pattern.

    Returns
    -------
        list[numpy.array]
            The aforementioned base lattice.
    """

    s = side_length

    # Work out base lattice spacing based on the tiling_type
    match tiling_type:
        case "triangles":
            h = s * np.sqrt(3.0) / 2.0
            dx = s
            dy = 2.0 * h

        case "rectangles":
            if isinstance(side_length, float):
                dx = side_length
                dy = dx
            else:
                dx, dy = side_length

        case "hexagons":
            h = s * np.sqrt(3) / 2.0
            dx = 3.0 * s  # type: ignore
            dy = 2.0 * h

        case _:
            raise ValueError(f"tiling_type = {tiling_type} is invalid.")

    # Set up lattice points arrays
    x_vals = np.arange(0.0, 2 * r_lim, dx)  # type: ignore
    x_vals_negative = -x_vals[1:][::-1]
    x_vals = np.concatenate((x_vals_negative, x_vals))
    y_vals = np.arange(0.0, 2 * r_lim, dy)
    y_vals_negative = -y_vals[1:][::-1]
    y_vals = np.concatenate((y_vals_negative, y_vals))

    for x in x_vals:
        for y in y_vals:
            point = np.array([x, y])
            yield point


def _generate_tiling_vertices_list(
    side_length: float | tuple[float, float],
    base_lattice: Iterator[npt.NDArray[np.float64]],
    tiling_type: str,
    r_lim: float,
    rotation_angle: float,
    translation_vector: npt.NDArray[np.float64],
) -> Iterator[npt.NDArray[np.float64]]:
    """Generates vertices for generation from tiling.

    Given the base lattice and tiling type, this method generate the list of
    vertices for each individual mode.

    Parameters
    ----------
        side_length : float or (float, float)
            The side length of the polygons in the tiling unit cell. Tiling
            polygons are assumed to be regular. In the case of rectangles,
            however, a tuple describing the height and width of the rectangles
            can be specified. If a float is given, the rectangles will be
            squares.
        base_lattice : numpy.array
            Lattice of points at which the unit cell is repeated.
        tiling_type : str
            The type of tiling. Possible options are:

            "triangles"
            "rectangles"
            "hexagons"

        r_lim : float
            The radial extent of the tiling pattern.
        rotation_angle : float
            Angle by which the grid is rotated relative to its
            definition.
        translation_vector: numpy.array
            Vector by which the grid will be translated relative to its
            standard definition.

    Returns
    -------
        list[numpy.array]
            List of mode boundaries.
    """

    # Get unit cells at points in base lattice
    for point in base_lattice:
        unit_cell = _generate_unit_cell(
            center=point, tiling_type=tiling_type, side_length=side_length
        )

        for vertices in unit_cell:
            # Reject mode if the inner-most point lies beyond r_lim
            norms = np.linalg.norm(vertices, axis=1)
            if np.min(norms) >= r_lim:
                continue

            vertices = geometry_utils.rotate_points(vertices, rotation_angle)
            vertices = geometry_utils.translate_points(
                vertices, translation_vector
            )

            # Give points correct rotational order
            vertices = geometry_utils.order_points(vertices)
            yield vertices


def _generate_unit_cell(
    center: npt.NDArray[np.float64],
    tiling_type: str,
    side_length: float | tuple[float, float],
) -> Iterator[npt.NDArray[np.float64]]:
    """Generates unit cells for generation from tiling.

    Generates unit cells based on the tyling type. These are then
    reapeated at points in the base lattice.

    Parameters
    ----------
        center : numpy.array
            The coordinates of the center of the unit cell. These
            values come from the base lattice.
        tiling_type : str
            The type of tiling. Possible options are:

            "triangles"
            "rectangles"
            "hexagons"

        side_length : float or (float, float)
            The side length of the polygons in the tiling unit cell.
            Tiling polygons are assumed to be regular. In the case of
            rectangles, however, a tuple describing the height and width
            of the rectangles can be specified. If a float is given,
            the rectangles will be squares

    Returns
    -------
        list[numpy.array]
            Array containing vertices of the unit cell.
    """

    match tiling_type:
        case "triangles":
            yield from _generate_triangles(
                center=center, side_length=side_length  # type: ignore
            )

        case "rectangles":
            yield from _generate_rectangles(
                center=center, side_length=side_length
            )

        case "hexagons":
            yield from _generate_hexagons(
                center=center, side_length=side_length  # type: ignore
            )

        case _:
            pass


def _get_polar_mode_boundary_dict_list(
    r_vals: npt.NDArray[np.float64],
    t_vals: npt.NDArray[np.float64],
    include_central_mode: bool,
    rotation_angle: float,
) -> Iterator[dict[str, Any]]:
    """Generates polar mode boundaries for polar grids.

    Parameters
    ----------
        r_vals : numpy.array
            Array of radial values.
        t_vals : numpy.array
            Array of angular values.
        include_central_mode : bool
            If True, a circle of radius dr will included at the origin.
        rotation_angle : float
            Angle by which the grid is rotated relative to its standard
            definition.

    Returns
    -------
        list[numpy.array]
            Array containing vertices of the polar modes.
    """

    # Reduce t values moduli 2PI for consistency
    t_vals = np.mod(t_vals, 2 * np.pi)

    # Check that 0.0 and 1.0 are in r_vals. If not, add them
    if not array_utils.is_in_array(0.0, r_vals):  # type: ignore
        r_vals = np.append(0.0, r_vals)
    if not array_utils.is_in_array(1.0, r_vals):  # type: ignore
        r_vals = np.append(r_vals, 1.0)

    # Check that 0.0 and 2PI are in t_vals. If not, add them
    if not array_utils.is_in_array(0.0, t_vals):  # type: ignore
        t_vals = np.append(t_vals, 0.0)
    if not array_utils.is_in_array(2 * np.pi, t_vals):  # type: ignore
        t_vals = np.append(t_vals, 2 * np.pi)

    # Remove duplicates and sort
    r_vals = array_utils.remove_duplicate_points(r_vals)
    t_vals = array_utils.remove_duplicate_points(t_vals)
    r_vals = np.sort(r_vals)
    t_vals = np.sort(t_vals)

    r_central = r_vals[1]
    # Handle central mode case
    if include_central_mode:
        vertices = np.array(
            [
                [r_central, 0.0],
                [0.0, r_central],
                [-r_central, 0.0],
                [0.0, -r_central],
            ]
        )

        arc_points_list = list(array_utils.get_pairs(vertices, cyclic=True))
        new_mode_boundary_dict = {
            "vertices": vertices,
            "arc_points_list": arc_points_list,
        }
        yield new_mode_boundary_dict

    else:
        # Get wedges
        for t_1, t_2 in array_utils.get_pairs(t_vals):
            vertices_polar = np.array(
                [[0.0, 0.0], [r_central, t_1], [r_central, t_2]]
            )
            vertices = geometry_utils.polar_to_cartesian(vertices_polar)

            # Rotate according to the provided rotation angle
            vertices = geometry_utils.rotate_points(vertices, rotation_angle)

            arc_points_list = [vertices[1:]]
            new_mode_boundary_dict = {
                "vertices": vertices,
                "arc_points_list": arc_points_list,
            }
            yield new_mode_boundary_dict

    # All modes beyond the central ones
    # Get rid of 0 from the r_vals list
    r_vals = r_vals[1:]
    for r_1, r_2 in array_utils.get_pairs(r_vals):
        for t_1, t_2 in array_utils.get_pairs(t_vals):
            vertices_polar = np.array(
                [[r_1, t_1], [r_1, t_2], [r_2, t_2], [r_2, t_1]]
            )
            vertices = geometry_utils.polar_to_cartesian(vertices_polar)
            # Rotate according to the provided rotation angle
            vertices = geometry_utils.rotate_points(vertices, rotation_angle)

            arc_points_list = [vertices[0:2], vertices[2:]]
            new_mode_boundary_dict = {
                "vertices": vertices,
                "arc_points_list": arc_points_list,
            }
            yield new_mode_boundary_dict


def _generate_random_vertices_list(
    num_points: int, r_lim: float, random_type: str
) -> Iterator[npt.NDArray[np.float64]]:
    """Generate mode boundaries for randomly generated grids.

    Parameters
    ----------
        num_points : int
            Number of randomly generated points.
        r_lim : float
            The radial extent of the tiling pattern.
        random_type : str
            The method by which the modes are randomly generated.
            Possible options are:

            "delaunay"
            "voronoi"

    Returns
    -------
        ModeGrid
            The vertices of the randomly generated modes.
    """

    factor = 1.1
    points = np.random.uniform(
        -factor * r_lim, factor * r_lim, (num_points, 2)
    )

    corner_points = np.array(
        [
            [-factor * r_lim, -factor * r_lim],
            [-factor * r_lim, factor * r_lim],
            [factor * r_lim, -factor * r_lim],
            [factor * r_lim, factor * r_lim],
        ]
    )
    points = np.vstack((points, corner_points))

    match random_type:
        case "delaunay":
            triangulation = scipy.spatial.Delaunay(points)
            points = points[triangulation.simplices]
        case "voronoi":
            v = scipy.spatial.Voronoi(points)
            poly_points = []
            regions = [
                region
                for region in v.regions
                if len(region) > 0 and not -1 in region
            ]
            for region in regions:
                poly_points.append(np.array([v.vertices[i] for i in region]))
            points = poly_points
    yield from points


# -------------------------------------------------------------------------
# Cell-circle interaction methods
# -------------------------------------------------------------------------


def _cut_and_filter(
    mode_boundary_dict_list: Iterator[dict[str, Any]],
    r_lim: float,
    grid_wave_type: str,
) -> Iterator[dict[str, Any]]:
    """Separates modes across circular boundaries and filter modes lying
    beyond r_lim.

    Parameters
    ----------
        mode_boundary_dict_list : list[dict]
            List containing dictionaries for each mode.
        r_lim : float
            Limit beyond which modes will be filtered
        grid_wave_type : str
            Determines what types of modes will be included in the grid.
            Possible options are

            "propagating"
            "evanescent"
            "all"

    Returns
    -------
        list[dict]
            List of the filtered mode dictionaries.
    """

    # Cut across propagating-evanescent mode boundary
    mode_boundary_dict_list = _cut_by_circle(
        mode_boundary_dict_list, radius=1.0
    )

    # Cut across maximum evanescent mode radial boundary
    mode_boundary_dict_list = _cut_by_circle(
        mode_boundary_dict_list, radius=r_lim
    )

    # Filter according to what modes are desired.
    match grid_wave_type:
        case "all":
            r_min = 0.0
            r_max = r_lim
        case "propagating":
            r_min = 0.0
            r_max = 1.0
        case "evanescent":
            r_min = 1.0
            r_max = r_lim

    mode_boundary_dict_list = _filter_by_radius(
        mode_boundary_dict_list=mode_boundary_dict_list,
        r_min=r_min,
        r_max=r_max,
    )

    yield from mode_boundary_dict_list


def _cut_by_circle(
    mode_boundary_dict_list: Iterator[dict[str, Any]],
    radius: float,
) -> Iterator[dict[str, Any]]:
    """Core method for splitting modes across circular rings.

    For a given input mode, a collection of modes that are separated by
    a circle of r=radius are returned. Furthermore, the points at which
    the boundary of a mode is a circular arc is recorded in the
    arc_points_list value in the dictionaries.

    Parameters
    ----------
        mode_boundary_dict_list : list[dict]
            List containing initial dictionaries for each uncut mode.
        radius : float
            radius of the circle through which the mode will be split.

    Returns
    -------
        list[dict]
            Updated list of dictionaries of split modes.
    """

    for mode_boundary_dict in mode_boundary_dict_list:
        vertices = mode_boundary_dict.get("vertices")
        arc_points_list = mode_boundary_dict.get("arc_points_list")

        vertices_r_vals = np.linalg.norm(vertices, axis=1)
        vertices_r_vals = vertices_r_vals[~np.isclose(vertices_r_vals, radius)]

        # If there are no points outside the circle, we don't need to cut
        # our mode. We can just return the data as is
        if len(vertices_r_vals[vertices_r_vals > radius]) == 0:
            yield mode_boundary_dict
            continue

        # Now find all intersection points
        intersection_points = (
            geometry_utils.get_polygon_circle_intersection_points(
                vertices,
                skspatial.objects.Circle([0.0, 0.0], radius),
            )
        )

        # If there are no intersections, we also don't need to cut,
        # since the mode is entirely outside of the circle
        if intersection_points is None:
            yield mode_boundary_dict
            continue

        # Combine the original points with the intersection points
        # and clean them up
        augmented_vertices = np.vstack((vertices, intersection_points))
        augmented_vertices = array_utils.remove_duplicate_points(
            augmented_vertices
        )
        augmented_vertices = geometry_utils.order_points(augmented_vertices)

        # Roll points around so that an evanescent one is at the beginning
        # This is necessary for the algorithm to work
        r_vals = np.linalg.norm(augmented_vertices, axis=1)
        # This intimidating expression just gives the index of the first
        # value for which r > radius
        index = np.ravel(
            np.argwhere(
                np.where(np.isclose(r_vals, radius), 0.0, r_vals) > radius
            )
        )[0]

        augmented_vertices = np.roll(augmented_vertices, -2 * index)

        # We now figure out where all the circular arcs are that will cut
        # up the modes
        intersection_types = _get_intersection_types(
            augmented_vertices, radius
        )
        arc_indices = _get_circular_arc_indices(intersection_types)

        # arc_indices is None if there are only deflection points.
        # In this case, there is no need to cut our mode.
        if arc_indices is None:
            yield mode_boundary_dict
            continue

        # From here on, we will need to cut the mode. There are two
        # components to this. We will have a bunch of modes lying beyond
        # the circle and a leftover mode inside. leftover_points will
        # keep track of those that will be left over after we have made
        # all the new modes. remove_list are points that are removed from
        # augmented_vertices
        leftover_points = np.copy(augmented_vertices)
        remove_list = []
        leftover_mode_arc_list = []
        length = len(leftover_points)

        # Each element of arc_indices gives one new mode
        for i, j in arc_indices:
            new_circular_arc = np.array(
                [
                    augmented_vertices[i],
                    augmented_vertices[j],
                ]
            )
            leftover_mode_arc_list.append(new_circular_arc)

            # Case where the first cross joined with the last
            if j > i:
                new_vertices = augmented_vertices[i : j + 1]
                if abs(j - i) > 1:
                    remove_list += [z for z in range(i + 1, j)]
            else:
                new_vertices = np.vstack(
                    (
                        augmented_vertices[i:],
                        augmented_vertices[: j + 1],
                    )
                )

                remove_list += [z for z in range(i + 1, length)]
                remove_list += [z for z in range(j)]

            new_dict = {
                "vertices": new_vertices,
                "arc_points_list": [new_circular_arc],
            }
            yield new_dict

        leftover_points = np.delete(leftover_points, remove_list, axis=0)
        leftover_points = array_utils.remove_duplicate_points(leftover_points)
        new_dict = {
            "vertices": leftover_points,
            "arc_points_list": arc_points_list + leftover_mode_arc_list,
        }
        yield new_dict


def _get_intersection_types(
    points: npt.NDArray[np.float64], radius: float
) -> list[str]:
    """Intermediate method used in splitting modes across circles.

    This method returns a list of strings that describe what type of
    intersection with the circle occurs for each vertex in the augmented
    vertices of the mode (union of original vertices and all intersection
    points with the circle).

    The options for the entries of this list are

    "0"     - No intersection at this vertex
    "D"     - Deflect: vertex lies on the boundary, but the next segment
              moves away from the circle.
    "C"     - Cross: boundary crosses the circle at this vertex.
    "T"     - Tangent: segment is tangent to the circle at this vertex.

    Parameters
    ----------
        points : numpy.array
            Points in the augmented vertices.
        radius : float
            radius of the circle through which the mode will be split.

    Returns
    -------
        list[str]
            List containing strings as described.
    """

    intersection_types = []
    # These vectors are directed along the line segments of the polygon
    vectors = np.vstack(
        (
            points[1:] - points[:-1],  # type:ignore
            points[0] - points[-1],
        )
    )

    # We loop through each vertex (point) of the polygon
    # vector is the vector pointing towards the next vertex. It is thus
    # the local tangent vector of the polygon as one moves along the boundary
    for i, (point, vector) in enumerate(zip(points, vectors)):
        if not np.isclose(np.linalg.norm(point), radius):
            # This point doesn't intersect the circle
            intersection_types.append("0")
        else:
            # t is the local tangent vector to the circle
            t = np.mod(np.arctan2(point[1], point[0]), 2 * np.pi)
            tangent = np.array([-np.sin(t), np.cos(t)])

            # We want to determine whether our polygon tangent vector
            # points into or away from the circle. This tells us if the
            # polygon crosses the circle or is deflected away from it.
            cross_product = np.cross(tangent, vector)

            # Check if the previous point was inside the circle or not
            previous_point = points[i - 1]
            prev_point_norm = np.linalg.norm(previous_point)
            previous_point_inside = (
                np.isclose(prev_point_norm, radius)
                or prev_point_norm <= radius
            )

            if previous_point_inside:
                # Going from inside to outside or vice versa requires a
                # change in sign. Draw a picture to see why.
                cross_product = -cross_product

            if np.isclose(cross_product, 0.0):
                # Tangent to the circle
                intersection_types.append("T")
            elif cross_product < 0.0:
                # Deflection (hits circle but doesn't cross)
                intersection_types.append("D")
            else:
                # Crosses the circle
                intersection_types.append("C")

    return intersection_types


def _get_circular_arc_indices(
    intersection_types: list[str],
) -> set[tuple[int, int]] | None:
    """Intermediate method used in splitting modes across circles.

    This method returns a list of tuples describing pairs of indices for which
    the boundary of the cut mode is an arc, rather than a line segment. Thus,
    when we later have a mode whose vertices are defined by an array of 2D
    points, we will know which sides should be line segments and which should
    be circular arcs.

    Parameters
    ----------
        intersection_types : list[str]
            Return of the _get_intersection_types methods.

    Returns
    -------
        list[tuple]
            List containing tuples as described.
    """

    ignore_list = {"0", "D"}
    completed_list = set()
    arc_indices = set()
    length = len(intersection_types)

    # We find the first element of the list that is not in the ignore list
    # This element is a special case: it must be paired with the last one.
    # Once the loop finishes, i will be its index.
    found = False
    for i, type_first in enumerate(intersection_types):
        if type_first not in ignore_list:
            found = True
            break
    # We didn't find any. Conclusion: there are no tangent or crossing
    # points. Therefore the mode won't get cut.
    if not found:
        return None

    # Now we find the last element. This will be paired with the first one
    # we just found. j will be its index
    for j, type_last in reversed(list(enumerate(intersection_types))):
        if type_last not in ignore_list:
            break

    # C can only be paired with one other. Thus, add them to the completed
    # list if they are Cs
    arc_indices.add((j, i))
    if type_first == "C":
        completed_list.add(i)
    if type_last == "C":
        completed_list.add(j)

    # Now we go through and simply pair up everything with the next one
    for i, type_one in enumerate(intersection_types):
        if type_one in ignore_list or i in completed_list:
            continue
        # At this point we've reached a new C or T
        # Its partner will be the next one in the list.
        for j in range(i + 1, length):
            type_two = intersection_types[j]
            if type_two in ignore_list or j in completed_list:
                continue
            # At this point we've found a partner.
            arc_indices.add((i, j))
            completed_list.add(i)
            # If type two is a "T", it can still connect with one after it
            # so we don't flag it as completed.
            if type_two == "C":
                completed_list.add(j)
            break

    return arc_indices


def _filter_by_radius(
    mode_boundary_dict_list: Iterator[dict[str, Any]],
    r_min: float,
    r_max: float,
) -> Iterator[dict[str, Any]]:
    """Method for filtering modes that lie beyond given radial limits.

    Modes fully contained in the regions r < r_min and r > r_max will be
    discarded by this method.

    Parameters
    ----------
        mode_boundary_dict_list : list[dict]
            List containing initial dictionaries for each uncut mode.
        r_min : float
            Lower limit for accepting modes.
        r_max : float
            Upper limit for accepting modes.

    Returns
    -------
        list[dict]
            Updated list of dictionaries of filtered modes.
    """

    for mode_boundary_dict in mode_boundary_dict_list:
        vertices = mode_boundary_dict["vertices"]
        r_vals = np.linalg.norm(vertices, axis=1)

        boundary_r_max = np.max(r_vals)
        boundary_r_min = np.min(r_vals)

        # Reject if the maximum r_val is beyind r_max
        if boundary_r_max > r_max and not np.isclose(boundary_r_max, r_max):
            continue

        # Reject if the minimum r_val is beneath r_min
        if boundary_r_min < r_min and not np.isclose(boundary_r_min, r_min):
            continue

        # Passed all tests
        yield mode_boundary_dict


# -------------------------------------------------------------------------
# Unit cell generator methods
# -------------------------------------------------------------------------


def _generate_triangles(
    center: npt.NDArray[np.float64],
    side_length: float,
) -> Iterator[npt.NDArray[np.float64]]:
    """Generate triangular unit cells (equilateral).

    The unit cell for our triangular lattice is

    ____
    \  /\
     \/__\  <--- center at cross-shaped intersection point on the left
     /\  /
    /__\/

    The "center", whose coordiantes are (x,y) is at the intersection of the
    two triangles on the left. Generated triangles are rotated so as to align
    with the base lattice.

    Parameters
    ----------
        x : float
            The x coordinate of the center of the unit cell.
        y : float
            The y coordinate of the center of the unit cell.
        side_length : float
            The side length of the triangles.
        rotation_angle : float
            Angle by which the triangles are rotated.

    Returns
    -------
        numpy.ndarray (generator)
            Array containing the vertices of the four triangles.
    """

    # h is the vertical height of a triangle
    x, y = center
    s = side_length
    h = s * np.sqrt(3) / 2

    yield np.array([[x, y], [x - s / 2, y + h], [x + s / 2, y + h]])
    yield np.array([[x, y], [x + s / 2, y + h], [x + s, y]])
    yield np.array([[x, y], [x + s / 2, y - h], [x + s, y]])
    yield np.array([[x, y], [x - s / 2, y - h], [x + s / 2, y - h]])


def _generate_rectangles(
    center: npt.NDArray[np.float64],
    side_length: float | tuple[float, float],
) -> Iterator[npt.NDArray[np.float64]]:
    """Generate rectangular unit cells.

    The unit cell for the rectangular lattice is simply a rectangle. The
    "center", whose coordiantes are (x,y) is at the center of the rectangle.
    Generated rectangles are rotated so as to align with the base lattice.

    Parameters
    ----------
        x : float
            The x coordinate of the center of the unit cell.
        y : float
            The y coordinate of the center of the unit cell.
        side_length : float or (float, float)
            The side lengths of the rectangles. If only one is given, the
            rectangles will be made as squares.
        rotation_angle : float
            Angle by which the rectangles are rotated.

    Returns
    -------
        numpy.ndarray (generator)
            Array containing the vertices of the rectangle.
    """
    x, y = center

    if isinstance(side_length, float):
        dx = side_length
        dy = dx
    else:
        dx, dy = side_length

    # width and height of the rectangle
    w = dx / 2
    h = dy / 2

    yield np.array(
        [[x - w, y + h], [x - w, y - h], [x + w, y + h], [x + w, y - h]]
    )


def _generate_hexagons(
    center: npt.NDArray[np.float64],
    side_length: float,
) -> Iterator[npt.NDArray[np.float64]]:
    """Generate regular hexagonal unit cells.

    The unit cell for our hexagonal lattice is
      _____
     /     \
    /       \_____<--- center at centroid of the left hexagon
    \       /     \
     \_____/       \
           \       /
            \_____/

    The "center", whose coordiantes are (x,y) is at the center of the hexagon
    on the left. Generated hexagons are rotated so as to align with the base
    lattice.

    Parameters
    ----------
        x : float
            The x coordinate of the center of the unit cell.
        y : float
            The y coordinate of the center of the unit cell.
        side_length : float
            The side length of the hexagons.
        rotation_angle : float
            Angle by which the hexagons are rotated.

    Returns
    -------
        numpy.ndarray (generator)
            Array containing the vertices of the two hexagons.
    """

    x, y = center

    # s is the side length and h is the vertical height of a hexagon
    s = side_length
    h = s * np.sqrt(3) / 2

    yield np.array(
        [
            [x - s, y],
            [x - s / 2, y + h],
            [x + s / 2, y + h],
            [x + s, y],
            [x + s / 2, y - h],
            [x - s / 2, y - h],
        ]
    )
    yield np.array(
        [
            [x + s / 2, y - h],
            [x + s, y],
            [x + 2 * s, y],
            [x + 5 / 2 * s, y - h],
            [x + 2 * s, y - 2 * h],
            [x + s, y - 2 * h],
        ]
    )
