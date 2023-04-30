"""This module defines the GridGenerator class, which is used to generate
instances of ModeGrid.
"""

from typing import Iterator, Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
import skspatial.objects
import matplotlib.pyplot as plt

from random_matrix.mode import Mode
from random_matrix.mode_grid import ModeGrid
from random_matrix.utils import geometry_utils, array_utils, plotting_utils
from random_matrix.utils.typevars import Numeric


class GridGenerator:
    """Generator class for ModeGrid.

    A class for generating grids to be used to generate ModeGrid instances.
    Specifically, this class partitions the interior of the circle of
    radius r_lim into disjoint regions (or modes). These regions are also cut
    along the circimference of the circle of radius 1.0 where they are then
    split into multiple regions.

    To use this class, please call one of the constructor methods. These are

    from_tiling     -Random periodic tiling pattern
    from_dr_dt      -Polar grid from radial and angular spacings
    from_rt_vals    -Polar grid from arrays of radial and angular values
    from_dx_dy      -Rectangular grid from (x,y) lattice spacings
    from_xy_vals    -Rectangular grid from x and y boudary values.
    from_random     -Random grid from randomly generated points.

    More information about each of these methods can be found in their
    individual method documentation.

    Each constructor generates a list of dictionaries. Each dictionary
    represents an individual region of k-space within the partition that will
    be used as a mode. The dictionaries contain two key-value pairs.

        "mode_boundary" : np.ndarray
            An (N,2) numpy array containing the vertices of the convex boundary
            of the region.
        "arc_points_list": list[np.ndarray]
            A list of (2,2) numpy arrays, each of which represents a pair of
            points that will also be present within mode_bounday. The function
            of this list is to keep track of which pair of points within
            mode_boundary are connected by a circular arc, rather than a
            straight polygonal edge. These arcs arise where either in the
            polar grid generator or where a non-polar grid intersected either
            of the circles of radius 1.0 or r_lim.

    After setting up a list of such dictionaries, they are passed to the
    _get_mode_grid method, which builds a Mode object from each one. Finally,
    these Mode objects are passed to ModeGrid in a list.

    We note that, excluding the aforementioned circular arcs, only convex
    grids are currently supported in this class.

    """

    # -------------------------------------------------------------------------
    # Constructor methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_tiling(
        cls,
        tiling_type: str,
        side_length: float | tuple[float, float],
        r_lim: float = 1.1,
        rotation_angle: float = 0.0,
        translation_vector: npt.NDArray[Numeric] = np.array([0.0, 0.0]),
        grid_wave_type: str = "all",
    ) -> ModeGrid:
        base_lattice = cls.generate_base_lattice(
            side_length=side_length,
            tiling_type=tiling_type,
            r_lim=r_lim,
        )

        mode_boundary_list = cls.generate_tiling_mode_boundary_list(
            side_length=side_length,
            base_lattice=base_lattice,
            tiling_type=tiling_type,
            r_lim=r_lim,
            rotation_angle=rotation_angle,
            translation_vector=translation_vector,
        )

        return ModeGrid(
            mode_boundary_list=mode_boundary_list,
            r_lim=r_lim,
        )

    @classmethod
    def from_dr_dt(
        cls,
        dr: float,
        dt: float,
        r_lim: float = 1.5,
        include_central_mode: bool = True,
        rotation_angle: float = 0.0,
        grid_params: dict[str, Any] = {},
    ) -> ModeGrid:
        r_vals = np.arange(0, r_lim, dr)
        r_vals = np.append(r_vals, r_lim)
        t_vals = np.arange(0, 2 * np.pi, dt)
        return cls.from_rt_vals(
            r_vals=r_vals,
            t_vals=t_vals,
            include_central_mode=include_central_mode,
            rotation_angle=rotation_angle,
            grid_params=grid_params,
        )

    @classmethod
    def from_rt_vals(
        cls,
        r_vals: npt.NDArray[Numeric],
        t_vals: npt.NDArray[Numeric],
        include_central_mode: bool = True,
        rotation_angle: float = 0.0,
        grid_params: dict[str, Any] = {},
    ) -> ModeGrid:
        # Force grid_type to be polar
        grid_params["is_polar_grid"] = True
        r_lim = r_vals[-1]

        mode_boundary_list = cls.generate_polar_mode_boundary_list(
            r_vals=r_vals,
            t_vals=t_vals,
            rotation_angle=rotation_angle,
            include_central_mode=include_central_mode,
        )

        return ModeGrid(
            grid_params=grid_params,
            mode_boundary_list=mode_boundary_list,
            r_lim=r_lim,
        )

    @classmethod
    def from_dx_dy(cls):
        pass

    @classmethod
    def from_xy_vals(cls):
        pass

    @classmethod
    def from_random(
        cls,
        num_points: int = 100,
        r_lim: float = 1.5,
        random_type: str = "delaunay",
        grid_wave_type: str = "all",
    ):
        # Get a list of mode_boundaries
        mode_boundary_list = cls.generate_random_mode_boundary_list(
            num_points=num_points, r_lim=r_lim, random_type=random_type
        )  # type: ignore

        # Initialise mode_boundary dictionaries.
        mode_boundary_dict_list = (
            {
                "mode_boundary": mode_boundary,
                "arc_points_list": [],
            }
            for mode_boundary in mode_boundary_list
        )

        # Check where modes cut the circles and split into separate modes.
        mode_boundary_dict_list = cls.cut_and_filter(
            mode_boundary_dict_list=mode_boundary_dict_list,
            r_lim=r_lim,
            grid_wave_type=grid_wave_type,
        )

        return cls._get_mode_grid(mode_boundary_dict_list)

    @classmethod
    def _get_mode_grid(cls, mode_boundary_dict_list):
        mode_list = cls._get_mode_list(mode_boundary_dict_list)
        # return ModeGrid(mode_list=mode_list)

    def _get_mode_list(mode_boundary_dict_list):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        s = 0
        for i, mode_boundary_dict in enumerate(mode_boundary_dict_list):
            new_mode = Mode(index=i, mode_boundary_dict=mode_boundary_dict)
            s += new_mode.weight
            new_mode.plot(
                ax,
                boundary_color="red",
                triangulation_color="blue",
                index_color="black",
                show_index=False,
            )
            # yield new_mode

        print(s)
        print(np.pi * 2**2)

    # -------------------------------------------------------------------------
    # Lattice methods
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_base_lattice(
        side_length: float | tuple[float, float],
        tiling_type: str = "",
        r_lim: float = 1.0,
    ) -> Iterator[npt.NDArray[Numeric]]:
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

    @classmethod
    def generate_tiling_mode_boundary_list(
        cls,
        side_length: float | tuple[float, float],
        base_lattice: Iterator[npt.NDArray[Numeric]],
        tiling_type: str,
        r_lim: float,
        rotation_angle: float,
        translation_vector: npt.NDArray[Numeric],
    ) -> Iterator[npt.NDArray[Numeric]]:
        # Get unit cells at points in base lattice
        for point in base_lattice:
            unit_cell = cls.generate_unit_cell(
                center=point, tiling_type=tiling_type, side_length=side_length
            )

            for mode_boundary in unit_cell:
                # Reject mode if the inner-most point lies beyond r_lim
                norms = np.linalg.norm(mode_boundary, axis=1)
                if np.min(norms) >= r_lim:
                    continue

                # Rotate mode_boundaries obtained from unit_cell to align
                # with the rotated base lattice
                mode_boundary = geometry_utils.rotate_points(
                    mode_boundary, rotation_angle
                )
                mode_boundary = geometry_utils.translate_points(
                    mode_boundary, translation_vector
                )

                # Give points correct rotational order
                mode_boundary = geometry_utils.order_points(mode_boundary)

                yield mode_boundary

    @classmethod
    def generate_unit_cell(
        cls,
        center: npt.NDArray[Numeric],
        tiling_type: str,
        side_length: float | tuple[float, float],
    ) -> Iterator[npt.NDArray[Numeric]]:
        match tiling_type:
            case "triangles":
                yield from cls.generate_triangles(
                    center=center, side_length=side_length  # type: ignore
                )

            case "rectangles":
                yield from cls.generate_rectangles(
                    center=center, side_length=side_length
                )

            case "hexagons":
                yield from cls.generate_hexagons(
                    center=center, side_length=side_length  # type: ignore
                )

            case _:
                pass

    @staticmethod
    def generate_polar_mode_boundary_list(
        r_vals: npt.NDArray[Numeric],
        t_vals: npt.NDArray[Numeric],
        include_central_mode: bool,
        rotation_angle: float,
    ) -> Iterator[npt.NDArray[Numeric]]:
        # Reduce t values moduli 2PI for consistency
        t_vals = np.mod(t_vals, 2 * np.pi)

        # Check that 0.0 and 1.0 are in r_vals. If not, add them
        if not array_utils.is_in_array(0.0, r_vals):  # type: ignore
            r_vals = np.append(0.0, r_vals)
        if not array_utils.is_in_array(1.0, r_vals):  # type: ignore
            r_vals = np.append(r_vals, 1.0)

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not array_utils.is_in_array(0.0, t_vals):  # type: ignore
            t_vals = np.concatenate(t_vals, 0.0)  # type: ignore
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
            yield np.array(
                [
                    [r_central, 0.0],
                    [-r_central, 0.0],
                    [0.0, r_central],
                    [0.0, -r_central],
                ]
            )
        else:
            # Get wedges
            for t_1, t_2 in array_utils.get_pairs(t_vals):
                mode_boundary_polar = np.array(
                    [[0.0, 0.0], [r_central, t_1], [r_central, t_2]]
                )
                mode_boundary = geometry_utils.polar_to_cartesian(
                    mode_boundary_polar
                )
                # Rotate according to the provided rotation angle
                mode_boundary = geometry_utils.rotate_points(
                    mode_boundary, rotation_angle
                )

                mode_boundary = geometry_utils.order_points(mode_boundary)
                yield mode_boundary

        # All modes beyond the central ones
        # Get rid of 0 from the r_vals list
        r_vals = r_vals[1:]
        for r_1, r_2 in array_utils.get_pairs(r_vals):
            for t_1, t_2 in array_utils.get_pairs(t_vals):
                mode_boundary_polar = np.array(
                    [[r_1, t_1], [r_1, t_2], [r_2, t_1], [r_2, t_2]]
                )
                mode_boundary = geometry_utils.polar_to_cartesian(
                    mode_boundary_polar
                )
                # Rotate according to the provided rotation angle
                mode_boundary = geometry_utils.rotate_points(
                    mode_boundary, rotation_angle
                )

                mode_boundary = geometry_utils.order_points(mode_boundary)
                yield mode_boundary

    @staticmethod
    def generate_random_mode_boundary_list(
        num_points: int, r_lim: float, random_type: str
    ) -> Iterator[npt.NDArray[Numeric]]:
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

        triangulation = scipy.spatial.Delaunay(points)
        points = points[triangulation.simplices]

        for point in points:
            yield point

    # -------------------------------------------------------------------------
    # Cell-circle interaction methods
    # -------------------------------------------------------------------------

    @classmethod
    def cut_and_filter(
        cls,
        mode_boundary_dict_list: Iterator[
            dict[str, list[npt.NDArray[Numeric]] | npt.NDArray[Numeric]]
        ],
        r_lim: float,
        grid_wave_type: str,
    ) -> Iterator[
        dict[str, list[npt.NDArray[Numeric]] | npt.NDArray[Numeric]]
    ]:
        # Cut across propagating-evanescent mode boundary
        mode_boundary_dict_list = cls.cut_by_circle(
            mode_boundary_dict_list, radius=1.0
        )

        # Cut across maximum evanescent mode radial boundary
        mode_boundary_dict_list = cls.cut_by_circle(
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

        mode_boundary_dict_list = cls._filter_by_radius(
            mode_boundary_dict_list=mode_boundary_dict_list,
            r_min=r_min,
            r_max=r_max,
        )

        yield from mode_boundary_dict_list

    @classmethod
    def cut_by_circle(
        cls,
        mode_boundary_dict_list: Iterator[
            dict[str, list[npt.NDArray[Numeric]] | npt.NDArray[Numeric]]
        ],
        radius: float,
    ) -> Iterator[
        dict[str, list[npt.NDArray[Numeric]] | npt.NDArray[Numeric]]
    ]:
        for z, mode_boundary_dict in enumerate(mode_boundary_dict_list):
            boundary_points = mode_boundary_dict["mode_boundary"]
            arc_points_list = mode_boundary_dict["arc_points_list"]

            # Look at points in boundary_list that do not lie on the circle
            boundary_r_vals = np.linalg.norm(boundary_points, axis=1)
            boundary_r_vals = boundary_r_vals[
                ~np.isclose(boundary_r_vals, radius)
            ]

            # If there are 0 points outside the circle, we don't need to cut
            # Just return the original boundary_points. Also, there are no new
            # arcs
            if len(boundary_r_vals[boundary_r_vals > radius]) == 0:
                yield mode_boundary_dict  # type:ignore
                continue

            # Find all intersection points between boundary_points and the
            # circle
            intersection_points = (
                geometry_utils.get_polygon_circle_intersection_points(
                    boundary_points,
                    skspatial.objects.Circle([0.0, 0.0], radius),
                )
            )

            # If there are no intersections, we also don't need to cut
            if intersection_points is None:
                yield mode_boundary_dict  # type:ignore
                continue

            # Combine the original points with the intersection points
            augmented_boundary_points = np.vstack(
                (boundary_points, intersection_points)
            )
            augmented_boundary_points = array_utils.remove_duplicate_points(
                augmented_boundary_points
            )
            augmented_boundary_points = geometry_utils.order_points(
                augmented_boundary_points
            )

            # Roll points around so that an evanescent one is at the beginning
            # This is necessary for the algorithm to work
            r_vals = np.linalg.norm(augmented_boundary_points, axis=1)
            # index of the first value for which r > radius
            index = np.ravel(
                np.argwhere(
                    np.where(np.isclose(r_vals, radius), 0.0, r_vals) > radius
                )
            )[0]

            augmented_boundary_points = np.roll(
                augmented_boundary_points, -2 * index
            )

            # We now figure out where all the circular arcs are that will cut
            # up the modes
            intersection_types = cls._get_intersection_types(
                augmented_boundary_points, radius
            )
            arc_indices = cls._get_circular_arc_indices(intersection_types)

            # Leftover points are required to determine the left-over mode
            # once we've taken all the cut up pieces.
            leftover_points = np.copy(augmented_boundary_points).astype(
                np.complex128
            )
            leftover_mode_arc_list = []

            # Used for index tracking
            length = len(augmented_boundary_points)

            # yield a new mode for each comptued pair of arc indices
            for i, j in arc_indices:
                # Here we've found an arc and will cut across it
                new_circular_arc = np.array(
                    [
                        augmented_boundary_points[i],
                        augmented_boundary_points[j],
                    ]
                )

                # Add new arc to the total collection. This will be used
                # for the left-over mode
                leftover_mode_arc_list.append(new_circular_arc)

                # Build the new cut mode
                if j > i:
                    new_boundary_points = augmented_boundary_points[i : j + 1]
                    leftover_points[i : j + 1] = 1j * np.ones((j - i + 1, 2))
                else:
                    new_boundary_points = np.vstack(
                        (
                            augmented_boundary_points[i:],
                            augmented_boundary_points[: j + 1],
                        )
                    )
                    leftover_points[i:] = 1j * np.ones((length - i, 2))
                    leftover_points[: j + 1] = 1j * np.ones((j + 1, 2))

                new_dict = {
                    "mode_boundary": new_boundary_points,
                    "arc_points_list": [new_circular_arc],
                }

                yield new_dict

            # There is one last mode left. It includes: all the arc points
            # and any leftover points that weren't contained in any of the
            # previous slices
            leftover_mode_arc_array = np.array(leftover_mode_arc_list)

            num_rows = int(
                np.sum(np.isclose(np.imag(leftover_points), 0.0)) / 2
            )
            leftover_points = np.real(
                np.reshape(
                    leftover_points[np.isclose(np.imag(leftover_points), 0.0)],
                    (num_rows, 2),
                )
            )

            # No leftover points left. So the mode just consists of the arc
            # points
            if len(leftover_points) == 0:
                new_mode = np.vstack(leftover_mode_arc_array)
            else:
                new_mode = np.vstack(
                    (np.vstack(leftover_mode_arc_array), leftover_points)
                )

            new_mode = array_utils.remove_duplicate_points(new_mode)

            # Join the leftover arc list with the original one

            new_dict = {
                "mode_boundary": new_mode,
                "arc_points_list": arc_points_list + leftover_mode_arc_list,
            }
            yield new_dict

    @staticmethod
    def _get_intersection_types(points, radius):
        intersection_types = []
        vectors = np.vstack((points[1:] - points[:-1], points[0] - points[-1]))
        for i, (point, vector) in enumerate(zip(points, vectors)):
            if not np.isclose(np.linalg.norm(point), radius):
                # This point is not an inersection point
                intersection_types.append("0")
            else:
                # local tangent vector to the circle
                t = np.mod(np.arctan2(point[1], point[0]), 2 * np.pi)
                tangent = np.array([-np.sin(t), np.cos(t)])

                cross_product = np.cross(tangent, vector)

                # Check if the previous point was inside the circle or not
                previous_point = points[i - 1]
                prev_point_norm = np.linalg.norm(previous_point)
                last_point_inside = (
                    np.isclose(prev_point_norm, radius)
                    or prev_point_norm <= radius
                )
                if last_point_inside:
                    cross_product = -cross_product

                if np.isclose(cross_product, 0.0):
                    # Tangent
                    intersection_types.append("T")
                elif cross_product < 0.0:
                    # Deflection (hits circle but doesn't cross)
                    intersection_types.append("D")
                else:
                    # Crossing
                    intersection_types.append("C")

        return intersection_types

    @staticmethod
    def _get_circular_arc_indices(
        intersection_types: list[str],
    ) -> list[tuple[int, int]]:
        ignore_list = ["0", "D"]
        completed_list = []
        arc_indices = []
        length = len(intersection_types)

        found = False
        # The first is a special case
        # We want to pair it up with the final element
        for i, type_first in enumerate(intersection_types):
            if type_first == "C" or type_first == "T":
                found = True
                break
        if not found:
            raise RuntimeError(
                "WARNING: Strange edge case encountered."
                "Please change your grid."
            )

        for j, type_last in reversed(list(enumerate(intersection_types))):
            if type_last == "C" or type_last == "T":
                break

        arc_indices.append((j, i))
        if type_first == "C":
            completed_list.append(i)
        if type_last == "C":
            completed_list.append(j)

        # Begin main loop
        for i, type_one in enumerate(intersection_types):
            if type_one in ignore_list or i in completed_list:
                continue

            # WE've reached a new C or T
            # Find the next one along
            for j in range(i + 1, length):
                type_two = intersection_types[j]
                if type_two in ignore_list or j in completed_list:
                    continue

                # We've found a partner for type_one
                arc_indices.append((i, j))
                completed_list.append(i)
                if type_two == "C":
                    completed_list.append(j)
                break

        return arc_indices

    def _filter_by_radius(mode_boundary_dict_list, r_min, r_max):
        for mode_boundary_dict in mode_boundary_dict_list:
            mode_boundary = mode_boundary_dict["mode_boundary"]
            r_vals = np.linalg.norm(mode_boundary, axis=1)

            boundary_r_max = np.max(r_vals)
            boundary_r_min = np.min(r_vals)

            # Reject if the maximum r_val is beyind r_max
            if boundary_r_max > r_max and not np.isclose(
                boundary_r_max, r_max
            ):
                continue

            # Reject if the minimum r_val is beneath r_min
            if boundary_r_min < r_min and not np.isclose(
                boundary_r_min, r_min
            ):
                continue

            # Passed all tests
            yield mode_boundary_dict

    # -------------------------------------------------------------------------
    # Unit cell generator methods
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_triangles(
        center: npt.NDArray[Numeric],
        side_length: float,
    ) -> Iterator[npt.NDArray[Numeric]]:
        """Generate triangular unit cells (equilateral).

        The unit cell for our triangular lattice is

        _____
        \  / \
         \/___\  <--- center at mid-point on left
         /\   /
        /__\_/

        The "center", whose coordiantes are (x,y) is at the intersection
        of the two triangles on the left. Generated triangles are rotated
        so as to align with the base lattice.

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

    @staticmethod
    def generate_rectangles(
        center: npt.NDArray[Numeric],
        side_length: float | tuple[float, float],
    ) -> Iterator[npt.NDArray[Numeric]]:
        """Generate rectangular unit cells.

        The unit cell for the rectangular lattice is simply a rectangle. The
        "center", whose coordiantes are (x,y) is at the center of the
        rectangle. Generated rectangles are rotated so as to align with the
        base lattice.

        Parameters
        ----------
            x : float
                The x coordinate of the center of the unit cell.
            y : float
                The y coordinate of the center of the unit cell.
            side_length : float or (float, float)
                The side lengths of the rectangles. If only one is given,
                the rectangles will be made as squares.
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

    @staticmethod
    def generate_hexagons(
        center: npt.NDArray[Numeric],
        side_length: float,
    ) -> Iterator[npt.NDArray[Numeric]]:
        """Generate regular hexagonal unit cells.

        The unit cell for our hexagonal lattice is
          ____
         /    \
        /      \____  <--- center at mid-point of the left hexagon
        \      /    \
         \____/      \
              \      /
               \____/

        The "center", whose coordiantes are (x,y) is at the center of the
        hexagon on the left. Generated hexagons are rotated
        so as to align with the base lattice.

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
