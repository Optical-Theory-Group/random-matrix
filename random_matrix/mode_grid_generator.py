from typing import Iterator, Any

import numpy as np
import numpy.typing as npt
import scipy.spatial

from random_matrix.mode_grid import ModeGrid
from random_matrix.utils import geometry_utils, array_utils
from random_matrix.utils.typevars import Numeric


class ModeGridGenerator:
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
        grid_params: dict[str, Any] = {},
    ) -> ModeGrid:
        grid_params["is_polar_grid"] = False

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
            grid_params=grid_params,
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
    def from_dx_dy():
        pass

    @classmethod
    def from_xy_vals():
        pass

    @classmethod
    def from_random(
        cls,
        num_points: int = 100,
        r_lim: float = 1.5,
        random_type: str = "delaunay",
        grid_params: dict[str, Any] = {},
    ):
        grid_params["is_polar_grid"] = False

        mode_boundary_list = cls.generate_random_mode_boundary_list(
            num_points=num_points, r_lim=r_lim, random_type=random_type
        )

        return ModeGrid(
            grid_params=grid_params,
            mode_boundary_list=mode_boundary_list,
            r_lim=r_lim,
        )

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
                h = s * np.sqrt(3) / 2
                dx = s
                dy = 2 * h

            case "rectangles":
                if isinstance(side_length, float):
                    dx = side_length
                    dy = dx
                else:
                    dx, dy = side_length

            case "hexagons":
                h = s * np.sqrt(3) / 2
                dx = 3 * s
                dy = 2 * h

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
            r_vals = np.append(0.0, r_vals)  # type: ignore
        if not array_utils.is_in_array(1.0, r_vals):  # type: ignore
            r_vals = np.append(r_vals, 1.0)  # type: ignore

        # Check that 0.0 and 2PI are in t_vals. If not, add them
        if not array_utils.is_in_array(0.0, t_vals):  # type: ignore
            t_vals = np.concatenate(t_vals, 0.0)  # type: ignore
        if not array_utils.is_in_array(2 * np.pi, t_vals):  # type: ignore
            t_vals = np.append(t_vals, 2 * np.pi)  # type: ignore

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
        points = np.random.uniform(-2*r_lim, 2*r_lim, (num_points, 2))
        triangulation = scipy.spatial.Delaunay(points)
        points = points[triangulation.simplices]

        for point in points:
            yield point

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
