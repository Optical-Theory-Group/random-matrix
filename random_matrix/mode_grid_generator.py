from typing import Iterator

import numpy as np
import numpy.typing as npt
import scipy.spatial

from random_matrix.mode_grid import ModeGrid
from random_matrix.utils import geometry_utils
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
        r_lim: float = 1.0,
        rotation_angle: float = 0.0,
        translation_vector: npt.NDArray[Numeric] = np.array([0.0, 0.0]),
        grid_params: dict[str, str] = {},
    ) -> None:
        base_lattice = cls.generate_base_lattice(
            side_length=side_length,
            tiling_type=tiling_type,
            r_lim=r_lim,
            rotation_angle=rotation_angle,
            translation_vector=translation_vector,
        )

        mode_boundary_list = cls.generate_mode_boundary_list(
            side_length=side_length,
            base_lattice=base_lattice,
            tiling_type=tiling_type,
            r_lim=r_lim,
            rotation_angle=rotation_angle,
        )

        return ModeGrid(
            grid_params=grid_params,
            mode_boundary_list=mode_boundary_list,
            r_lim=r_lim,
        )

    @staticmethod
    def from_dr_dt():
        pass

    @staticmethod
    def from_rt_vals():
        pass

    @staticmethod
    def from_dx_dy():
        pass

    @staticmethod
    def from_xy_vals():
        pass

    @staticmethod
    def from_random():
        pass

    # -------------------------------------------------------------------------
    # Lattice methods
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_base_lattice(
        side_length: float | tuple[float, float],
        tiling_type: str = "",
        r_lim: float = 1.0,
        rotation_angle: float = 0.0,
        translation_vector: npt.NDArray[Numeric] = np.array([0.0, 0.0]),
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

                # Rotate and translate point
                point = geometry_utils.rotate_points(point, rotation_angle)
                point = geometry_utils.translate_points(
                    point, translation_vector
                )

                yield point

    @classmethod
    def generate_mode_boundary_list(
        cls,
        side_length: float | tuple[float, float],
        base_lattice: Iterator[npt.NDArray[Numeric]],
        tiling_type: str,
        r_lim: float,
        rotation_angle: float,
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
