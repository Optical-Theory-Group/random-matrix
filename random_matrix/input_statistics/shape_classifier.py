"""Module for classifying modes based on their shapes into equivalence classes
for simplying statistical integrals"""

import functools
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import scipy.stats
import shapely

from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.input_statistics_logger import (
    ShapeClassifierLogger,
)
from random_matrix.modes.mode import Mode
from random_matrix.modes.mode_grid import ModeGrid
from random_matrix.utils import array_utils, geometry_utils


@dataclass(slots=True)
class ShapeData:
    """Mini class for storing data associated with a single shape"""

    lengths: npt.NDArray
    angles: npt.NDArray
    exterior_angles: npt.NDArray


def get_shape_data(shape: npt.NDArray) -> ShapeData:
    """Method for generating ShapeData objects from arrays of vertices"""

    lengths = []
    angles = []
    exterior_angles = []

    # Find the lengths and angles of each side (thought of as a sequence of
    # vectors)
    for vertex_one, vertex_two in array_utils.get_pairs(shape, cyclic=True):
        side = vertex_two - vertex_one
        length = np.linalg.norm(side)
        lengths.append(length)
        angle = np.arctan2(side[1], side[0])
        angles.append(angle)

    lengths = np.array(lengths)
    angles = np.array(angles)

    # Get exterior angles
    for first, second in array_utils.get_pairs(angles, cyclic=True):
        exterior_angle = geometry_utils.get_small_angular_difference(
            first, second
        )
        exterior_angles.append(exterior_angle)

    exterior_angles = np.array(exterior_angles)

    data = ShapeData(
        lengths=lengths, angles=angles, exterior_angles=exterior_angles
    )

    return data


def get_angle(
    shape_data: ShapeData, reference_shape_data: ShapeData
) -> np.float64:
    lenghts = shape_data.lengths
    lengths_reference = reference_shape_data.lengths

    angles = shape_data.angles
    angles_reference = reference_shape_data.angles

    possible_outputs = []

    # Align so that all the sides coincide
    max_roll = len(lenghts)
    for roll in (-i for i in range(max_roll)):
        rolled_lengths = np.roll(lenghts, roll)
        sides_equal = np.allclose(lengths_reference, rolled_lengths)
        if sides_equal:
            # Find rotation angle for this particular configuration
            rolled_angles = np.roll(angles, roll)
            angle = np.mod(
                scipy.stats.mode(rolled_angles - angles_reference)[0][0],
                2 * np.pi,
            )
            angle = angle - 2 * np.pi if angle > np.pi else angle
            possible_outputs.append(angle)
    min_index = np.argmin(np.abs(possible_outputs))
    return possible_outputs[min_index]


# -----------------------------------------------------------------------------
# Shape classes used for optimising the construction of the integration domains
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ShapeSingle:
    """Mode that has been classified according to its geometric properties.

    Attributes
    ----------

    """

    vertices: npt.NDArray
    shape_data: ShapeData
    class_number: int
    mirror_type: int
    angle: float
    is_template: bool
    index: int

    @property
    def shape_data_inverted(self) -> ShapeData:
        """self.shape_data, but with all quantities going backwards"""

        inverted_lengths = self.shape_data.lengths[::-1]
        inverted_angles = self.shape_data.angles[::-1]
        inverted_exterior_angles = self.shape_data.exterior_angles[::-1]
        return ShapeData(
            lengths=inverted_lengths,
            angles=inverted_angles,
            exterior_angles=inverted_exterior_angles,
        )

    @property
    def lengths_angles(self) -> npt.NDArray:
        """Array containing the lenghts and angles"""

        lengths = self.shape_data.lengths
        angles = self.shape_data.exterior_angles
        lengths_angles = np.vstack((lengths, angles)).T
        return lengths_angles

    @property
    def lengths_angles_inverted(self) -> npt.NDArray:
        lengths_angles = self.lengths_angles
        lengths_angles_inverted = np.copy(lengths_angles[::-1])
        angles = lengths_angles_inverted[:, 1]
        rolled = np.roll(angles, -1)
        lengths_angles_inverted[:, 1] = rolled
        return lengths_angles_inverted

    @property
    def centroid(self) -> npt.NDArray:
        return np.mean(self.vertices, axis=0)

    @property
    def shapely(self) -> shapely.Polygon:
        return shapely.Polygon(self.vertices)


@dataclass
class ShapeQuadruple:
    singles: tuple[int, int, int, int]
    single_classes: tuple[int, int, int, int]
    class_number: int
    mirror_types: tuple[int, int, int, int]
    angles: npt.NDArray
    signed_angle: np.float128
    connection_angles: npt.NDArray
    connection_lengths: npt.NDArray
    is_template: bool
    domain = None

    def get_domain(self, mode_grid) -> None:
        if self.is_template:
            return self.get_domain_template(mode_grid)
        else:
            return self.get_domain_others()

    # -------------------------------------------------------------------------
    # Find base domain for template
    # -------------------------------------------------------------------------

    @staticmethod
    def raise_base_domain(
        base_domain: npt.NDArray,
        mode_i: npt.NDArray,
        mode_j: npt.NDArray,
        mode_u: npt.NDArray,
        mode_v: npt.NDArray,
    ) -> npt.NDArray:
        """Given points in the 4D base domain, calculate their extensions in the
        final two dimensions."""

        output = np.zeros((0, 6))
        for pp, row in enumerate(base_domain):
            p_ij = row[0:2]
            p_uv = row[2:4]

            # Differences
            mode_j_ref = geometry_utils.reflect_through_point(mode_j, p_ij)
            ij_intersect = geometry_utils.intersection(mode_i, mode_j_ref)
            new_ij = 2 * geometry_utils.translate_points(ij_intersect, -p_ij)
            new_ij = array_utils.remove_duplicate_points(new_ij)

            mode_v_ref = geometry_utils.reflect_through_point(mode_v, p_uv)
            uv_intersect = geometry_utils.intersection(mode_u, mode_v_ref)
            new_uv = 2 * geometry_utils.translate_points(uv_intersect, -p_uv)
            new_uv = array_utils.remove_duplicate_points(new_uv)

            ijuv_intersect = geometry_utils.intersection(new_ij, new_uv)

            if np.ndim(ijuv_intersect) == 0:
                ijuv_intersect = np.array([ijuv_intersect])
            if np.ndim(ijuv_intersect) == 1:
                ijuv_intersect = ijuv_intersect[np.newaxis, :]

            # print(new_ij)
            # print(new_uv)
            # print(ijuv_intersect, flush=True)
            # print(len(ijuv_intersect[0]) == 0, flush=True)
            # print("-----------", flush=True)
            # Check for special case of empty array
            if len(ijuv_intersect[0]) == 0:
                continue

            ijuv_intersect = array_utils.remove_duplicate_points(
                ijuv_intersect
            )

            if np.ndim(ijuv_intersect) == 1:
                ijuv_intersect = ijuv_intersect[np.newaxis, :]

            repeated_row = np.tile(row, (len(ijuv_intersect), 1))
            new_contribution = np.hstack((repeated_row, ijuv_intersect))
            output = np.vstack((output, new_contribution))
        return output

    def get_domain_template(
        self, mode_grid, sampling_method="centroid"
    ) -> None:
        """Gets the integration domain associated with a template quadruple"""

        i, j, u, v = self.singles
        mode_i = mode_grid.by_index(i).vertices
        mode_j = mode_grid.by_index(j).vertices
        mode_u = mode_grid.by_index(u).vertices
        mode_v = mode_grid.by_index(v).vertices

        # Build the base 4D space
        mean_ij = (geometry_utils.minkowski_sum(mode_i, mode_j)) / 2
        mean_uv = (geometry_utils.minkowski_sum(mode_u, mode_v)) / 2

        # Add mid points
        for first, second in array_utils.get_pairs(mean_ij, cyclic=True):
            mid = (first + second) / 2
            mean_ij = np.vstack((mean_ij, mid))
        for first, second in array_utils.get_pairs(mean_uv, cyclic=True):
            mid = (first + second) / 2
            mean_uv = np.vstack((mean_uv, mid))

        mean_ij = geometry_utils.order_points(mean_ij)
        mean_uv = geometry_utils.order_points(mean_uv)
        base_domain = geometry_utils.cartesian_product(mean_ij, mean_uv)

        # Add internal points according to sampling method variable
        match sampling_method:
            case "simplex":
                # Sample interior points by taking the centroids of a delaunay
                # simplex decomposition
                interior_delaunay = scipy.spatial.Delaunay(base_domain)
                interior_simplices = base_domain[interior_delaunay.simplices]
                interior_points = np.mean(interior_simplices, axis=1)

            case _:
                interior_points = np.mean(base_domain, axis=0)

        base_domain = np.vstack((base_domain, interior_points))
        integration_tower = self.raise_base_domain(
            base_domain, mode_i, mode_j, mode_u, mode_v
        )

        # Get the boundary points of the convex hull of the resultant 6
        # dimensional shape. Finally, perform a simplex decomposition so the
        # domain is ready for integration.
        # Note: The QJ option randomly jiggles points until the shape is
        # not degenerate. Helps fix some annoying bugs. Ideally, the integrals
        # associated with this shapes will be tiny anyway, so errors shouldn't
        # accumulate too much
        try:
            hull = scipy.spatial.ConvexHull(integration_tower)
        except scipy.spatial.QhullError:
            hull = scipy.spatial.ConvexHull(
                integration_tower, qhull_options="QJ"
            )

        boundary = integration_tower[hull.vertices]

        try:
            delaunay = scipy.spatial.Delaunay(boundary)
        except scipy.spatial.QhullError:
            delaunay = scipy.spatial.Delaunay(boundary, qhull_options="QJ")

        new_simplices = boundary[delaunay.simplices]
        return new_simplices

    def get_domain_others(self):
        pass


# -------------------------------------------------------------------------
# Shape classification to reduce number of calculations invovled
# -------------------------------------------------------------------------


class ShapeClassifier:
    def __init__(self, mode_grid, logger=ShapeClassifierLogger()):
        self.mode_grid = mode_grid
        self.logger = logger

    def classify_shapes(self, quadruple_indices):
        with self.logger.log("singles"):
            singles, single_templates = self._get_singles(self.mode_grid)

        self.number_single_templates = len(single_templates)

        with self.logger.log("quadruples"):
            quadruples, quadruple_templates = self._get_quadruples(
                singles, quadruple_indices
            )

        self.number_quadruple_templates = len(quadruple_templates)

        return quadruples, quadruple_templates, singles

    def get_domains(self, quadruples, quadruple_templates, singles) -> None:
        with self.logger.log("templates"):
            templates_domain = self._get_domains_templates(quadruple_templates)

        with self.logger.log("others"):
            others_domain = self._get_domains_others(
                quadruples, templates_domain, singles
            )

        quadruples = [q for _, q in templates_domain.items()] + others_domain

        return quadruples

    def show_report(self) -> None:
        self.logger.show_report(
            self.number_single_templates, self.number_quadruple_templates
        )

    # -------------------------------------------------------------------------
    # Mode classification methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_singles(mode_grid: ModeGrid) -> dict[str, ShapeSingle]:
        modes = mode_grid.mode_list
        templates = []
        singles = {}
        next_index = -1

        for mode in modes:
            # Calculate new shape data for comparison with the templates
            new_shape_data = get_shape_data(mode.vertices)
            lengths = new_shape_data.lengths
            angles = new_shape_data.exterior_angles
            lengths_angles = np.vstack((lengths, angles)).T

            # Check congruence of the new shape with templates
            is_congruent = False
            is_congruent_inverted = False

            for matched_index, template in enumerate(templates):
                is_congruent = array_utils.is_equal_cyclic(
                    lengths_angles, template.lengths_angles
                )
                if is_congruent:
                    break

                # # Check if a polygon is congruent to the mirror image of
                # # the template
                # is_congruent = array_utils.is_equal_cyclic(
                #     lengths_angles, template.lengths_angles_inverted
                # )
                # if is_congruent:
                #     is_congruent_inverted = True
                #     break

            if is_congruent:
                # We are equal to one of the templates. Build the new shape and
                # find its angle relative to the template. In addition, give it
                # the same class number as the template.
                new_class_number = matched_index
                is_template = False

                mirror_type = (
                    template.mirror_type
                    if not is_congruent_inverted
                    else -template.mirror_type
                )
                new_angle = (
                    get_angle(new_shape_data, template.shape_data)
                    if not is_congruent_inverted
                    else get_angle(
                        new_shape_data, template.shape_data_inverted
                    )
                )
                if not np.isclose(new_angle, 0.0):
                    is_congruent = False

            if not is_congruent:
                # If we reach here, we need to make a new template
                next_index += 1
                new_class_number = next_index
                is_template = True

                inverted = np.copy(lengths_angles[::-1])
                rolled_col = np.roll(inverted[:, 1], -1)
                inverted[:, 1] = rolled_col

                is_mirror_symmetric = array_utils.is_equal_cyclic(
                    lengths_angles, inverted
                )
                mirror_type = 0 if is_mirror_symmetric else 1
                new_angle = 0.0

            new_shape_single = ShapeSingle(
                vertices=mode.vertices,
                shape_data=new_shape_data,
                class_number=new_class_number,
                mirror_type=mirror_type,
                angle=new_angle,
                is_template=is_template,
                index=mode.index,
            )
            singles[str(mode.index)] = new_shape_single
            if not is_congruent:
                templates.append(new_shape_single)

        return singles, templates

    @staticmethod
    def _get_quadruples(singles, quadruple_indices):
        templates = []
        quadruples = []
        next_index = -1

        for count, (i, j, u, v) in enumerate(quadruple_indices):
            # print(count)
            # Get shapes
            first = singles[str(i)]
            second = singles[str(j)]
            third = singles[str(u)]
            fourth = singles[str(v)]

            # Compare equality with class numbers
            new_single_classes = (
                first.class_number,
                second.class_number,
                third.class_number,
                fourth.class_number,
            )

            # Mirror type
            new_mirror_type = (
                first.mirror_type,
                second.mirror_type,
                third.mirror_type,
                fourth.mirror_type,
            )
            new_mirror_type_inverted = tuple(-x for x in new_mirror_type)

            # Orientations
            new_angles = np.array(
                [
                    first.angle,
                    second.angle,
                    third.angle,
                    fourth.angle,
                ]
            )

            # Find the connections between shapes one,two and three,four
            first_connection = second.centroid - first.centroid
            second_connection = fourth.centroid - third.centroid
            first_connection_length = np.linalg.norm(first_connection)
            second_connection_length = np.linalg.norm(second_connection)
            new_connection_lengths = np.array(
                [first_connection_length, second_connection_length]
            )
            first_connection_angle = geometry_utils.get_circular_angle(
                first_connection
            )
            second_connection_angle = geometry_utils.get_circular_angle(
                second_connection
            )
            new_connection_angles = np.array(
                [first_connection_angle, second_connection_angle]
            )

            # Angle between connections
            new_signed_angle = (
                geometry_utils.get_signed_small_angular_difference(
                    first_connection, second_connection
                )
            )

            # Check congruence of the new shape with templates
            is_congruent = False

            for matched_index, template in enumerate(templates):
                # 1) Compare classes
                is_same_class = new_single_classes == template.single_classes

                # # 2) Compare mirror symmetry
                # is_equal_mirror_types = (
                #     new_mirror_type == template.mirror_types
                # )
                # is_opposite_mirror_types = (
                #     new_mirror_type_inverted == template.mirror_types
                # )
                # is_same_mirror_type = is_equal_mirror_types

                # # 3 ) Shape orientation angles are all the same
                # orientation_diff = new_angles - template.angles
                # is_same_orientation = np.all(
                #     np.isclose(orientation_diff, orientation_diff[0])
                # )

                # 2) Connection angle
                angle_diff = new_connection_angles - template.connection_angles
                is_same_connection_angle = np.allclose(angle_diff, 0.0)

                # 3) Connection lengths
                length_diff = (
                    template.connection_lengths - new_connection_lengths
                )
                is_same_length_diff = np.allclose(length_diff, 0.0)

                is_congruent = (
                    is_same_class
                    and is_same_connection_angle
                    and is_same_length_diff
                )

                if is_congruent:
                    break

            if is_congruent:
                # We are equal to one of the templates. Build the new shape
                new_class_number = matched_index
                is_template = False

            else:
                # If we reach here, we need to make a new template
                next_index += 1
                new_class_number = next_index
                is_template = True

            new_quadruple = ShapeQuadruple(
                singles=(i, j, u, v),
                single_classes=new_single_classes,
                class_number=new_class_number,
                mirror_types=new_mirror_type,
                angles=new_angles,
                signed_angle=new_signed_angle,
                connection_angles=new_connection_angles,
                connection_lengths=new_connection_lengths,
                is_template=is_template,
            )

            quadruples.append(new_quadruple)
            if not is_congruent:
                templates.append(new_quadruple)

        return quadruples, templates

    # -------------------------------------------------------------------------
    # Domain calculation methods
    # -------------------------------------------------------------------------

    def _get_domains_templates(self, quadruple_templates) -> None:
        """Get domain templates, paralelllised over multiple cores"""

        # Multiprocessing parameters
        num_templates = len(quadruple_templates)

        num_processes = min(num_templates, os.cpu_count())
        parallelised_function = functools.partial(
            self._get_domains_templates_partial,
            mode_grid=self.mode_grid,
        )
        partial_templates = array_utils.split_list(
            quadruple_templates, num_processes
        )

        with mp.Pool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_templates)

        templates_domain = {}
        for dic in out:
            templates_domain.update(dic)

        return templates_domain

    @staticmethod
    def _get_domains_templates_partial(quadruple_templates, mode_grid):
        """Get template domains as internal attribute"""

        templates_domain = {}
        for template in quadruple_templates:
            new_domain = template.get_domain(mode_grid)
            template.domain = new_domain
            templates_domain[str(template.class_number)] = template

        return templates_domain

    def _get_domains_others(
        self, quadruples, quadruple_templates, singles
    ) -> None:
        return self._get_domains_others_partial(
            quadruples, quadruple_templates, singles
        )

    def _get_domains_others_partial(
        self, quadruples, quadruple_templates, singles
    ) -> None:
        """Get quadruples with domains for quadruples that are not templates"""

        others_domain = []
        for quad in quadruples:
            # Skip over templates, since these are already done
            if quad.is_template:
                continue

            new_class_number = quad.class_number
            template = quadruple_templates[str(new_class_number)]

            i, j, u, v = quad.singles
            q_i = singles[str(i)].centroid
            q_j = singles[str(j)].centroid
            q_u = singles[str(u)].centroid
            q_v = singles[str(v)].centroid
            q_ij = (q_i + q_j) / 2
            q_uv = (q_u + q_v) / 2

            i, j, u, v = template.singles
            t_i = singles[str(i)].centroid
            t_j = singles[str(j)].centroid
            t_u = singles[str(u)].centroid
            t_v = singles[str(v)].centroid
            t_ij = (t_i + t_j) / 2
            t_uv = (t_u + t_v) / 2

            new_domain = np.copy(template.domain)
            new_domain[:, :, 0:2] = new_domain[:, :, 0:2] - t_ij + q_ij
            new_domain[:, :, 2:4] = new_domain[:, :, 2:4] - t_uv + q_uv
            quad.domain = new_domain
            others_domain.append(quad)

        return others_domain
