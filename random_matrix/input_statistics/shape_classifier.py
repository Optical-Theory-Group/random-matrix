"""Module for classifying modes based on their shapes into equivalence classes
for simplying statistical integrals"""

import functools
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.stats
import shapely

from random_matrix.input_statistics import input_statistics_logger
from random_matrix.modes.mode_grid import ModeGrid
from random_matrix.utils import array_utils, geometry_utils


@dataclass(slots=True)
class ShapeData:
    """Mini class for storing data associated with a single shape

    lengths: unsigned lengths of sides of the shape
    angles: arctan2 type angles of the sides, thought of as vectors
            (orientation taken into account)
    exterior_angles: the exterior angles of the polygon. These are not oriented
    """

    lengths: np.ndarray
    angles: np.ndarray
    exterior_angles: np.ndarray

    def __str__(self):
        return (
            f"Lengths: {self.lengths}\n"
            f"Angles: {self.angles}\n"
            f"Exterior Angles: {self.exterior_angles}"
        )


def get_shape_data(shape: np.ndarray) -> ShapeData:
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
        # Clean up edge case where -pi is returned. Make it pi for consistency
        angle = -angle if np.isclose(angle, -np.pi) else angle
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


def get_congruent_shape_rotation_angle(
    shape_data: ShapeData, reference_shape_data: ShapeData
) -> np.float64:
    """Gets the angle that the reference_shape_data would need to be rotated by
    to get the shape_data

    Angle is reduced to the inetrval [-pi, pi]
    """
    # Unpack data
    lengths = shape_data.lengths
    angles = shape_data.angles
    exterior_angles = shape_data.exterior_angles
    lengths_reference = reference_shape_data.lengths
    angles_reference = reference_shape_data.angles
    exterior_angles_reference = reference_shape_data.exterior_angles

    possible_outputs = []

    # Align so that all the sides coincide
    max_roll = len(lengths)
    for roll in (-i for i in range(max_roll)):

        # Check that the shapes are the same for this particular roll
        rolled_lengths = np.roll(lengths, roll)
        rolled_exteriors = np.roll(exterior_angles, roll)

        equal = np.allclose(lengths_reference, rolled_lengths) & np.allclose(
            exterior_angles_reference, rolled_exteriors
        )
        if equal:
            # Find rotation angle for this particular configuration
            rolled_angles = np.roll(angles, roll)

            angle_ref = angles_reference[0]
            angle_roll = rolled_angles[0]

            first = geometry_utils.get_unit_vector(angle_ref)
            second = geometry_utils.get_unit_vector(angle_roll)
            diff = geometry_utils.get_signed_small_angular_difference(
                first, second
            )
            possible_outputs.append(diff)
    min_index = np.argmin(np.abs(possible_outputs))
    angle_found = possible_outputs[min_index]
    return angle_found


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
    is_template: bool
    index: int
    template_index: int
    translation_vector: np.ndarray

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
        angles = self.shape_data.angles
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
    singles_indices: tuple[int, int, int, int]
    single_classes: tuple[int, int, int, int]
    class_number: int
    is_template: bool
    translation_vector: np.ndarray
    signs: np.ndarray
    centroids: np.ndarray
    vertices: np.ndarray | None = None

    def get_domain(
        self, mode_grid, sampling_method, points_per_simplex
    ) -> None:
        if self.is_template:
            return self.get_domain_template(
                mode_grid, sampling_method, points_per_simplex
            )
        else:
            return self.get_domain_others()


@dataclass
class ClassQuadruple:
    template: ShapeQuadruple
    index: int
    members: list[ShapeQuadruple] = field(default_factory=list)

    def add_member(self, new_shape_quadruple: ShapeQuadruple) -> None:
        """Append given quadruple to the members list"""
        self.members.append(new_shape_quadruple)

    @property
    def num_members(self) -> int:
        """Number of members in the class (note that this doesn't include
        the template)"""
        return len(self.members)

    @property
    def template_singles_indices(self) -> tuple[int, int, int, int]:
        """Get the singles_indices for the class's template"""
        return self.template.singles_indices

    @property
    def members_singles_indices(self) -> list[tuple[int, int, int, int]]:
        """Get a list of singles_indices for the class's members"""
        return [member.singles_indices for member in self.members]

    @property
    def singles_indices(self) -> list[tuple[int, int, int, int]]:
        """Get a list of singles_indices for the class's members and
        template"""
        return [self.template_singles_indices] + self.members_singles_indices

    @property
    def quadruples(self) -> list[ShapeQuadruple]:
        """List of all of the quadruples, including the template"""
        return [self.template] + self.members

    def __repr__(self):
        return f"ClassQuadruple"


@dataclass
class ClassQuadrupleList:
    classes: list[ClassQuadruple] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @property
    def num_quadruples(self) -> int:
        """Get the total number of quadruples in all of the classes"""
        return sum(1 + c.num_members for c in self.classes)

    @property
    def singles_indices(self) -> list[tuple[int, int, int, int]]:
        """List of all singles_indices for everything in all the classes"""
        return [index for c in self.classes for index in c.singles_indices]

    def get_class(
        self, singles_indices: tuple[int, int, int, int]
    ) -> ClassQuadruple:
        """Get the class containing the quadruple with given indices"""
        for c in self.classes:
            if singles_indices in c.singles_indices:
                return c
        raise ValueError(
            f"No class found containing quadruple with indices {singles_indices}"
        )

    def get_quadruple(
        self, singles_indices: tuple[int, int, int, int]
    ) -> ShapeQuadruple:
        """Return the quadruple with the given indices"""
        c = self.get_class(singles_indices)
        for m in c.members:
            if m.singles_indices == singles_indices:
                return m

    def get_template(
        self, singles_indices: tuple[int, int, int, int]
    ) -> ShapeQuadruple:
        """Get the template for a given quadruple"""
        return self.get_class(singles_indices).template

    def add_class(self, new_class: ClassQuadruple) -> None:
        """Append new class to classes"""
        self.classes.append(new_class)

    def append(self, new_class: ClassQuadruple) -> None:
        self.add_class(new_class)

    def __repr__(self):
        return f"ClassQuadrupleList"


# -------------------------------------------------------------------------
# Shape classification to reduce number of calculations invovled
# -------------------------------------------------------------------------


class ShapeClassifier:

    def __init__(
        self,
        mode_grid: ModeGrid,
        logger: input_statistics_logger.InputStatisticsLogger,
    ) -> None:
        self.mode_grid = mode_grid
        self.logger = logger

    def classify_shapes(self, quadruple_indices=None) -> ClassQuadrupleList:
        with self.logger.log("singles"):
            singles, single_templates = self._get_singles(self.mode_grid)

        self.number_single_templates = len(single_templates)

        with self.logger.log("quadruples"):
            classes = self._get_quadruples(singles, quadruple_indices)

        return classes

    def show_report(self) -> None:
        self.logger.show_report(
            self.number_single_templates,
        )

    # -------------------------------------------------------------------------
    # Mode classification methods
    # -------------------------------------------------------------------------

    def _get_singles(self, mode_grid: ModeGrid) -> dict[str, ShapeSingle]:
        modes = mode_grid.mode_list
        templates = []
        singles = {}
        next_class_index = -1

        for mode in self.logger.progress_bar(modes):
            # Calculate new shape data for comparison with the templates
            new_shape_data = get_shape_data(mode.vertices)
            lengths = new_shape_data.lengths
            angles = new_shape_data.angles
            lengths_angles = np.vstack((lengths, angles)).T

            is_congruent = False
            for matched_index, template in enumerate(templates):
                is_congruent = array_utils.is_equal_array(
                    lengths_angles, template.lengths_angles, order_matters=True
                )
                if is_congruent:
                    break

                # To do: Check congruence with mirror image

            if is_congruent:
                # We are equal to one of the templates. Build the new shape and
                # find its angle relative to the template. In addition, give it
                # the same class number as the template.
                new_class_number = matched_index
                is_template = False
                translation_vector = template.vertices[0] - mode.vertices[0]
                template_index = template.index

            if not is_congruent:
                # If we reach here, we need to make a new template
                next_class_index += 1
                new_class_number = next_class_index
                is_template = True
                translation_vector = np.array([0.0, 0.0])
                template_index = mode.index

            new_shape_single = ShapeSingle(
                vertices=mode.vertices,
                shape_data=new_shape_data,
                class_number=new_class_number,
                translation_vector=translation_vector,
                is_template=is_template,
                index=mode.index,
                template_index=template_index,
            )
            singles[str(mode.index)] = new_shape_single
            if not is_congruent:
                templates.append(new_shape_single)

        return singles, templates

    def _get_quadruples(self, singles, quadruple_indices):
        classes = ClassQuadrupleList()
        next_index = -1

        for count, (i, j, u, v) in enumerate(
            self.logger.progress_bar(quadruple_indices)
        ):
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

            # Work out the signs array for the new quadruple
            cartesian_product = geometry_utils.iterated_cartesian_product(
                [
                    first.vertices,
                    second.vertices,
                    third.vertices,
                    fourth.vertices,
                ]
            )
            hyperplane_normals = np.array(
                [[1, 0, -1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, -1, 0, 1]]
            ).T
            signs = np.sign(cartesian_product @ hyperplane_normals)

            new_centroids = np.array(
                [
                    *first.centroid,
                    *second.centroid,
                    *third.centroid,
                    *fourth.centroid,
                ]
            )

            is_new_class = True
            for matched_index, next_class in enumerate(classes.classes):
                # Check equality of the set of shapes. This filters out
                # lots of complex cases
                template = next_class.template
                are_all_congruent = (
                    new_single_classes == template.single_classes
                )
                if not are_all_congruent:
                    continue

                # In order to reach this point, it must be the case that we
                # have matched a quadruple with a template on the basis of the
                # single shape classes. Now we check the 8D shape properties
                are_same_opposite_signs = (signs == template.signs).all() or (
                    signs == -template.signs
                ).all()

                if are_same_opposite_signs:
                    # The given quadruple belongs to the same class as the
                    # current template
                    new_class_number = matched_index
                    is_new_class = False
                    translation_vector = new_centroids - template.centroids
                    break

            if is_new_class:
                # If we reach here, we need to make a new class
                next_index += 1
                new_class_number = next_index

                translation_vector = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )

                new_quadruple = ShapeQuadruple(
                    singles_indices=(i, j, u, v),
                    single_classes=new_single_classes,
                    class_number=new_class_number,
                    is_template=True,
                    translation_vector=translation_vector,
                    signs=signs,
                    centroids=new_centroids,
                    vertices=cartesian_product,
                )
                new_class = ClassQuadruple(new_quadruple, new_class_number)
                classes.add_class(new_class)

            else:
                new_quadruple = ShapeQuadruple(
                    singles_indices=(i, j, u, v),
                    single_classes=new_single_classes,
                    class_number=new_class_number,
                    is_template=False,
                    translation_vector=translation_vector,
                    signs=signs,
                    centroids=new_centroids,
                )
                next_class.add_member(new_quadruple)

        return classes
