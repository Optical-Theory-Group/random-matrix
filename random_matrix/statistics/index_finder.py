import numpy as np
import multiprocess as mp
import shapely
import os
import functools
from random_matrix.modes.mode_grid import ModeGrid
from random_matrix.statistics.medium_parameters import MediumParameters
from random_matrix.utils import geometry_utils, array_utils


class IndexFinder:
    def __init__(
        self, mode_grid: ModeGrid, medium_parameters: MediumParameters
    ) -> None:
        self.mode_grid = mode_grid

        # Only necessary if delta functions are later relaxed
        # but left in for now
        self.medium_parameters = medium_parameters

    def get_indices(self) -> dict[str, dict[str, set[tuple[int, int]]]]:
        # Determine the independent element indices
        self.independent_element_indices = (
            self._get_independent_element_indices()
        )

        indices = {}
        indices["mean"] = self._get_mean_indices()
        indices["covariance"] = self._get_covariance_indices()
        # indices["pseudo_covariance"] = self._get_pseudo_covariance_indices()
        return indices

    # -------------------------------------------------------------------------
    # Methods for finding the independent elements (based on reciprocity)
    # -------------------------------------------------------------------------

    def _get_independent_element_indices(
        self,
    ) -> dict[str, dict[str, set[tuple[int, int]]]]:
        """Return a nested dictionary ultimately containing lists of
        independent elements of the scattering matrix."""

        elements = {
            "pp": self._get_independent_element_indices_pp(),
            "pe": {},
            "ep": {},
            "ee": {},
        }
        return elements  # type:ignore

    def _get_independent_element_indices_pp(
        self,
    ) -> dict[str, set[tuple[int, int]]]:
        """Get the independent elements of the propagating-propagating block
        of the scattering matrix."""

        t_elements = []
        t2_elements = []
        r_elements = []
        r2_elements = []

        indices = self.mode_grid.propagating_indices
        for i in indices:
            for j in indices:
                t_elements.append((j, i))

                if not self.mode_grid.is_reciprocal:
                    t2_elements.append((j, i))
                    r_elements.append((j, i))
                    r2_elements.append((j, i))
                else:
                    # Only accept elements above or on the anti-diagonal
                    # No elements of t2 are accepted
                    if i + j <= 0:
                        r_elements.append((j, i))
                        r2_elements.append((j, i))

        elements = {
            "t": t_elements,
            "r": r_elements,
            "t2": t2_elements,
            "r2": r2_elements,
        }
        return elements

    # -------------------------------------------------------------------------
    # Methods for finding the indices where the mean is non-zero
    # -------------------------------------------------------------------------

    def _get_mean_indices(self) -> dict[str, set[tuple[int, int]]]:
        """Get indices of the scattering matrix for which the mean needs to
        be calculated."""

        mean_indices = {
            "pp": self._get_mean_indices_pp(),
            "pe": {},
            "ep": {},
            "ee": {},
        }
        return mean_indices  # type:ignore

    def _get_mean_indices_pp(self) -> dict[str, set[tuple[int, int]]]:
        """Get indices of 2x2 blocks in the pp section of the scattering
        matrix for which the mean needs to be calculated.

        Currently only uses a delta function implementation.
        """
        t_elements = []
        t2_elements = []
        r_elements = []
        r2_elements = []

        indices = self.mode_grid.propagating_indices
        independent_elements = self.independent_element_indices["pp"]

        for index in indices:
            new_indices = (index, index)
            if new_indices in independent_elements["t"]:
                t_elements.append(new_indices)
            if new_indices in independent_elements["r"]:
                r_elements.append(new_indices)
            if new_indices in independent_elements["t2"]:
                t2_elements.append(new_indices)
            if new_indices in independent_elements["r2"]:
                r2_elements.append(new_indices)

        mean_indices = {
            "t": t_elements,
            "r": r_elements,
            "t2": t2_elements,
            "r2": r2_elements,
        }
        return mean_indices

    # -------------------------------------------------------------------------
    # Methods for finding the indices where the covariance is non-zero
    # -------------------------------------------------------------------------

    def _get_covariance_indices(
        self,
    ) -> dict[str, dict[str, set[tuple[int, int, int, int]]]]:
        """Get indices of the scattering matrix for which the covariance needs
        tobe calculated."""

        covariance_indices = {
            "pp,pp": self._get_covariance_indices_pppp(),
            "pp,pe": {},
            "pp,ep": {},
            "pp,ee": {},
            "pe,pe": {},
            "pe,ep": {},
            "pe,ee": {},
            "ep,ep": {},
            "ep,ee": {},
            "ee,ee": {},
        }
        return covariance_indices  # type: ignore

    def _get_covariance_indices_pppp(
        self,
    ) -> dict[str, set[tuple[int, int, int, int]]]:
        """Get indices of 2x2 blocks in the pp section of the scattering
        matrix for which the mean needs to be calculated.

        Currently only uses a delta function implementation.
        """
        covariance_indices = {
            "t,t": set(),
            "t,r": set(),
            "t,t2": set(),
            "t,r2": set(),
            "r,r": set(),
            "r,t2": set(),
            "r,r2": set(),
            "t2,t2": set(),
            "t2,r2": set(),
            "r2,r2": set(),
        }

        indices = self.mode_grid.propagating_indices
        num_indices = len(indices)
        independent_elements = self.independent_element_indices["pp"]

        elements = [
            (index_i * num_indices + index_j, i, j)
            for index_i, i in enumerate(indices)
            for index_j, j in enumerate(indices)
        ]
        num_elements = len(elements)

        # Multiprocessing parameters
        num_processes = min(num_elements, os.cpu_count())
        print(f"Number of processes: {num_processes}")

        parallelised_function = functools.partial(
            self._get_covariance_indices_pppp_partial,
            mode_grid=self.mode_grid,
            independent_elements=independent_elements,
            elements=elements,
        )

        partial_elements = array_utils.split_list(elements, num_processes)

        with mp.Pool(processes=num_processes) as pool:
            out = pool.map(parallelised_function, partial_elements)

        for new_covariance_indices in out:
            for key in covariance_indices:
                covariance_indices[key].update(new_covariance_indices[key])

        return covariance_indices

    @staticmethod
    def _get_covariance_indices_pppp_partial(
        partial_elements,
        mode_grid,
        independent_elements,
        elements,
    ):
        covariance_indices = {
            "t,t": set(),
            "t,r": set(),
            "t,t2": set(),
            "t,r2": set(),
            "r,r": set(),
            "r,t2": set(),
            "r,r2": set(),
            "t2,t2": set(),
            "t2,r2": set(),
            "r2,r2": set(),
        }

        # Area used to determine whether memory effect type correlations are
        # kept or not
        threshold_area = mode_grid.by_index(mode_grid.central_index).weight * 2

        for index_one, i, j in partial_elements:
            # Second loop starts form index one. We can ignore half of
            # the cases, because they're just the same thing with the order
            # of the shapes swapped.
            for index_two, u, v in elements[index_one:]:
                # Boolean flag for autocorrelations
                # If we do have autocorrelations, we need to add the indices
                # because these define the intensity statistics within a block
                # The next code block is skipped if this is true
                # print(i, j, u, v)
                is_autocorrelation = i == u and j == v

                if not is_autocorrelation:
                    # Here we are looking at off-diagonal terms of the
                    # correlation matrix. These will be memory effect type
                    # correlations.

                    mode_i = mode_grid.by_index(i).vertices
                    mode_j = mode_grid.by_index(j).vertices
                    mode_u = mode_grid.by_index(u).vertices
                    mode_v = mode_grid.by_index(v).vertices

                    # Means
                    mean_i = np.mean(mode_i, axis=0)
                    mean_j = np.mean(mode_j, axis=0)
                    centre_ij = np.mean(np.vstack((mean_i, mean_j)), axis=0)
                    mean_u = np.mean(mode_u, axis=0)
                    mean_v = np.mean(mode_v, axis=0)
                    centre_uv = np.mean(np.vstack((mean_u, mean_v)), axis=0)

                    mode_j_ref = geometry_utils.reflect_through_point(
                        mode_j, centre_ij
                    )
                    ij_intersect = geometry_utils.intersection(
                        mode_i, mode_j_ref
                    )
                    new_ij = 2 * geometry_utils.translate_points(
                        ij_intersect, -centre_ij
                    )

                    mode_v_ref = geometry_utils.reflect_through_point(
                        mode_v, centre_uv
                    )
                    uv_intersect = geometry_utils.intersection(
                        mode_u, mode_v_ref
                    )
                    new_uv = 2 * geometry_utils.translate_points(
                        uv_intersect, -centre_uv
                    )

                    ijuv_intersect = geometry_utils.intersection(
                        new_ij, new_uv
                    )
                    area = geometry_utils.get_polygon_area(ijuv_intersect)

                    # If the area is too small, we ignore these effects
                    if (
                        np.isclose(area, threshold_area)
                        or area < threshold_area
                    ):
                        continue

                # If we reach this point, we want to add the indices
                # to the eventual list of tasks
                covariance_indices["t,t"].add((i, j, u, v))
                if (u, v) in independent_elements["r"]:
                    covariance_indices["t,r"].add((i, j, u, v))
                if (u, v) in independent_elements["t2"]:
                    covariance_indices["t,t2"].add((i, j, u, v))
                if (u, v) in independent_elements["r2"]:
                    covariance_indices["t,r2"].add((i, j, u, v))
                if (i, j) in independent_elements["r"]:
                    if (u, v) in independent_elements["r"]:
                        covariance_indices["r,r"].add((i, j, u, v))
                    if (u, v) in independent_elements["t2"]:
                        covariance_indices["r,t2"].add((i, j, u, v))
                    if (u, v) in independent_elements["r2"]:
                        covariance_indices["r,r2"].add((i, j, u, v))
                if (i, j) in independent_elements["t2"]:
                    if (u, v) in independent_elements["t2"]:
                        covariance_indices["t2,t2"].add((i, j, u, v))
                    if (u, v) in independent_elements["r2"]:
                        covariance_indices["t2,r2"].add((i, j, u, v))
                if (i, j) in independent_elements["r2"]:
                    if (u, v) in independent_elements["r2"]:
                        covariance_indices["r2,r2"].add((i, j, u, v))

        return covariance_indices

    def _get_covariance_block_indices(
        self, block_one: str, block_two: str
    ) -> set[tuple[int, int, int, int]]:
        elements = set()

        for index_one in range(
            len(self.independent_element_indices["pp"][block_one])
        ):
            print(index_one)
            # Pick out the first elements
            i, j = list(self.independent_element_indices["pp"][block_one])[
                index_one
            ]

            # If we are looping over the same block, we can avoid cases where
            # we correlate the same pairs of elements but in the reverse order
            starting_point = index_one if block_one == block_two else 0

            for index_two in range(
                starting_point,
                len(self.independent_element_indices["pp"][block_two]),
            ):
                u, v = list(self.independent_element_indices["pp"][block_two])[
                    index_two
                ]

                if i == u and j == v:
                    elements.add((i, j, u, v))

        return elements

    # -------------------------------------------------------------------------
    # Methods for finding the indices where the pseudo-covariance is non-zero
    # -------------------------------------------------------------------------

    def _get_pseudo_covariance_indices(
        self,
    ) -> dict[str, dict[str, set[tuple[int, int, int, int]]]]:
        """Get indices of the scattering matrix for which the pseudo-covariance
        needs tobe calculated."""

        pseudo_covariance_indices = {
            "pp,pp": self._get_pseudo_covariance_indices_pppp(),
            "pp,pe": {},
            "pp,ep": {},
            "pp,ee": {},
            "pe,pe": {},
            "pe,ep": {},
            "pe,ee": {},
            "ep,ep": {},
            "ep,ee": {},
            "ee,ee": {},
        }
        return pseudo_covariance_indices

    def _get_pseudo_covariance_indices_pppp(
        self,
    ) -> dict[str, set[tuple[int, int, int, int]]]:
        return {}

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------

    @property
    def num_independent_elements(self):
        independent_elements = self.independent_element_indices
        independent_elements = independent_elements["pp"]
        total = (
            len(independent_elements["t"])
            + len(independent_elements["r"])
            + len(independent_elements["t2"])
            + len(independent_elements["r2"])
        )
        return total
