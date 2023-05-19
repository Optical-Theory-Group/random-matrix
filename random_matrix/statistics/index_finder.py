from random_matrix.modes.mode_grid import ModeGrid
from random_matrix.statistics.medium_parameters import MediumParameters


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
        indices["pseudo_covariance"] = self._get_pseudo_covariance_indices()
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

        t_elements = set()
        t2_elements = set()
        r_elements = set()
        r2_elements = set()

        indices = self.mode_grid.propagating_indices
        for i in indices:
            for j in indices:
                t_elements.add((j, i))

                if not self.mode_grid.is_reciprocal:
                    t2_elements.add((j, i))
                    r_elements.add((j, i))
                    r2_elements.add((j, i))
                else:
                    # Only accept elements above or on the anti-diagonal
                    # No elements of t2 are accepted
                    if i + j <= 0:
                        r_elements.add((j, i))
                        r2_elements.add((j, i))

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
        t_elements = set()
        t2_elements = set()
        r_elements = set()
        r2_elements = set()

        indices = self.mode_grid.propagating_indices
        independent_elements = self.independent_element_indices["pp"]

        for index in indices:
            new_indices = (index, index)
            if new_indices in independent_elements["t"]:
                t_elements.add(new_indices)
            if new_indices in independent_elements["r"]:
                r_elements.add(new_indices)
            if new_indices in independent_elements["t2"]:
                t2_elements.add(new_indices)
            if new_indices in independent_elements["r2"]:
                r2_elements.add(new_indices)

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
            "pppp": self._get_covariance_indices_pppp(),
            "pppe": {},
            "ppep": {},
            "ppee": {},
            "pepe": {},
            "peep": {},
            "peee": {},
            "epep": {},
            "epee": {},
            "eeee": {},
        }
        return covariance_indices  # type: ignore

    def _get_covariance_indices_pppp(
        self,
    ) -> dict[str, set[tuple[int, int, int, int]]]:
        return {}

    # -------------------------------------------------------------------------
    # Methods for finding the indices where the pseudo-covariance is non-zero
    # -------------------------------------------------------------------------

    def _get_pseudo_covariance_indices(
        self,
    ) -> dict[str, dict[str, set[tuple[int, int, int, int]]]]:
        """Get indices of the scattering matrix for which the pseudo-covariance
        needs tobe calculated."""

        pseudo_covariance_indices = {
            "pppp": self._get_pseudo_covariance_indices_pppp(),
            "pppe": {},
            "ppep": {},
            "ppee": {},
            "pepe": {},
            "peep": {},
            "peee": {},
            "epep": {},
            "epee": {},
            "eeee": {},
        }
        return pseudo_covariance_indices

    def _get_pseudo_covariance_indices_pppp(
        self,
    ) -> dict[str, set[tuple[int, int, int, int]]]:
        return {}
