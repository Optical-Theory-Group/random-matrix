from typing import Protocol

import numpy as np
import numpy.typing as npt


class AmplitudeMatrix(Protocol):
    def __call__(
        self,
        k_inc: npt.NDArray[np.float64 | np.complex128],
        k_sca: npt.NDArray[np.float64 | np.complex128],
        params: dict[str, np.float64 | np.complex128],
    ) -> npt.NDArray[np.complex128]:
        ...
