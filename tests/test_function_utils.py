import inspect

import numpy as np
import scipy

from random_matrix.utils import function_utils
from random_matrix.utils.types import FloatLike


# vectorizing arguments
def test_vectorize_arguments() -> None:
    def function(x: float, y: float, z: float) -> float:
        return x * y * z

    assert len(inspect.signature(function).parameters) == 3
    vectorized = function_utils.vectorize_arguments(function)
    assert len(inspect.signature(vectorized).parameters) == 1
