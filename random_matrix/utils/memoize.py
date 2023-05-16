"""Function memoization module.

Memoization can be used either by crating a "Memoize" object, or by applying
"memoize" as a decorator to another function.
"""

import atexit
import hashlib
import json
import pathlib
from typing import Any, Callable

import numpy as np

from random_matrix.utils.types import FloatLike

float_types = (float, complex, np.float64, np.complex128, np.ndarray)


class Memoize:
    """Class for memoizing functions

    Attributes
    ----------
    function:
        The function that will be memoized.
    cache_path:
        Path object that points to the location of the cache. Can be given as
        a string. If not given, a new cache will be made at the location.
    float_atol:
        Absolute tolerance for equating float-like arguments to the function.
    float_rtol:
        Relative tolerance for equating float-like arguments to the function.
    cache:
        The cache where all the function returns are stored.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        cache_path: pathlib.Path | str,
        array_output: bool = False,
        complex_output: bool = False,
        float_atol: float = 1e-8,
        float_rtol: float = 1e-8,
    ):
        self.function = function
        self.float_atol = float_atol
        self.float_rtol = float_rtol
        self.array_output = array_output
        self.complex_output = complex_output

        # Convert cache_path to Path object if given as a string
        self.cache_path = (
            pathlib.Path(cache_path)
            if isinstance(cache_path, str)
            else cache_path
        )

        # Handle loading of the cache
        self.cache = (
            self._get_cache_from_file(self.cache_path)
            if self.cache_path.exists()
            else self._get_new_cache()
        )

        # Save cache when program finishes executing
        atexit.register(self._save_cache, self.cache_path)

    # -------------------------------------------------------------------------
    # Init methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_cache_from_file(cache_path: pathlib.Path) -> Any:
        with cache_path.open("r") as cache_file:
            return json.load(cache_file)

    def _get_new_cache(self) -> dict[Any, Any]:
        cache: dict[Any, Any] = {}
        cache["metadata"] = {
            "function_name": self.function.__name__,
            "description": "",
        }
        cache["info"] = {
            "num_entries": 0,
            "cache_hits": 0,
            "function_calls": 0,
        }
        cache["data"] = {}
        return cache

    def _save_cache(self, cache_path: pathlib.Path) -> None:
        with cache_path.open("w") as cache_file:
            json.dump(self.cache, cache_file)

    # -------------------------------------------------------------------------
    # Hashing methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_hash(*args: Any, **kwargs: Any) -> str:
        kwargs_tuple = tuple(kwargs.items())
        combined_tuple = (*args, *kwargs_tuple)
        hash_value = hashlib.sha256(str(combined_tuple).encode()).hexdigest()
        return hash_value

    def _round(self, value: FloatLike) -> FloatLike:
        rounded = np.round(value / self.float_atol) * self.float_atol
        return rounded

    # -------------------------------------------------------------------------
    # Calling the memoized function
    # -------------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.cache["info"]["function_calls"] += 1

        # Round float-like arguments
        rounded_args = [
            self._round(arg) if isinstance(arg, float_types) else arg
            for arg in args
        ]
        rounded_kwargs = {
            key: self._round(value)
            if isinstance(value, float_types)
            else value
            for key, value in kwargs.items()
        }

        params_hash = self._get_hash(*rounded_args, **rounded_kwargs)
        value = self.cache["data"].get(params_hash, None)

        # Value was found in the cache
        if value is not None:
            self.cache["info"]["cache_hits"] += 1

            # Handle different cases
            match (self.complex_output, self.array_output):
                # Complex array
                case True, True:
                    real = np.array(value[0])
                    imag = np.array(value[1])
                    return real + 1j * imag

                # Complex number
                case True, False:
                    real = value[0]
                    imag = value[1]
                    return real + 1j * imag

                # Real array
                case False, True:
                    return np.array(value)

                # Real number
                case False, False:
                    return value
        else:
            output = self.function(*args, **kwargs)

            # Handle different cases
            match (self.complex_output, self.array_output):
                # Complex array
                case True, True:
                    self.cache["data"][params_hash] = [
                        np.real(output).tolist(),
                        np.imag(output).tolist(),
                    ]

                # Complex number
                case True, False:
                    self.cache["data"][params_hash] = [
                        np.real(output),
                        np.imag(output),
                    ]

                # Real array
                case False, True:
                    self.cache["data"][params_hash] = output.tolist()

                # Real number
                case False, False:
                    self.cache["data"][params_hash] = output

            self.cache["info"]["num_entries"] += 1
            return output


# -----------------------------------------------------------------------------
# Function decorator
# -----------------------------------------------------------------------------


def memoize(
    cache_path: pathlib.Path | str,
    array_output: bool = False,
    complex_output: bool = False,
    float_atol: float = 1e-8,
    float_rtol: float = 1e-8,
) -> Callable[..., Any]:
    def inner(function: Callable[..., Any]) -> Memoize:
        return Memoize(
            function=function,
            array_output=array_output,
            complex_output=complex_output,
            cache_path=cache_path,
            float_atol=float_atol,
            float_rtol=float_rtol,
        )

    return inner
