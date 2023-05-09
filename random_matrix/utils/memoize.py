import atexit
import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable

import numpy as np

from random_matrix.utils.types import FloatLike


class Memoize:
    """Class for memoizing functions"""

    def __init__(
        self,
        function: Callable[..., Any],
        cache_filename: str = "",
        float_tol: float = 1e-8,
    ):
        # Load cache if user specifices filename
        if cache_filename != "" and os.path.isfile(cache_filename):
            cache = self.get_cache_from_file(cache_filename)
        else:
            cache = {}
            cache["metadata"] = {
                "function_name": function.__name__,
                "description": "",
            }
            cache["info"] = {
                "num_entries": 0,
                "cache_hits": 0,
                "function_calls": 0,
            }
            cache["data"] = {}

        self.cache = cache
        self.float_tol = float_tol
        self.function = function

        if cache_filename != "":
            atexit.register(self.save_cache, cache_filename)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.cache["info"]["function_calls"] += 1
        rounded_args = [
            self._round(arg, self.float_tol)
            if isinstance(arg, FloatLike)
            else arg
            for arg in args
        ]
        rounded_kwargs = {
            key: self._round(value, self.float_tol)
            if isinstance(value, FloatLike)
            else value
            for key, value in kwargs.items()
        }
        params_hash = self._get_hash(*rounded_args, **rounded_kwargs)
        val = self.cache["data"].get(params_hash, None)
        if val is not None:
            self.cache["info"]["cache_hits"] += 1
            return val
        else:
            output = self.function(*args, **kwargs)
            self.cache["data"][params_hash] = output
            self.cache["info"]["num_entries"] += 1
            return output

    @staticmethod
    def _get_hash(*args: Any, **kwargs: Any) -> str:
        arg_tuple = tuple(args)
        kwargs_tuple = tuple(kwargs.items())
        combined_tuple = (arg_tuple, kwargs_tuple)
        hash_value = hashlib.sha256(str(combined_tuple).encode()).hexdigest()
        return hash_value

    @staticmethod
    def _round(number: FloatLike, float_tol: float) -> FloatLike:
        rounded = np.round(number / float_tol) * float_tol
        return rounded

    @staticmethod
    def _validate_cache_filename(cache_filename: str) -> None:
        if not isinstance(cache_filename, str):
            raise ValueError("Please provide cache_filename as a string.")
        if cache_filename == "":
            raise ValueError("cache_filename cannot be an empty string.")

    def get_cache_from_file(self, cache_filename: str) -> Any:
        self._validate_cache_filename(cache_filename)
        try:
            with open(cache_filename, "r") as cache_file:
                return json.load(cache_file)
        except FileNotFoundError:
            print(f"Filename: '{cache_filename}' does not exist.")

    def save_cache(self, cache_filename: str) -> None:
        self._validate_cache_filename(cache_filename)
        with open(cache_filename, "w") as cache_file:
            json.dump(self.cache, cache_file)


def memoize(
    cache_filename: str = "", float_tol: float = 1e-8
) -> Callable[..., Any]:
    def memoize_decorator(function: Callable[..., Any]) -> Memoize:
        return Memoize(
            function=function,
            cache_filename=cache_filename,
            float_tol=float_tol,
        )

    return memoize_decorator
