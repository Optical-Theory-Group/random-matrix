import time

import numpy as np

from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.utils.memoize import memoize

# -----------------------------------------------------------------------------
# Example 1: Fibonacci numbers
# -----------------------------------------------------------------------------


# @memoize(
#     cache_path="/var/home/niall/Code/Science/random-matrix/data/function_cache/examples/fib.json"
# )
# def fib(n: int) -> int:
#     if n in {0, 1}:
#         return n
#     return fib(n - 1) + fib(n - 2)


# for i in range(1000):
#     print(f"fib({i}) = {fib(i)}")

# -----------------------------------------------------------------------------
# Example 2: Mixture of params
# -----------------------------------------------------------------------------


# @memoize(
#     cache_path="/var/home/niall/Code/Science/random-matrix/data/function_cache/examples/slow_function.json"
# )
# def slow_addition(x: int, y: int, sleep: bool = True) -> int:
#     if sleep:
#         time.sleep(3)
#     return x + y

# start = time.perf_counter()
# slow_addition(1, 1)
# end = time.perf_counter()
# print(f"Function ran in {end-start} seconds.")


# -----------------------------------------------------------------------------
# Example 3: Complex matrix calculations
# -----------------------------------------------------------------------------

# np.random.seed(0)
# mat_size = 10
# num_mats = 10**5
# reals = np.random.randn(num_mats, mat_size, mat_size)
# imags = np.random.randn(num_mats, mat_size, mat_size)
# matrices = reals + 1j * imags


# @memoize(
#     cache_path="/var/home/niall/Code/Science/random-matrix/data/function_cache/examples/matrix.json",
#     array_output=True,
#     complex_output=True,
# )
# def product(matrices: np.ndarray) -> np.ndarray:
#     product = np.identity(mat_size)
#     for matrix in matrices:
#         u, s, vh = np.linalg.svd(matrix)
#         unitary = u @ vh
#         product = product @ unitary
#     return product


# start = time.perf_counter()
# product(matrices)
# end = time.perf_counter()
# print(f"Function ran in {end-start} seconds.")

# -----------------------------------------------------------------------------
# T3
# -----------------------------------------------------------------------------
a_matrix = isotropic_sphere.get_A
a_matrix(np.array([0,0,1]),np.array([0,0,1]),1.0,1.2)