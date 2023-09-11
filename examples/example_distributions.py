from typing import Any

import numpy as np

from random_matrix.input_statistics import density_function, density_integrals


# Function to be integrated
def f(x: float, y: float, a: float, b: float) -> float:
    return np.sin(x - y) * a * b * (1 - b)


print("f(x,y,a,b) = sin(x-y) * a * b(1-b)\n")
# -----------------------------------------------------------------------------
# Example 1) Dirac delta function
# -----------------------------------------------------------------------------
print("------------")
print("Example one\n")
print("p(a,b) = delta(a - 1) * delta(b - 0.5)\n")
# p(a,b) = delta(a-1) * delta(b-0.5)

dirac_density = density_function.DensityFunction.from_delta({"a": 1.0, "b": 0.5})

g = density_integrals.integrate_by_density(f, dirac_density)

x, y = np.pi, np.pi / 2
print(f"f({x:.2f}, {y:.2f}, {1.0:.2f}, {0.5:.2f}) = {f(x,y,1.0,0.5):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x, y):.4f}")

x, y = np.random.randn(2)
print(f"f({x:.2f}, {y:.2f}, {1.0:.2f}, {0.5:.2f}) = {f(x,y,1.0,0.5):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x,y):.4f}")

# -----------------------------------------------------------------------------
# Example 2) Regular pdf
# -----------------------------------------------------------------------------
print("\n------------")
print("Example two\n")
print("p(a,b) = 6ab^2\n")


def p(a: float, b: float) -> float:
    return 2 * a * 3 * b**2


domain = {"a": [0.0, 1.0], "b": [0.0, 1.0]}

regular_density = density_function.DensityFunction.from_regular(p, domain)
g = density_integrals.integrate_by_density(f, regular_density)


# Wolfram alpha says the answer is
# 1/10 sin(x-y)
def expected(x: float, y: float) -> float:
    return 0.1 * np.sin(x - y)


x, y = np.pi, np.pi / 2
print(f"expected({x:.2f}, {y:.2f}) = {expected(x,y):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x, y):.4f}")

x, y = np.random.randn(2)
print(f"expected({x:.2f}, {y:.2f}) = {expected(x,y):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x, y):.4f}")

# -----------------------------------------------------------------------------
# Example 3) Complicated pdf
# -----------------------------------------------------------------------------
print("\n------------")
print("Example three\n")
print(
    "p(a,b) = \n"
    "0.3 * delta(a - 1) * delta(b - 0.5) +\n"
    "0.3 * 6ab^2 +\n"
    "0.4 * delta(a - 0.4) * 2b\n"
)

# p(a,b) =
# 0.3* d(a-0.5) * d(b-0.5) +
# 0.3 * 6ab^2 +
# 0.4 * d(a-0.4) * 2b

# ----------
# Term 1
# ----------

term_one = density_function.DensityFunctionTerm.from_delta({"a": 0.5, "b": 0.5}, 0.3)

# ----------
# Term 2
# ----------


def p2(a: float, b: float) -> float:
    return 1.8 * a * b * b


domain = {"a": [0.0, 1.0], "b": [0.0, 1.0]}

term_two = density_function.DensityFunctionTerm.from_regular(p2, domain)

# ----------
# Term 3
# ----------


def p3(b: float) -> float:
    return 2 * b


domain = {"b": [0.0, 1.0]}

regular_factor = density_function.RegularDensityFactor(p3, domain)
delta_factor = density_function.DeltaDensityFactor({"a": 0.4}, 0.4)

term_three = density_function.DensityFunctionTerm(regular_factor, delta_factor)

# ----------
# Full density function
# ----------

density = density_function.DensityFunction([term_one, term_two, term_three])

g = density_integrals.integrate_by_density(f, density)


def expected(x: float, y: float) -> float:
    return 113 / 1200 * np.sin(x - y)


x, y = np.pi, np.pi / 2
print(f"expected({x:.2f}, {y:.2f}) = {expected(x,y):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x, y):.4f}")

x, y = np.random.randn(2)
print(f"expected({x:.2f}, {y:.2f}) = {expected(x,y):.4f}")
print(f"g({x:.2f},{y:.2f}) = {g(x, y):.4f}")
