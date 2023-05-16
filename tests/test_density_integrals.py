import inspect

import numpy as np

from random_matrix.statistics import density_function, density_integrals


def test_integrate_by_delta_density_factor_pass() -> None:
    """Cases where all the variables are valid"""

    def function(x, y, z):
        return np.sin(x) * np.exp(y) * z**2

    # 1 variable

    delta_one = {"x": 1.0}
    delta_dist_one = density_function.DeltaDensityFactor(delta_one)
    integrated_function_one = (
        density_integrals.integrate_by_delta_density_factor(
            function, delta_dist_one
        )
    )

    random_yz = np.random.randn(100, 2)
    for y, z in random_yz:
        assert np.isclose(integrated_function_one(y, z), function(1.0, y, z))

    # 2 variables

    delta_two = {"z": 3.0, "y": 1.2}
    delta_dist_two = density_function.DeltaDensityFactor(delta_two)
    integrated_function_two = (
        density_integrals.integrate_by_delta_density_factor(
            function, delta_dist_two
        )
    )

    random_x = np.random.randn(100)
    for x in random_x:
        assert np.isclose(integrated_function_two(x), function(x, 1.2, 3.0))

    # 3 variables

    delta_three = {"y": 1.0, "x": 3.0, "z": 5.0}
    delta_dist_three = density_function.DeltaDensityFactor(delta_three)
    integrated_function_three = (
        density_integrals.integrate_by_delta_density_factor(
            function, delta_dist_three
        )
    )

    assert np.isclose(integrated_function_three(), function(3.0, 1.0, 5.0))


def test_integrate_by_regular_density_factor_pass() -> None:
    """Cases where all the variables are valid"""

    def function(x, y, z):
        return np.sin(x) * np.exp(-y) * z**2

    # ----------------
    # One variable
    # ----------------

    # Note that these need not be normalised at this point
    def density_one(z):
        return z**2

    domain_one = {"z": [0.0, 1.0]}
    regular_dist_one = density_function.RegularDensityFactor(
        density_one, domain_one
    )

    integrated_function_one = (
        density_integrals.integrate_by_regular_density_factor(
            function, regular_dist_one
        )
    )

    # Total integrand is
    # sin(x) * exp(-y) * z^4
    # Answer is 1/5 * sin(x) * exp(-y)
    def answer_one(x, y):
        return 1 / 5 * np.sin(x) * np.exp(-y)

    random_xy = np.random.randn(100, 2)
    for x, y in random_xy:
        assert np.isclose(integrated_function_one(x, y), answer_one(x, y))

    # ----------------
    # Two variables
    # ----------------

    # Note that these need not be normalised at this point
    def density_two(x, z):
        return np.sin(x - z)

    domain_two = {"z": [0.0, 1.0], "x": [1.0, 2.0]}

    regular_dist_two = density_function.RegularDensityFactor(
        density_two, domain_two
    )

    integrated_function_two = (
        density_integrals.integrate_by_regular_density_factor(
            function, regular_dist_two
        )
    )

    # Answer is -9/2 e^y ( 5sin1 - 9sin2 + cos1 + 6cos2)
    def answer_two(y):
        return (
            1
            / 4
            * np.exp(-y)
            * (
                -2 * np.sin(3)
                + 3 * np.cos(1)
                + 2 * np.cos(2)
                + np.cos(3)
                - 2 * np.cos(4)
            )
        )

    random_y = np.random.randn(100)
    for y in random_y:
        assert np.isclose(answer_two(y), integrated_function_two(y))

    # --------------------------------------------------
    # Silly complex case
    # --------------------------------------------------

    def function_three(a, b, c, d, e, f, g):
        return np.sin(a - b + c) + np.sin(2 * f + 4 * g) - np.sin(e / g)

    def density_three(c, e, g):
        return np.cos(c - e) + g

    integration_domain = {"e": [0.4, 0.8], "c": [0.1, 0.3], "g": [0.9, 1.2]}
    distribution = density_function.RegularDensityFactor(
        density_three, integration_domain
    )

    integrated_function_three = (
        density_integrals.integrate_by_regular_density_factor(
            function_three, distribution
        )
    )

    a = 0.12
    b = 0.34
    d = -0.23
    f = 0.01
    result = integrated_function_three(a, b, d, f)
    # From wolframalpha:
    # https://www.wolframalpha.com/input?i=integrate+%28sin%280.12-0.34%2Bc%29%2Bsin%282*0.01%2B4g%29+-+sin%28x%2Fg%29+%29*%28cos%28c-x%29%2Bg%29+dx+dc+dg+for+0.4+%3C+x+%3C+0.8%2C+0.1+%3C+c+%3C+0.3%2C+0.9+%3C+g+%3C+1.2
    answer = -0.0655871
    assert np.isclose(result, answer)


def test_integrate_by_density_diracs():
    delta_func_one = {"x": 1.0, "y": 2.0, "z": 3.0}
    delta_func_two = {"x": 2.0, "y": -1.0, "z": 1.0}
    const_factor_one = 0.4
    const_factor_two = 0.6

    delta_term_one = density_function.DensityFunctionTerm.from_delta(
        delta_func_one, const_factor_one
    )
    delta_term_two = density_function.DensityFunctionTerm.from_delta(
        delta_func_two, const_factor_two
    )
    terms = [delta_term_one, delta_term_two]

    density = density_function.DensityFunction(terms)

    def f(x, y, z):
        return x * y**2 * z**3

    integral = density_integrals.integrate_by_density(f, density)
    answer = 0.4 * (1.0 * 2**2 * 3**3) + 0.6 * (
        2.0 * (-1.0) ** 2 * 1.0**3
    )

    assert np.isclose(integral(), answer)


def test_integrate_by_density_general():
    # p(x,m) =   0.2*delta(x-0.3)delta(m-0.4)
    #          + 0.4*delta(x-0.7)*2*m
    #          + 0.4*2*x*2*m
    # For 0 < x < 1, 0 < m < 1

    # -------------------------------------------------------------------------
    # First term
    # -------------------------------------------------------------------------

    delta = {"x": 0.3, "m": 0.4}
    const = 0.2

    term_one = density_function.DensityFunctionTerm.from_delta(delta, const)

    # -------------------------------------------------------------------------
    # Second term
    # -------------------------------------------------------------------------

    delta = {"x": 0.7}
    const = 0.4

    def regular(m):
        return 2 * m

    domain = {"m": [0.0, 1.0]}

    delta_dist = density_function.DeltaDensityFactor(delta, const)
    regular_dist = density_function.RegularDensityFactor(regular, domain)
    term_two = density_function.DensityFunctionTerm(
        delta_factor=delta_dist, regular_factor=regular_dist
    )

    # -------------------------------------------------------------------------
    # Third term
    # -------------------------------------------------------------------------

    def regular(x, m):
        return 2 * x * 2 * m * 0.4

    domain = {"x": [0.0, 1.0], "m": [0.0, 1.0]}

    term_three = density_function.DensityFunctionTerm.from_regular(
        regular, domain
    )

    # -------------------------------------------------------------------------
    # Combined
    # -------------------------------------------------------------------------

    terms = [term_one, term_two, term_three]
    density = density_function.DensityFunction(terms)

    def f(k1, k2, x, m):
        return (k1 + k2) * x * m * (m - 1)

    integral = density_integrals.integrate_by_density(f, density)

    def answer(k1, k2):
        return -1187 / 11250 * (k1 + k2)

    random_k1k2 = np.random.randn(100, 2)
    for k1, k2 in random_k1k2:
        assert np.isclose(integral(k1, k2), answer(k1, k2))
