"""Module for integrating functions over regions defined by Mode objects"""


import numpy as np
import quadpy
import scipy


def integrate(function, mode, scheme=None, degree=None, function_params=None):
    scheme = quadpy.t2.get_good_scheme(10)

    integral = 0.0
    for triangle in mode.triangulation:
        print(triangle)
        a = scheme.integrate(function, triangle)

    return integral
