"""Statistical properties of the particles in the medium"""
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import quadpy


@dataclass
class IsotropicSphereStatistics:
    size: Distribution
    refractive_index: Distribution


@dataclass
class OpticallyActiveSphereStatistics:
    size: Distribution
    refractive_indices: Distribution


class ProbabilityDistribution:
    def __init__(self, parameters, limits, density_function):
        pass


def f(x: float, y: float) -> float:
    return x*y



def get_g(function, y_low, y_high):
    scheme = quadpy.c1.gauss_legendre(5)
    def g(x):
        f_y = lambda y: function(x,y)
        output = scheme.integrate(f_y, [y_low, y_high])
        return output
    return g
    
    