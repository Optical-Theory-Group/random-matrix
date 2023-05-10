from random_matrix.amplitude_matrix import (
    _test,
    _isotropic_sphere,
    _chiral_sphere,
)
from random_matrix.utils.types import FloatLike
from dataclasses import dataclass, astuple

# -----------------------------------------------------------------------------
# Test case
# -----------------------------------------------------------------------------


@dataclass
class TestParams:
    a: float
    b: float
    c: float


def test(k_inc: FloatLike, k_sca: FloatLike, params: TestParams) -> FloatLike:
    return _test.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Isotropic sphere
# -----------------------------------------------------------------------------


@dataclass
class IsotropicSphereParams:
    # Size parameter
    x: float
    # Relative refractive index
    m: float


def isotropic_sphere(
    k_inc: FloatLike, k_sca: FloatLike, params: IsotropicSphereParams
) -> FloatLike:
    return _isotropic_sphere.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Chiral sphere
# -----------------------------------------------------------------------------


@dataclass
class ChiralSphereParams:
    # Size parameter
    x: float
    # Relative refractive index for left circular
    m_l: float
    # Relative refractive index for right circular
    m_r: float


def chiral_sphere(
    k_inc: FloatLike, k_sca: FloatLike, params: ChiralSphereParams
) -> FloatLike:
    return _chiral_sphere.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Anisotropic T matrix
# -----------------------------------------------------------------------------
