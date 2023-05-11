from random_matrix.amplitude_matrix import (
    _test,
    _isotropic_sphere,
    _chiral_sphere,
    _anisotropic_tmatrix,
)
from random_matrix.utils.types import FloatLike, MathematicalFunction
from dataclasses import dataclass, astuple, asdict
from typing import Protocol

# -----------------------------------------------------------------------------
# Test case
# -----------------------------------------------------------------------------


@dataclass
class TestParameters:
    a: float
    b: float
    c: float


def test(
    k_inc: FloatLike, k_sca: FloatLike, params: TestParameters
) -> FloatLike:
    return _test.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Isotropic sphere
# -----------------------------------------------------------------------------


@dataclass
class IsotropicSphereParameters:
    # Size parameter
    x: float

    # Relative refractive index
    m: float


def isotropic_sphere(
    k_inc: FloatLike, k_sca: FloatLike, params: IsotropicSphereParameters
) -> FloatLike:
    return _isotropic_sphere.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Chiral sphere
# -----------------------------------------------------------------------------


@dataclass
class ChiralSphereParameters:
    # Size parameter
    x: float

    # Relative refractive index for left circular
    m_l: float

    # Relative refractive index for right circular
    m_r: float


def chiral_sphere(
    k_inc: FloatLike, k_sca: FloatLike, params: ChiralSphereParameters
) -> FloatLike:
    return _chiral_sphere.get_A(k_inc, k_sca, *astuple(params))


# -----------------------------------------------------------------------------
# Anisotropic T matrix
# -----------------------------------------------------------------------------


@dataclass
class AnisotropicTMatrixParameters:
    # maximum polar index for spherical harmonics
    nmax: int

    # optical frequency
    omega: float

    # permitivitty of surrounding medium
    eps_s: complex

    # permitivitty tensor of particle
    eps_p: complex

    # magnetic permeability of particle and surroundings
    mu: float

    # cartesian coordinates defining particle surface
    psurf_coords: FloatLike

    # integration weights across particle surface
    psurf_weights: float

    # cartesian normal to particle surface
    psurf_norms: FloatLike

    # number of polar angles used in angular spectrum integral
    Nt: int

    # number of azimuthal angles used in angular spectrum integral
    Np: int

    # wavenumber in surround medium
    k_s: float

    # euler angle alpha defining particle rotation(s)
    euler_alpha: FloatLike

    # euler angle beta defining particle rotation(s) multiple values used when
    # rotational averaging performed
    euler_beta: float

    # euler angle gamma defining particle rotation(s)
    euler_gamma: float

    # flag to perform rotational averaging
    rotational_average: bool

    # pdf corresponding to meshgrid of euler angles used for rotational
    # averaging
    angular_pdf: MathematicalFunction


def anisotropic_t_matrix(
    k_inc: FloatLike, k_sca: FloatLike, params: AnisotropicTMatrixParameters
) -> FloatLike:
    a_matrix_generator = _anisotropic_tmatrix.AmatrixGenerator()
    return a_matrix_generator.get_A(k_inc, k_sca, **asdict(params))
