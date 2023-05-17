import numpy as np

from random_matrix.utils.types import FloatLike, MathematicalFunction

# -----------------------------------------------------------------------------
# kz and functions involved in statistics
# -----------------------------------------------------------------------------


def kz(kappa: FloatLike) -> FloatLike:
    """Note this is NORMALISED by k"""

    return np.sqrt(1 - np.linalg.norm(kappa))


def inverse_kz(kappa: FloatLike) -> FloatLike:
    """Factor that appears in lots of integrals.

    Often appears as sec(theta)
    """

    return 1.0 / kz(kappa)


def sinc(x: FloatLike) -> FloatLike:
    """Standard sinc function. Differs to scipy's by not including pi
    normalization."""

    if np.isclose(x, 0.0):
        return 1.0
    else:
        return np.sin(x) / x


def sinc_mean(k: FloatLike, L: FloatLike, kappa: FloatLike) -> FloatLike:
    """Sinc function that appears in calculations of mean"""
    return sinc(k * L * kz(kappa))


def get_sinc_mean_kappa(k: FloatLike, L: FloatLike) -> MathematicalFunction:
    """Returns sinc mean as a function of kappa for given k and L arguments"""

    def output(kappa: FloatLike) -> FloatLike:
        return sinc_mean(k, L, kappa)

    return output
