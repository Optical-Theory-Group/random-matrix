import numpy as np

from random_matrix.utils.types import Numeric, MathematicalFunction

# -----------------------------------------------------------------------------
# kz and functions involved in statistics
# -----------------------------------------------------------------------------


def kz(kappa: Numeric) -> Numeric:
    """Note this is NORMALISED by k"""
    return np.sqrt(1 - np.linalg.norm(kappa, axis=0) ** 2)


def inverse_kz(kappa: Numeric) -> Numeric:
    """Factor that appears in lots of integrals.

    Often appears as sec(theta)
    """

    return 1.0 / kz(kappa)


def sinc(x: Numeric) -> Numeric:
    """Standard sinc function. Differs to scipy's by not including pi
    normalization."""

    if isinstance(x, list):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        return 1.0 if np.isclose(x, 0.0) else np.sin(x) / x
    result = np.empty_like(x)
    mask = np.isclose(x, 0.0)
    non_zero_x_vals = x[~mask]
    result[mask] = 1.0
    result[~mask] = np.sin(non_zero_x_vals) / non_zero_x_vals
    return result


def sinc_mean(k: Numeric, L: Numeric, kappa: Numeric) -> Numeric:
    """Sinc function that appears in calculations of mean"""
    return sinc(k * L * kz(kappa))


def get_sinc_mean_kappa(k: Numeric, L: Numeric) -> MathematicalFunction:
    """Returns sinc mean as a function of kappa for given k and L arguments"""

    def output(kappa: Numeric) -> Numeric:
        return sinc_mean(k, L, kappa)

    return output
