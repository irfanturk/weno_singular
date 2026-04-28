"""
WENO3 (Jiang-Shu) reconstruction on a 1D uniform periodic mesh.

This module provides a vectorized implementation of the third-order
finite-volume WENO reconstruction at cell interfaces x_{i+1/2}, for
upwind-biased problems (positive wave speed).

Compared to WENO5, the WENO3 reconstruction uses only TWO substencils,
each spanning TWO cells, giving formal third-order accuracy on smooth
solutions.  It is computationally cheaper but less accurate; it is
useful as a baseline and as a building block for adaptive schemes.

References
----------
Liu, X.-D., Osher, S., and Chan, T. (1994),
    "Weighted essentially non-oscillatory schemes",
    J. Comput. Phys., 115(1), 200-212.

Türk, İ. (2016),
    Section 5.1 of the Ph.D. thesis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# WENO3 constants
# ----------------------------------------------------------------------

#: Linear (optimal) weights for the two substencils, summing to 1.
GAMMA: NDArray[np.float64] = np.array([1.0 / 3.0, 2.0 / 3.0])

#: Tiny constant in the nonlinear weights to avoid division by zero.
EPS: float = 1e-6


# ----------------------------------------------------------------------
# Smoothness indicators
# ----------------------------------------------------------------------

def smoothness_indicators(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Compute Jiang-Shu smoothness indicators for periodic 1D data.

    For each cell index i, two indicators :math:`\beta_0, \beta_1`
    are computed from a three-cell stencil
    :math:`\{u_{i-1}, u_i, u_{i+1}\}`:

    .. math::
        \beta_0 = (u_i - u_{i-1})^2,
        \qquad \beta_1 = (u_{i+1} - u_i)^2.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Cell averages on a uniform mesh.  Periodicity is assumed.

    Returns
    -------
    beta : ndarray, shape (2, n)
        ``beta[r, i]`` is :math:`\beta_r` at cell ``i``.
    """
    um1 = np.roll(u, 1)
    u0 = u
    up1 = np.roll(u, -1)

    beta0 = (u0 - um1) ** 2
    beta1 = (up1 - u0) ** 2

    return np.stack([beta0, beta1], axis=0)


def nonlinear_weights(
    beta: NDArray[np.float64],
    gamma: NDArray[np.float64] = GAMMA,
    eps: float = EPS,
) -> NDArray[np.float64]:
    r"""
    Convert smoothness indicators into normalized WENO nonlinear weights.

    .. math::
        \alpha_r = \gamma_r / (\varepsilon + \beta_r)^2,
        \qquad \omega_r = \alpha_r \big/ \sum_s \alpha_s.

    Parameters
    ----------
    beta : ndarray, shape (2, n)
        Smoothness indicators.
    gamma : ndarray, shape (2,), optional
        Linear weights.  Defaults to ``GAMMA``.
    eps : float, optional
        Regularization constant.  Defaults to ``EPS``.

    Returns
    -------
    omega : ndarray, shape (2, n)
        Nonlinear weights, columns sum to 1.
    """
    alpha = gamma[:, None] / (eps + beta) ** 2
    return alpha / alpha.sum(axis=0, keepdims=True)


# ----------------------------------------------------------------------
# WENO3 reconstruction at cell interfaces x_{i+1/2}
# ----------------------------------------------------------------------

def reconstruct(
    u: NDArray[np.float64],
    indicator_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Reconstruct ``u`` at cell interfaces :math:`x_{i+1/2}` using WENO3-JS.

    The reconstruction is upwind-biased and assumes periodic boundary
    conditions.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Cell averages on a uniform mesh.
    indicator_fn : callable, optional
        Function ``u -> beta`` returning a ``(2, n)`` array of smoothness
        indicators.  Defaults to :func:`smoothness_indicators`.

    Returns
    -------
    v_half : ndarray, shape (n,)
        Reconstruction at :math:`x_{i+1/2}`.
    omega : ndarray, shape (2, n)
        Nonlinear weights used.  Returned for analysis and diagnostics.
    """
    if indicator_fn is None:
        indicator_fn = smoothness_indicators

    um1 = np.roll(u, 1)
    u0 = u
    up1 = np.roll(u, -1)

    # Two candidate reconstructions at i+1/2
    v0 = -0.5 * um1 + 1.5 * u0
    v1 = 0.5 * u0 + 0.5 * up1

    beta = indicator_fn(u)
    omega = nonlinear_weights(beta)

    v_half = omega[0] * v0 + omega[1] * v1
    return v_half, omega


# ----------------------------------------------------------------------
# Spatial operator for u_t + u_x = 0 (semi-discrete RHS)
# ----------------------------------------------------------------------

def L_advection(
    u: NDArray[np.float64], h: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Compute the semi-discrete spatial operator for :math:`u_t + u_x = 0`
    using WENO3.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Cell averages.
    h : float
        Uniform mesh spacing.

    Returns
    -------
    Lu : ndarray, shape (n,)
        Semi-discrete spatial operator value at each cell.
    v_half : ndarray, shape (n,)
        Reconstruction at right interfaces (returned for diagnostics).
    omega : ndarray, shape (2, n)
        Nonlinear WENO weights (returned for diagnostics).
    """
    v_half, omega = reconstruct(u)
    Lu = -(v_half - np.roll(v_half, 1)) / h
    return Lu, v_half, omega


# ----------------------------------------------------------------------
# Sparse matrix form of the WENO3 spatial operator (frozen weights)
# ----------------------------------------------------------------------

def build_matrix(omega: NDArray[np.float64], h: float):
    r"""
    Assemble the sparse periodic banded matrix :math:`L` such that
    :math:`L\,u` approximates :math:`-\partial_x u` using WENO3 with
    the supplied (frozen) nonlinear weights.

    Each row has four nonzero entries at columns
    :math:`i-2, i-1, i, i+1` (mod ``n``).

    Parameters
    ----------
    omega : ndarray, shape (2, n)
        Nonlinear weights at each cell.
    h : float
        Uniform mesh spacing.

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Sparse periodic operator approximating :math:`-\partial_x`.
    """
    from scipy.sparse import csr_matrix

    o0, o1 = omega
    n = o0.size

    # Reconstruction at i+1/2 in row i uses columns i-1, i, i+1:
    #   V_{i+1/2} = o0[i] * (-0.5 u_{i-1} + 1.5 u_i)
    #             + o1[i] * ( 0.5 u_i     + 0.5 u_{i+1})
    #
    # Reconstruction at i-1/2 in row i uses columns i-2, i-1, i:
    #   V_{i-1/2} = o0[i-1] * (-0.5 u_{i-2} + 1.5 u_{i-1})
    #             + o1[i-1] * ( 0.5 u_{i-1} + 0.5 u_i)
    #
    # The WENO operator is L u = -(V_{i+1/2} - V_{i-1/2}) / h
    # = (V_{i-1/2} - V_{i+1/2}) / h.
    # Collect contributions to each column in row i:

    o0_im1 = np.roll(o0, 1)
    o1_im1 = np.roll(o1, 1)
    inv_h = 1.0 / h

    # column i-2:  comes only from V_{i-1/2}, with sign +
    d_m2 = (-0.5 * o0_im1) * inv_h
    # column i-1:  V_{i-1/2} contributes (+1.5 o0_im1 + 0.5 o1_im1),
    #              V_{i+1/2} contributes (-0.5 o0)
    d_m1 = (1.5 * o0_im1 + 0.5 * o1_im1 - (-0.5) * o0) * inv_h
    # column i:    V_{i-1/2} contributes (+0.5 o1_im1),
    #              V_{i+1/2} contributes (1.5 o0 + 0.5 o1)
    d_0 = (0.5 * o1_im1 - 1.5 * o0 - 0.5 * o1) * inv_h
    # column i+1:  comes only from V_{i+1/2}, with sign -
    d_p1 = -(0.5 * o1) * inv_h

    i_arr = np.arange(n)
    rows = np.tile(i_arr[:, None], (1, 4)).ravel()
    cols = ((i_arr[:, None] + np.array([-2, -1, 0, 1])) % n).ravel()
    data = np.column_stack([d_m2, d_m1, d_0, d_p1]).ravel()
    return csr_matrix((data, (rows, cols)), shape=(n, n))


__all__ = [
    "GAMMA",
    "EPS",
    "smoothness_indicators",
    "nonlinear_weights",
    "reconstruct",
    "L_advection",
    "build_matrix",
]
