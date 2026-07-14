"""
WENO5 (Jiang-Shu) reconstruction on a 1D uniform periodic mesh.

This module provides a vectorized implementation of the fifth-order
finite-volume WENO reconstruction at cell interfaces x_{i+1/2}, for
upwind-biased problems (positive wave speed).

The implementation follows Jiang & Shu (1996) and is identical, up to
floating-point round-off, to the WENO5-JS algorithm described in
Section 5.2 of Türk (2016).

References
----------
Jiang, G.-S. and Shu, C.-W. (1996),
    "Efficient implementation of weighted ENO schemes",
    J. Comput. Phys., 126(1), 202-228.

Türk, İ. (2016),
    "On the numerical solution of advection diffusion reaction equations
    with singular source terms",
    Ph.D. thesis, İstanbul University.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# WENO5 constants
# ----------------------------------------------------------------------

#: Linear (optimal) weights for the three substencils, summing to 1.
GAMMA: NDArray[np.float64] = np.array([1.0 / 10.0, 3.0 / 5.0, 3.0 / 10.0])

#: Tiny constant in the WENO5-JS nonlinear weights to avoid division by zero.
EPS: float = 1e-6

#: Regularization constant for the WENO-Z weights (Borges et al., 2008).
#: Much smaller than ``EPS``: the WENO-Z weights are insensitive to it.
EPS_Z: float = 1e-40


# ----------------------------------------------------------------------
# Smoothness indicators
# ----------------------------------------------------------------------

def smoothness_indicators(u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Compute Jiang-Shu smoothness indicators for periodic 1D data.

    For each cell index i, three indicators :math:`\beta_0, \beta_1, \beta_2`
    are computed from a five-cell stencil
    :math:`\{u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}\}`.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Cell averages on a uniform mesh.  Periodicity is assumed.

    Returns
    -------
    beta : ndarray, shape (3, n)
        ``beta[r, i]`` is :math:`\beta_r` at cell ``i``.
    """
    um2 = np.roll(u, 2)
    um1 = np.roll(u, 1)
    u0 = u
    up1 = np.roll(u, -1)
    up2 = np.roll(u, -2)

    beta0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + u0) ** 2 \
            + 0.25 * (um2 - 4.0 * um1 + 3.0 * u0) ** 2
    beta1 = (13.0 / 12.0) * (um1 - 2.0 * u0 + up1) ** 2 \
            + 0.25 * (um1 - up1) ** 2
    beta2 = (13.0 / 12.0) * (u0 - 2.0 * up1 + up2) ** 2 \
            + 0.25 * (3.0 * u0 - 4.0 * up1 + up2) ** 2

    return np.stack([beta0, beta1, beta2], axis=0)


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
    beta : ndarray, shape (3, n)
        Smoothness indicators.
    gamma : ndarray, shape (3,), optional
        Linear weights.  Defaults to ``GAMMA``.
    eps : float, optional
        Regularization constant.  Defaults to ``EPS``.

    Returns
    -------
    omega : ndarray, shape (3, n)
        Nonlinear weights, columns sum to 1.
    """
    alpha = gamma[:, None] / (eps + beta) ** 2
    return alpha / alpha.sum(axis=0, keepdims=True)


# ----------------------------------------------------------------------
# WENO5 reconstruction at cell interfaces x_{i+1/2}
# ----------------------------------------------------------------------

def nonlinear_weights_z(
    beta: NDArray[np.float64],
    gamma: NDArray[np.float64] = GAMMA,
    eps: float = EPS_Z,
    p: int = 2,
) -> NDArray[np.float64]:
    r"""
    WENO-Z nonlinear weights (Borges et al., 2008).

    A global smoothness indicator :math:`\tau_5 = |\beta_0 - \beta_2|`
    is used to rescale the Jiang-Shu weights:

    .. math::
        \alpha_r = \gamma_r
                   \left[1 + \left(\frac{\tau_5}{\beta_r + \varepsilon}\right)^{p}\right],
        \qquad \omega_r = \alpha_r \big/ \sum_s \alpha_s .

    On smooth data :math:`\tau_5 = O(h^5)`, so :math:`\omega_r \to \gamma_r`
    much faster than for WENO5-JS; the scheme therefore suffers less
    accuracy loss at critical points.  Near a discontinuity the
    behaviour reverts to that of a standard WENO reconstruction.

    Parameters
    ----------
    beta : ndarray, shape (3, n)
        Jiang-Shu smoothness indicators.
    gamma : ndarray, shape (3,), optional
        Linear weights.  Defaults to ``GAMMA``.
    eps : float, optional
        Regularization constant.  Defaults to ``EPS_Z`` (1e-40), which
        is the value recommended by Borges et al.; unlike WENO5-JS, the
        WENO-Z weights are not sensitive to this parameter.
    p : int, optional
        Power in the rescaling.  ``p = 2`` (default) is the standard
        choice for fifth-order WENO-Z.

    Returns
    -------
    omega : ndarray, shape (3, n)
        Nonlinear weights, columns sum to 1.

    References
    ----------
    Borges, R., Carmona, M., Costa, B., and Don, W. S. (2008),
        "An improved weighted essentially non-oscillatory scheme for
        hyperbolic conservation laws",
        J. Comput. Phys., 227(6), 3191-3211.
    """
    tau5 = np.abs(beta[0] - beta[2])
    alpha = gamma[:, None] * (1.0 + (tau5[None, :] / (beta + eps)) ** p)
    return alpha / alpha.sum(axis=0, keepdims=True)


#: Registry of available nonlinear-weight functions, keyed by scheme name.
WEIGHT_FUNCTIONS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "js": nonlinear_weights,
    "z": nonlinear_weights_z,
}


def reconstruct(
    u: NDArray[np.float64],
    indicator_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    weight_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    variant: str = "js",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Reconstruct ``u`` at cell interfaces :math:`x_{i+1/2}` using WENO5.

    The candidate stencil polynomials are those of Jiang & Shu (1996);
    the ``variant`` argument selects how the nonlinear weights are
    formed from the smoothness indicators (``"js"`` for Jiang-Shu,
    ``"z"`` for WENO-Z).

    The reconstruction is upwind-biased (it assumes the wave propagates
    in the positive ``x`` direction, i.e.\ the characteristic speed
    :math:`a > 0`).  Periodic boundary conditions are assumed.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Cell averages on a uniform mesh.
    indicator_fn : callable, optional
        Function ``u -> beta`` returning a ``(3, n)`` array of smoothness
        indicators.  Defaults to :func:`smoothness_indicators`.
    weight_fn : callable, optional
        Function ``beta -> omega`` returning a ``(3, n)`` array of
        nonlinear weights.  Overrides ``variant`` when given.
    variant : {"js", "z"}, optional
        Selects the nonlinear-weight formula: ``"js"`` (default) for
        Jiang-Shu, ``"z"`` for WENO-Z (Borges et al., 2008).

    Returns
    -------
    v_half : ndarray, shape (n,)
        Reconstruction at :math:`x_{i+1/2}`.
    omega : ndarray, shape (3, n)
        Nonlinear weights used.  Returned for analysis and diagnostics.
    """
    if indicator_fn is None:
        indicator_fn = smoothness_indicators
    if weight_fn is None:
        try:
            weight_fn = WEIGHT_FUNCTIONS[variant]
        except KeyError:
            raise ValueError(
                f"unknown WENO5 variant {variant!r}; "
                f"expected one of {sorted(WEIGHT_FUNCTIONS)}"
            ) from None

    um2 = np.roll(u, 2)
    um1 = np.roll(u, 1)
    u0 = u
    up1 = np.roll(u, -1)
    up2 = np.roll(u, -2)

    # Three candidate reconstructions at i+1/2
    v0 = (1.0 / 3.0) * um2 + (-7.0 / 6.0) * um1 + (11.0 / 6.0) * u0
    v1 = (-1.0 / 6.0) * um1 + (5.0 / 6.0) * u0 + (1.0 / 3.0) * up1
    v2 = (1.0 / 3.0) * u0 + (5.0 / 6.0) * up1 + (-1.0 / 6.0) * up2

    beta = indicator_fn(u)
    omega = weight_fn(beta)

    v_half = omega[0] * v0 + omega[1] * v1 + omega[2] * v2
    return v_half, omega


# ----------------------------------------------------------------------
# Spatial operator for u_t + u_x = 0 (semi-discrete RHS)
# ----------------------------------------------------------------------

def L_advection(
    u: NDArray[np.float64], h: float, variant: str = "js"
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Compute the semi-discrete spatial operator for :math:`u_t + u_x = 0`.

    Returns :math:`L(u)` such that the method-of-lines ODE is
    :math:`du/dt = L(u)`.  For the upwind-biased linear flux this
    reduces to

    .. math::
        L(u)_i = -\frac{F_{i+1/2} - F_{i-1/2}}{h},
        \qquad F_{i+1/2} = u^{-}_{i+1/2}.

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
    omega : ndarray, shape (3, n)
        Nonlinear WENO weights (returned for diagnostics).
    """
    v_half, omega = reconstruct(u, variant=variant)
    # F_{i-1/2} is v_half rolled right by one (periodic).
    Lu = -(v_half - np.roll(v_half, 1)) / h
    return Lu, v_half, omega


# ----------------------------------------------------------------------
# Sparse matrix form of the WENO5 spatial operator (frozen weights)
# ----------------------------------------------------------------------

def build_matrix(omega: NDArray[np.float64], h: float):
    r"""
    Assemble the sparse periodic banded matrix :math:`L` such that
    :math:`L\,u` approximates :math:`-\partial_x u` using WENO5 with
    the supplied (frozen) nonlinear weights.

    Each row has six nonzero entries at columns
    :math:`i-3, i-2, i-1, i, i+1, i+2` (mod ``n``).  This linear-
    operator form is the building block of the semi-implicit
    Crank-Nicolson corrector
    (:func:`weno_singular.time_steppers.crank_nicolson_corrector`).

    Parameters
    ----------
    omega : ndarray, shape (3, n)
        Nonlinear weights at each cell, e.g.\ from a predictor stage.
    h : float
        Uniform mesh spacing.

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Sparse periodic operator approximating :math:`-\partial_x`.
    """
    from scipy.sparse import csr_matrix

    o0, o1, o2 = omega
    n = o0.size

    inv6h = 1.0 / (6.0 * h)
    a1 = 1.0 * inv6h
    a2 = 2.0 * inv6h
    a5 = 5.0 * inv6h
    a7 = 7.0 * inv6h
    a11 = 11.0 * inv6h

    o0_im1 = np.roll(o0, 1)
    o1_im1 = np.roll(o1, 1)
    o2_im1 = np.roll(o2, 1)

    # Six diagonals (length n each, one entry per row i)
    d_m3 = a2 * o0_im1
    d_m2 = -a7 * o0_im1 - a1 * o1_im1 - a2 * o0
    d_m1 = a11 * o0_im1 + a5 * o1_im1 + a2 * o2_im1 + a7 * o0 + a1 * o1
    d_0 = a2 * o1_im1 + a5 * o2_im1 - a11 * o0 - a5 * o1 - a2 * o2
    d_p1 = -a1 * o2_im1 - a2 * o1 - a5 * o2
    d_p2 = a1 * o2

    i_arr = np.arange(n)
    rows = np.tile(i_arr[:, None], (1, 6)).ravel()
    cols = ((i_arr[:, None] + np.array([-3, -2, -1, 0, 1, 2])) % n).ravel()
    data = np.column_stack([d_m3, d_m2, d_m1, d_0, d_p1, d_p2]).ravel()
    return csr_matrix((data, (rows, cols)), shape=(n, n))


__all__ = [
    "GAMMA",
    "EPS",
    "EPS_Z",
    "WEIGHT_FUNCTIONS",
    "smoothness_indicators",
    "nonlinear_weights",
    "nonlinear_weights_z",
    "reconstruct",
    "L_advection",
    "build_matrix",
]
