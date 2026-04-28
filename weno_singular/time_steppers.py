"""
Time integration schemes for the method-of-lines ODE :math:`du/dt = L(u)`.

This module provides two strategies:

1.  :func:`ssp_rk3_step` — the third-order strong-stability-preserving
    Runge-Kutta scheme of Shu & Osher (1988), used for fully explicit
    time stepping.  When the right-hand side depends explicitly on time
    (e.g. time-dependent forcing), the substage times are
    :math:`t^n`, :math:`t^n + \\Delta t`, and :math:`t^n + \\Delta t / 2`.

2.  :func:`semi_implicit_step` — an SSP-RK3 predictor followed by a
    Crank-Nicolson corrector with the WENO weights frozen at the
    predictor.  This recovers second-order accuracy in time while
    keeping the corrector linear (one sparse solve per step).

References
----------
Shu, C.-W. and Osher, S. (1988),
    "Efficient implementation of essentially non-oscillatory
    shock-capturing schemes",
    J. Comput. Phys., 77(2), 439-471.

Türk, İ. (2016),
    Section 6.1.2 of the Ph.D. thesis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------

#: An RHS callable.  The simple form ``rhs(u)`` is used for time-
#: independent right-hand sides; the time-dependent variant accepts
#: an extra scalar argument and is wrapped by :func:`ssp_rk3_step`.
TimeIndependentRHS = Callable[[NDArray[np.float64]], NDArray[np.float64]]
TimeDependentRHS = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


# ----------------------------------------------------------------------
# Explicit SSP-RK3 (Shu-Osher)
# ----------------------------------------------------------------------

def ssp_rk3_step(
    u: NDArray[np.float64],
    dt: float,
    rhs: TimeIndependentRHS | TimeDependentRHS,
    t: float | None = None,
) -> NDArray[np.float64]:
    r"""
    Take one third-order SSP-RK (Shu-Osher) step.

    The update is

    .. math::
        u^{(1)} &= u^n + \Delta t\,L(u^n, t^n)\\
        u^{(2)} &= \tfrac{3}{4}u^n + \tfrac{1}{4}
                   \big(u^{(1)} + \Delta t\,L(u^{(1)}, t^n + \Delta t)\big)\\
        u^{n+1} &= \tfrac{1}{3}u^n + \tfrac{2}{3}
                   \big(u^{(2)} + \Delta t\,L(u^{(2)}, t^n + \tfrac{1}{2}\Delta t)\big)

    Parameters
    ----------
    u : ndarray
        Current state at time :math:`t^n`.
    dt : float
        Time-step size :math:`\Delta t`.
    rhs : callable
        Either ``rhs(u)`` if the right-hand side is time-independent, or
        ``rhs(u, t)`` if it depends on time.  The second form is used
        whenever ``t`` is passed.
    t : float, optional
        Current time :math:`t^n`.  Required when ``rhs`` is
        time-dependent; ignored otherwise.

    Returns
    -------
    u_next : ndarray
        State at time :math:`t^n + \Delta t`.
    """
    if t is None:
        # Time-independent RHS.
        rhs_ti: TimeIndependentRHS = rhs                                # type: ignore[assignment]
        u1 = u + dt * rhs_ti(u)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs_ti(u1))
        u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs_ti(u2))
        return u3

    # Time-dependent RHS: substage times t^n, t^n+dt, t^n+dt/2.
    rhs_td: TimeDependentRHS = rhs                                      # type: ignore[assignment]
    u1 = u + dt * rhs_td(u, t)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs_td(u1, t + dt))
    u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs_td(u2, t + 0.5 * dt))
    return u3


# ----------------------------------------------------------------------
# Crank-Nicolson corrector with frozen coefficients
# ----------------------------------------------------------------------

def crank_nicolson_corrector(
    u: NDArray[np.float64],
    dt: float,
    L_old: csr_matrix,
    L_new: csr_matrix,
    source_old: NDArray[np.float64] | None = None,
    source_new: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    r"""
    Apply one Crank-Nicolson step with frozen spatial operators.

    Solves the linear system

    .. math::
        \big(I - \tfrac{\Delta t}{2} L_\text{new}\big) u^{n+1}
        = \big(I + \tfrac{\Delta t}{2} L_\text{old}\big) u^n
          + \tfrac{\Delta t}{2}\big(s^n + s^{n+1}\big),

    where :math:`L_\text{old}` and :math:`L_\text{new}` are sparse
    discretizations of :math:`L = -\partial_x` evaluated with the WENO
    weights at :math:`t^n` and at the predictor :math:`u^{n+1, *}`,
    respectively.  Optional source vectors :math:`s^n, s^{n+1}` model
    additional explicit terms (e.g. Dirac forcing).

    Parameters
    ----------
    u : ndarray, shape (n,)
        Current state :math:`u^n`.
    dt : float
        Time step.
    L_old, L_new : scipy.sparse.csr_matrix, shape (n, n)
        Frozen spatial operators at the old and new times.
    source_old, source_new : ndarray, optional
        Source vectors at :math:`t^n` and :math:`t^{n+1}`.

    Returns
    -------
    u_next : ndarray, shape (n,)
        Updated state :math:`u^{n+1}`.
    """
    n = u.size
    Iden = identity(n, format="csr")

    A = Iden - 0.5 * dt * L_new
    b = (Iden + 0.5 * dt * L_old) @ u
    if source_old is not None:
        b = b + 0.5 * dt * source_old
    if source_new is not None:
        b = b + 0.5 * dt * source_new

    return spsolve(A, b)


__all__ = [
    "TimeIndependentRHS",
    "TimeDependentRHS",
    "ssp_rk3_step",
    "crank_nicolson_corrector",
]
