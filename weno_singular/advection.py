"""
Solver for the linear scalar advection equation with optional singular
source terms.

Problem
-------
Solve

.. math::
    u_t + u_x = g(t)\\,\\delta(x - \\xi),
    \\qquad x \\in [a, b], \\quad t \\in [0, T],

with periodic boundary conditions.  When ``g`` is identically zero, this
reduces to the homogeneous linear advection :math:`u_t + u_x = 0`.

The method-of-lines discretization combines a WENO5 reconstruction
(:mod:`weno_singular.weno5`) with an SSP-RK3 time integrator
(:mod:`weno_singular.time_steppers`).  The singular source term is
discretized as a direct injection into a single "delta cell"
:math:`\\Omega_j` containing :math:`\\xi`:

.. math::
    \\frac{1}{h_j} \\int_{\\Omega_j} g(t)\\,\\delta(x - \\xi)\\,dx
    = \\frac{g(t)}{h_j}.

This treatment, used throughout Türk (2016, Section 6.1.3), preserves
mass exactly to machine precision and reproduces the published
convergence tables (e.g.\\ Table 6.4) for the canonical test problem
:math:`g(t) = \\sin(\\pi t),\\ \\xi = 1/3`.

References
----------
Türk, İ. (2016), Section 6.1 of the Ph.D. thesis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from weno_singular.time_steppers import ssp_rk3_step
from weno_singular.weno5 import L_advection


# ----------------------------------------------------------------------
# Helpers for delta-cell indexing and exact reference solutions
# ----------------------------------------------------------------------

def find_delta_cell(xi: float, x_left: float, M: int) -> int:
    """
    Return the 0-based Python index of the cell whose right interface
    is the smallest one >= ``xi``.

    Mirrors the convention used in Türk (2016): on a uniform mesh of
    ``M - 1`` cells over ``[x_left, x_right]``, the cell index is
    ``ceil((xi - x_left) * (M - 1)) - 1``.

    Parameters
    ----------
    xi : float
        Source location.
    x_left : float
        Left endpoint of the domain.
    M : int
        Number of mesh interfaces (so there are ``M - 1`` cells).

    Returns
    -------
    j : int
        Python index of the cell containing ``xi``.
    """
    return int(np.ceil((xi - x_left) * (M - 1))) - 1


def exact_cell_averages_singular(
    x: NDArray[np.float64],
    h: float,
    xi: float,
    t: float,
) -> NDArray[np.float64]:
    r"""
    Exact cell averages for the canonical singular source problem.

    For the problem :math:`u_t + u_x = \sin(\pi t)\,\delta(x - \xi)`
    with :math:`u(x, 0) = 0` and periodic BCs, the analytical solution
    is

    .. math::
        u(x, t) = \begin{cases}
            \sin\!\big(\pi(\xi + t - x)\big), & \xi \le x \le \xi + t,\\
            0, & \text{otherwise}.
        \end{cases}

    This routine returns its cell averages on the uniform mesh defined
    by cell centers ``x`` and width ``h``, evaluated at time ``t``.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Cell centers.
    h : float
        Uniform cell width.
    xi : float
        Source location.
    t : float
        Time at which to evaluate.

    Returns
    -------
    u_avg : ndarray, shape (n,)
        Exact cell averages.
    """
    if t <= 0.0:
        return np.zeros_like(x)
    a = x - 0.5 * h
    b = x + 0.5 * h
    a_clip = np.clip(a, xi, xi + t)
    b_clip = np.clip(b, xi, xi + t)
    F_a = np.cos(np.pi * (xi + t - a_clip)) / np.pi
    F_b = np.cos(np.pi * (xi + t - b_clip)) / np.pi
    return np.where(b_clip > a_clip, (F_b - F_a) / h, 0.0)


# ----------------------------------------------------------------------
# Solver: explicit RK3
# ----------------------------------------------------------------------

def solve_advection_singular(
    M: int = 181,
    N: int = 5001,
    T_final: float = 0.5,
    x_left: float = 0.0,
    x_right: float = 1.0,
    xi: float = 1.0 / 3.0,
    source_fn: Callable[[float], float] = lambda t: np.sin(np.pi * t),
    initial_fn: Callable[[NDArray[np.float64], float], NDArray[np.float64]] | None = None,
    track_max_error: bool = True,
) -> dict:
    r"""
    Solve :math:`u_t + u_x = g(t)\,\delta(x - \xi)` with WENO5 + SSP-RK3.

    Parameters
    ----------
    M : int
        Number of mesh interfaces (``M - 1`` cells).
    N : int
        Number of time levels (``N - 1`` steps).
    T_final : float
        Final time.
    x_left, x_right : float
        Domain endpoints.  Periodic boundary conditions are imposed.
    xi : float
        Location of the Dirac source.
    source_fn : callable, optional
        Time-dependent amplitude :math:`g(t)`.  Defaults to
        :math:`\sin(\pi t)`, matching Türk (2016, Example 6.1.3).
    initial_fn : callable, optional
        Function ``(x, h) -> u_avg`` returning the cell averages of the
        initial condition.  Defaults to zero (the canonical test case).
    track_max_error : bool, optional
        If True (default) and ``initial_fn`` is None and the source is
        the canonical sine, also compute the running maximum
        :math:`L_\infty` error against the analytical solution.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``u`` : final cell averages
        - ``x`` : cell centers
        - ``h`` : mesh spacing
        - ``dt`` : time step
        - ``delta_idx`` : index of the delta cell
        - ``T_final``, ``xi`` : echoes of input
        - ``max_err_inf`` : running max :math:`L_\infty` cell-average
          error if applicable, else ``None``
        - ``L1_err_face`` : final :math:`L_1` error at right interfaces
          (matches the convention of Türk 2016 Table 6.4) if applicable,
          else ``None``
    """
    n = M - 1
    h = (x_right - x_left) / n
    dt = T_final / (N - 1)
    x = np.linspace(x_left + h / 2, x_right - h / 2, n)
    delta_idx = find_delta_cell(xi, x_left, M)

    # Initial condition (default: zero)
    if initial_fn is None:
        u = np.zeros(n)
        canonical = True
    else:
        u = initial_fn(x, h)
        canonical = False

    # Time-dependent RHS = WENO advection + delta source
    def rhs(state: NDArray[np.float64], t_eval: float) -> NDArray[np.float64]:
        Lu, _, _ = L_advection(state, h)
        Lu[delta_idx] += source_fn(t_eval) / h
        return Lu

    # Time loop
    track = track_max_error and canonical
    max_err_inf = 0.0 if track else None
    t_n = 0.0
    for _ in range(N - 1):
        u = ssp_rk3_step(u, dt, rhs, t=t_n)
        t_n += dt
        if track:
            u_exact = exact_cell_averages_singular(x, h, xi, t_n)
            err = float(np.max(np.abs(u - u_exact)))
            if err > max_err_inf:
                max_err_inf = err

    # Final L1 error at right interfaces (matches Türk 2016 Table 6.4)
    if track:
        from weno_singular.weno5 import reconstruct
        v_half, _ = reconstruct(u)
        x_face = x + h / 2
        in_supp = (x_face > xi) & (x_face < xi + T_final)
        exact_face = np.where(
            in_supp, np.sin(np.pi * (xi + T_final - x_face)), 0.0
        )
        L1_err_face = float(np.sum(np.abs(v_half - exact_face)) * h)
    else:
        L1_err_face = None

    return {
        "u": u,
        "x": x,
        "h": h,
        "dt": dt,
        "delta_idx": delta_idx,
        "T_final": T_final,
        "xi": xi,
        "max_err_inf": max_err_inf,
        "L1_err_face": L1_err_face,
    }


__all__ = [
    "find_delta_cell",
    "exact_cell_averages_singular",
    "solve_advection_singular",
]
