"""
WENO solver for the inviscid Burgers equation with a singular source:

.. math::
    u_t + \\left(\\tfrac{1}{2}u^2\\right)_x = g(t)\\,\\delta(x - \\xi),
    \\qquad x \\in [a, b],\\quad t \\in [0, T],

on a uniform periodic mesh.

The spatial reconstruction is shared with the linear solver
(:mod:`weno_singular.weno5` / :mod:`weno_singular.weno3`).  What changes
relative to linear advection is the flux: it is now the nonlinear
:math:`f(u) = u^2/2`, whose characteristic speed :math:`f'(u) = u` is
solution-dependent and changes sign.  A single upwind-biased
reconstruction is therefore no longer sufficient, so the interface flux
is evaluated with a **local Lax--Friedrichs (Rusanov)** numerical flux:

.. math::
    \\hat f_{i+1/2}
    = \\tfrac12\\big[f(u^-_{i+1/2}) + f(u^+_{i+1/2})\\big]
    - \\tfrac{\\alpha}{2}\\big[u^+_{i+1/2} - u^-_{i+1/2}\\big],
    \\qquad \\alpha = \\max_i |u_i|,

where :math:`u^-` and :math:`u^+` are the WENO reconstructions of the
left- and right-biased states at the interface.  For the smooth-solution
convergence studies used here this reduces to the correct upwind flux
wherever the wind has a definite sign.

The singular source is injected into a single cell exactly as in the
linear solver (:func:`weno_singular.advection.find_delta_cell`), so the
``on_interface`` placement policy carries over unchanged.  This is what
lets the interface-alignment study be repeated for a nonlinear flux.

There is in general no closed-form solution once a shock forms, so
accuracy is assessed by self-convergence against a fine-mesh reference,
following Suarez, Jacobs & Don (2014).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from weno_singular import weno3 as _weno3
from weno_singular import weno5 as _weno5
from weno_singular.advection import find_delta_cell
from weno_singular.time_steppers import ssp_rk3_step

# ----------------------------------------------------------------------
# Reconstruction dispatch (shared with the linear solver)
# ----------------------------------------------------------------------

_RECON = {
    "weno5": (_weno5, "js"),
    "weno5z": (_weno5, "z"),
    "weno3": (_weno3, None),
}


def _reconstructors(scheme: str):
    """
    Return two callables ``(recon_minus, recon_plus)`` that reconstruct
    the left-biased (``u^-``) and right-biased (``u^+``) interface states
    at :math:`x_{i+1/2}` for the requested scheme.

    ``u^-`` is the existing upwind-biased reconstruction.  ``u^+`` is the
    mirror-image reconstruction, obtained by reflecting the stencil:
    reconstructing the flipped array and shifting recovers the
    right-biased value at the same interface.
    """
    try:
        mod, variant = _RECON[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; expected one of {sorted(_RECON)}"
        ) from None

    if variant is None:
        def recon_minus(u):
            return mod.reconstruct(u)[0]
    else:
        def recon_minus(u):
            return mod.reconstruct(u, variant=variant)[0]

    def recon_plus(u):
        # Right-biased state at i+1/2 = left-biased state at i+1/2 of the
        # mirror problem.  Reflect, reconstruct, reflect back and shift.
        uf = u[::-1]
        vf = recon_minus(uf)
        # vf[k] is the value at the right interface of flipped cell k,
        # i.e. at the left interface of original cell (n-1-k).  Rolling
        # aligns it with the right interface x_{i+1/2} of original cell i.
        return np.roll(vf[::-1], -1)

    return recon_minus, recon_plus


def _burgers_rhs_factory(h: float, scheme: str, delta_idx, source_fn):
    """Build the method-of-lines RHS for Burgers with a Rusanov flux."""
    recon_minus, recon_plus = _reconstructors(scheme)

    def flux(u):
        return 0.5 * u * u

    def rhs(state: NDArray[np.float64], t_eval: float) -> NDArray[np.float64]:
        um = recon_minus(state)            # u^-_{i+1/2}
        up = recon_plus(state)             # u^+_{i+1/2}
        alpha = float(np.max(np.abs(state)))          # global LF speed
        f_hat = 0.5 * (flux(um) + flux(up)) - 0.5 * alpha * (up - um)
        # Lu_i = -(f_{i+1/2} - f_{i-1/2}) / h
        Lu = -(f_hat - np.roll(f_hat, 1)) / h
        if delta_idx is not None:
            Lu[delta_idx] += source_fn(t_eval) / h
        return Lu

    return rhs


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------

def solve_burgers_singular(
    M: int = 201,
    N: int = 4001,
    T_final: float = 0.3,
    x_left: float = 0.0,
    x_right: float = 1.0,
    xi: float = 1.0 / 3.0,
    source_fn: Callable[[float], float] | None = lambda t: 1.0,
    initial_fn: Callable[[NDArray[np.float64], float], NDArray[np.float64]] | None = None,
    scheme: str = "weno5",
    on_interface: str = "downwind",
) -> dict:
    r"""
    Solve :math:`u_t + (u^2/2)_x = g(t)\,\delta(x-\xi)` with WENO + SSP-RK3
    and a local Lax--Friedrichs flux.

    Parameters mirror :func:`weno_singular.advection.solve_advection_singular`.
    The default ``source_fn`` is the constant unit forcing ``g(t) = 1``.

    Returns
    -------
    result : dict
        Keys ``u``, ``x``, ``h``, ``dt``, ``delta_idx``, ``T_final``,
        ``xi``, ``scheme``.  No error field is computed here because the
        nonlinear problem has no closed-form solution in general; use
        :func:`burgers_self_convergence` for accuracy studies.
    """
    n = M - 1
    h = (x_right - x_left) / n
    dt = T_final / (N - 1)
    x = np.linspace(x_left + h / 2, x_right - h / 2, n)
    has_source = source_fn is not None
    delta_idx = (
        find_delta_cell(xi, x_left, M, x_right, on_interface) if has_source else None
    )

    u = np.zeros(n) if initial_fn is None else initial_fn(x, h)

    rhs = _burgers_rhs_factory(h, scheme, delta_idx, source_fn)

    t_n = 0.0
    for _ in range(N - 1):
        u = ssp_rk3_step(u, dt, rhs, t=t_n)
        t_n += dt

    return {
        "u": u, "x": x, "h": h, "dt": dt,
        "delta_idx": delta_idx, "T_final": T_final, "xi": xi,
        "scheme": scheme,
    }


def _sample_to_coarse(u_fine, n_fine, n_coarse):
    """Average a fine-mesh solution down to a coarse mesh (both uniform,
    n_fine an integer multiple of n_coarse). Returns coarse cell averages."""
    r = n_fine // n_coarse
    return u_fine.reshape(n_coarse, r).mean(axis=1)


def burgers_self_convergence(
    cells: list[int],
    T_final: float = 0.3,
    xi: float = 1.0 / 3.0,
    source_fn: Callable[[float], float] | None = lambda t: 1.0,
    scheme: str = "weno5",
    on_interface: str = "downwind",
    cfl: float = 0.2,
    refine: int = 8,
) -> dict:
    r"""
    Self-convergence study for the singular Burgers problem.

    Since there is no analytical solution once the source builds a shock,
    accuracy is measured against a reference computed on a mesh ``refine``
    times finer than the finest mesh in ``cells`` (following Suarez,
    Jacobs & Don 2014).  The reference is averaged down onto each coarse
    mesh, and the ``L1`` and ``L_inf`` cell-average errors are reported.

    All meshes use the same ``xi``; whether ``xi`` lands on an interface
    depends on divisibility, exactly as in the linear case.

    Returns
    -------
    dict with keys ``cells``, ``L1``, ``Linf`` (lists aligned with
    ``cells``).
    """
    n_max = max(cells)
    n_ref = n_max * refine

    def run(n):
        h = 1.0 / n
        dt = cfl * h / np.sqrt(2.0)          # speed ~ sqrt(2) at steady state
        N = int(round(T_final / dt)) + 1
        return solve_burgers_singular(
            M=n + 1, N=N, T_final=T_final, xi=xi,
            source_fn=source_fn, scheme=scheme, on_interface=on_interface,
        )["u"]

    u_ref = run(n_ref)

    L1, Linf = [], []
    for n in cells:
        u = run(n)
        u_ref_c = _sample_to_coarse(u_ref, n_ref, n)
        h = 1.0 / n
        L1.append(float(np.sum(np.abs(u - u_ref_c)) * h))
        Linf.append(float(np.max(np.abs(u - u_ref_c))))

    return {"cells": list(cells), "L1": L1, "Linf": Linf}


__all__ = [
    "solve_burgers_singular",
    "burgers_self_convergence",
]
