"""
Convergence test for the WENO5 reconstruction module on smooth data.

We reconstruct the cell averages of u(x) = sin(pi*x) on a sequence of
uniform meshes and check that the L_inf reconstruction error decays at
the expected fifth-order rate.

This is a regression test: a measured rate below ~4.5 indicates a bug
or a change in the reconstruction (e.g. different smoothness indicators
or linear weights).
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.weno5 import reconstruct


def _setup_problem(M: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Set up a uniform mesh on [0, 2] with cell averages of sin(pi*x)."""
    n = M - 1
    h = 2.0 / n
    x = np.linspace(h / 2, 2.0 - h / 2, n)
    # Exact cell averages of sin(pi*x):
    #   (cos(pi*(x - h/2)) - cos(pi*(x + h/2))) / (pi * h)
    u_avg = (np.cos(np.pi * (x - h / 2))
             - np.cos(np.pi * (x + h / 2))) / (np.pi * h)
    return u_avg, x, h


def test_omega_sum_to_one() -> None:
    """Nonlinear weights at every cell must sum to 1."""
    u, _, _ = _setup_problem(M=81)
    _, omega = reconstruct(u)
    np.testing.assert_allclose(omega.sum(axis=0), 1.0, atol=1e-14)


def test_omega_nonnegative() -> None:
    """Nonlinear weights are convex combinations and so must be >= 0."""
    u, _, _ = _setup_problem(M=81)
    _, omega = reconstruct(u)
    assert (omega >= 0.0).all()


@pytest.mark.parametrize("M", [21, 41, 81, 161, 321])
def test_reconstruction_error_decreases(M: int) -> None:
    """
    Sanity check: the L_inf error at right interfaces is small and
    decreases with mesh refinement.  We do not pin a specific value
    here -- ``test_fifth_order_convergence_rate`` does that.
    """
    u, x, h = _setup_problem(M=M)
    v_half, _ = reconstruct(u)
    v_exact = np.sin(np.pi * (x + h / 2))
    err = np.max(np.abs(v_half - v_exact))
    # Error must be at least as good as second order on the coarsest mesh.
    assert err < 5.0 * h ** 2


def test_fifth_order_convergence_rate() -> None:
    """
    Reconstruct sin(pi*x) on doubling meshes and verify that the L_inf
    error decays with rate close to 5 in the asymptotic regime.

    On a smooth solution and for h small enough, WENO5 collapses to its
    optimal linear combination and is fifth-order accurate.
    """
    rates: list[float] = []
    M_values = [21, 41, 81, 161, 321]
    prev_err: float | None = None
    for M in M_values:
        u, x, h = _setup_problem(M=M)
        v_half, _ = reconstruct(u)
        v_exact = np.sin(np.pi * (x + h / 2))
        err = float(np.max(np.abs(v_half - v_exact)))
        if prev_err is not None:
            rates.append(np.log2(prev_err / err))
        prev_err = err

    # The first refinement may be pre-asymptotic; require all later
    # rates to be at least 4.5 (well clear of fourth order).
    assert all(r > 4.5 for r in rates[1:]), (
        f"Expected 5th-order convergence; observed rates: {rates}"
    )
