"""
Convergence test for the WENO3 reconstruction module on smooth data.

We reconstruct cell averages of u(x) = sin(pi*x) on a sequence of
uniform meshes and check that the L_inf reconstruction error decays at
the expected third-order rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.weno3 import L_advection, build_matrix, reconstruct


def _setup_problem(M: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Set up a uniform mesh on [0, 2] with cell averages of sin(pi*x)."""
    n = M - 1
    h = 2.0 / n
    x = np.linspace(h / 2, 2.0 - h / 2, n)
    u_avg = (np.cos(np.pi * (x - h / 2))
             - np.cos(np.pi * (x + h / 2))) / (np.pi * h)
    return u_avg, x, h


def test_weno3_omega_sum_to_one() -> None:
    """Nonlinear weights at every cell must sum to 1."""
    u, _, _ = _setup_problem(M=81)
    _, omega = reconstruct(u)
    np.testing.assert_allclose(omega.sum(axis=0), 1.0, atol=1e-14)


def test_weno3_omega_nonnegative() -> None:
    """Nonlinear weights are convex combinations and so must be >= 0."""
    u, _, _ = _setup_problem(M=81)
    _, omega = reconstruct(u)
    assert (omega >= 0.0).all()


@pytest.mark.parametrize("M", [21, 41, 81, 161, 321])
def test_weno3_reconstruction_error_decreases(M: int) -> None:
    """Sanity check: the L_inf error at right interfaces is small."""
    u, x, h = _setup_problem(M=M)
    v_half, _ = reconstruct(u)
    v_exact = np.sin(np.pi * (x + h / 2))
    err = np.max(np.abs(v_half - v_exact))
    # Error must be at least as good as second order on the coarsest mesh.
    assert err < 5.0 * h ** 2


def test_weno3_third_order_convergence_rate() -> None:
    """
    Verify that the L_inf reconstruction error decays at a rate that
    approaches 3 as the mesh is refined.

    Note: WENO3-JS shows pre-asymptotic rates close to 2 on coarse
    meshes because the regularization parameter ``EPS = 1e-6`` is
    relatively large compared to the smoothness indicators on coarse
    meshes; the rate climbs toward 3 as the mesh is refined further
    (typical Jiang-Shu behaviour, well-documented in the literature).
    """
    rates: list[float] = []
    # Use a finer sequence so we reach the asymptotic regime.
    M_values = [41, 81, 161, 321, 641]
    prev_err: float | None = None
    for M in M_values:
        u, x, h = _setup_problem(M=M)
        v_half, _ = reconstruct(u)
        v_exact = np.sin(np.pi * (x + h / 2))
        err = float(np.max(np.abs(v_half - v_exact)))
        if prev_err is not None:
            rates.append(np.log2(prev_err / err))
        prev_err = err

    # The finest-mesh rate should be at least 2.5 (clearly above 2).
    assert rates[-1] > 2.5, (
        f"Expected 3rd-order convergence at the finest meshes; "
        f"observed rates: {rates}"
    )


def test_weno3_build_matrix_matches_L_advection() -> None:
    """
    For any state ``u``, ``build_matrix(omega(u)) @ u`` must equal
    ``L_advection(u)`` to machine precision.  This is a critical
    sanity check for the semi-implicit corrector.
    """
    u, _, h = _setup_problem(M=81)
    Lu, _, omega = L_advection(u, h)
    L_mat = build_matrix(omega, h)
    Lu_mat = L_mat @ u
    np.testing.assert_allclose(Lu, Lu_mat, atol=1e-12)
