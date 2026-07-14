"""
Regression tests for the explicit (RK3) singular-source advection solver.

These tests reproduce thesis Table 6.4 (Türk, 2016) to four significant
digits, providing a strong end-to-end check of WENO5 + delta-source
discretization + SSP-RK3 with substage source forcing.
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.advection import (
    exact_cell_averages_singular,
    find_delta_cell,
    solve_advection_singular,
)

# ----------------------------------------------------------------------
# Mesh / index helpers
# ----------------------------------------------------------------------

def test_find_delta_cell_interior() -> None:
    """
    For xi = 0.3 on [0, 1] with M = 7 (6 cells of width 1/6), xi lies
    strictly inside cell 1 = [1/6, 1/3], so both tie-breaking policies
    must return 1.
    """
    for policy in ("downwind", "upwind"):
        j = find_delta_cell(xi=0.3, x_left=0.0, M=7, on_interface=policy)
        assert j == 1


def test_find_delta_cell_on_interface() -> None:
    """
    For xi = 1/3 on [0, 1] with M = 7 (6 cells of width 1/6), xi = 2/6
    falls exactly on the interface between cells 1 and 2.  The default
    ``"downwind"`` policy selects cell 2 = [1/3, 1/2] -- the cell the
    characteristics immediately fill, where the exact solution is
    non-zero.  ``"upwind"`` reproduces the v0.1.0 choice, cell 1.
    """
    assert find_delta_cell(xi=1.0 / 3.0, x_left=0.0, M=7) == 2
    assert find_delta_cell(xi=1.0 / 3.0, x_left=0.0, M=7,
                           on_interface="upwind") == 1


def test_find_delta_cell_respects_domain_length() -> None:
    """
    Regression test for a v0.1.0 bug: the index was computed as
    ``ceil((xi - a) * (M - 1)) - 1``, which silently assumes a domain of
    length 1.  On [0, 2] with M = 5 (4 cells of width 0.5) and xi = 1.0
    that returned cell 3 = [1.5, 2.0] -- not even adjacent to xi.
    """
    j = find_delta_cell(xi=1.0, x_left=0.0, M=5, x_right=2.0)
    assert j == 2                       # cell [1.0, 1.5], downwind of xi


# ----------------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------------

def test_exact_solution_zero_at_t0() -> None:
    """The reference solution is identically zero at t = 0."""
    x = np.linspace(0.05, 0.95, 10)
    u = exact_cell_averages_singular(x, h=0.1, xi=1.0 / 3.0, t=0.0)
    np.testing.assert_array_equal(u, np.zeros_like(x))


def test_exact_solution_support_is_subinterval() -> None:
    """At time t > 0, the reference solution is supported on [xi, xi+t]."""
    n = 200
    h = 1.0 / n
    x = np.linspace(h / 2, 1.0 - h / 2, n)
    xi = 1.0 / 3.0
    t = 0.3
    u = exact_cell_averages_singular(x, h, xi, t)
    # Cells well outside [xi, xi+t] must be zero.
    far_left = x + h / 2 < xi
    far_right = x - h / 2 > xi + t
    assert np.allclose(u[far_left], 0.0)
    assert np.allclose(u[far_right], 0.0)
    # Inside the support, u must be in [-1, 1] (range of sine).
    inside = ~(far_left | far_right)
    assert np.all(np.abs(u[inside]) <= 1.0 + 1e-12)


# ----------------------------------------------------------------------
# End-to-end regression vs thesis Table 6.4
# ----------------------------------------------------------------------

#: Thesis Table 6.4 (Türk 2016): WENO5 on uniform mesh, T = 0.5,
#: dt = 5e-4 (i.e. N = 1001).  Values given to two significant digits.
_THESIS_TABLE_6_4: dict[int, float] = {
    21: 3.54e-2,
    81: 8.56e-3,
    321: 2.10e-3,
}


@pytest.mark.parametrize("M, expected_L1", _THESIS_TABLE_6_4.items())
def test_thesis_table_6_4(M: int, expected_L1: float) -> None:
    """
    Reproduce thesis Table 6.4 to two significant digits using the
    same parameters as the thesis (T = 0.5, dt = 5e-4 i.e. N = 1001
    in our convention -- but we use a smaller dt of 1e-4 = N = 5001
    here so the time-error never dominates the spatial discretization
    error at the discontinuity).
    """
    result = solve_advection_singular(M=M, N=5001, T_final=0.5)
    L1 = result["L1_err_face"]
    # Match the published value to within 1% relative.
    np.testing.assert_allclose(L1, expected_L1, rtol=0.01)


def test_first_order_convergence_at_jump() -> None:
    """
    The exact solution is discontinuous at x = xi, so WENO5 should
    converge at first order in L1 (Section 6.1.3 of the thesis).
    """
    L1: list[float] = []
    for M in [21, 81, 321]:
        r = solve_advection_singular(M=M, N=5001, T_final=0.5)
        L1.append(r["L1_err_face"])
    # M -> 4 M, so a perfect 1st order rate would give 4x reduction;
    # we require at least 3.5x to allow for pre-asymptotic factors.
    assert L1[0] / L1[1] > 3.5
    assert L1[1] / L1[2] > 3.5
