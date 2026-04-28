"""
Regression tests for the semi-implicit (RK3 + Crank-Nicolson) advection
solver.

These tests reproduce thesis Table 6.4 (Türk, 2016) using the same time
step (dt = 5e-4, i.e. N = 1001) as the published numbers.  Reproducing
the published values with the published time step is the strongest
end-to-end validation of the scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.advection import (
    solve_advection_singular,
    solve_advection_singular_CR,
)

#: Thesis Table 6.4 (Türk 2016), values to two significant figures.
_THESIS_TABLE_6_4: dict[int, float] = {
    21: 3.54e-2,
    81: 8.56e-3,
    321: 2.10e-3,
}


@pytest.mark.parametrize("M, expected_L1", _THESIS_TABLE_6_4.items())
def test_thesis_table_6_4_semi_implicit(M: int, expected_L1: float) -> None:
    """
    Reproduce thesis Table 6.4 to two significant digits using the SAME
    time step as the thesis (dt = 5e-4, N = 1001).
    """
    result = solve_advection_singular_CR(M=M, N=1001, T_final=0.5)
    L1 = result["L1_err_face"]
    np.testing.assert_allclose(L1, expected_L1, rtol=0.01)


def test_explicit_and_implicit_agree_at_fine_dt() -> None:
    """
    With a small enough time step, explicit RK3 and semi-implicit
    RK3+CN must give numerically indistinguishable results, since the
    spatial discretization dominates and the temporal scheme makes
    almost no difference at sub-CFL time steps.
    """
    r_exp = solve_advection_singular(M=81, N=5001, T_final=0.5)
    r_imp = solve_advection_singular_CR(M=81, N=5001, T_final=0.5)
    diff = float(np.max(np.abs(r_exp["u"] - r_imp["u"])))
    # Both schemes should give the same answer to better than 0.5%.
    assert diff < 5e-3


def test_first_order_convergence_at_jump_implicit() -> None:
    """
    With a discontinuous solution the L1 error of the semi-implicit
    scheme converges at first order in space.
    """
    L1: list[float] = []
    for M in [21, 81, 321]:
        r = solve_advection_singular_CR(M=M, N=1001, T_final=0.5)
        L1.append(r["L1_err_face"])
    # Each refinement (M -> 4M) should reduce L1 by a factor of about 4.
    assert L1[0] / L1[1] > 3.5
    assert L1[1] / L1[2] > 3.5
