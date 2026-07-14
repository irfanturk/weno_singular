"""
Tests for the v0.2.0 features:

* scheme selection (``weno3`` / ``weno5`` / ``weno5z``) in the solvers,
* reproduction of thesis Table 6.3 (WENO3), which v0.1.0 could not do,
* the WENO-Z nonlinear weights,
* the delta-cell tie-breaking policy when xi lies on a cell interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.advection import (
    find_delta_cell,
    solve_advection_singular,
    solve_advection_singular_CR,
)
from weno_singular.weno5 import (
    GAMMA,
    nonlinear_weights_z,
    reconstruct,
    smoothness_indicators,
)

LOG4 = np.log(4.0)

#: Thesis Table 6.3 (Türk 2016): semi-implicit WENO3, uniform mesh.
_THESIS_TABLE_6_3 = {20: 3.74e-2, 80: 9.12e-3, 320: 2.22e-3}

#: Thesis Table 6.4 (Türk 2016): semi-implicit WENO5, uniform mesh.
_THESIS_TABLE_6_4 = {20: 3.54e-2, 80: 8.56e-3, 320: 2.10e-3}


# ----------------------------------------------------------------------
# Scheme selection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["weno3", "weno5", "weno5z"])
def test_all_schemes_run(scheme: str) -> None:
    """Every registered scheme must run end to end and report its name."""
    r = solve_advection_singular_CR(M=81, N=1001, T_final=0.5, scheme=scheme)
    assert r["scheme"] == scheme
    assert np.isfinite(r["u"]).all()


def test_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="unknown scheme"):
        solve_advection_singular(M=81, N=1001, scheme="weno7")


@pytest.mark.parametrize("n, expected_L1", _THESIS_TABLE_6_3.items())
def test_thesis_table_6_3_weno3(n: int, expected_L1: float) -> None:
    """
    Reproduce thesis Table 6.3 (WENO3).  This table was unreachable in
    v0.1.0 because the solvers hard-wired the WENO5 reconstruction.
    """
    r = solve_advection_singular_CR(M=n + 1, N=1001, T_final=0.5, scheme="weno3")
    np.testing.assert_allclose(r["L1_err_face"], expected_L1, rtol=0.01)


@pytest.mark.parametrize("n, expected_L1", _THESIS_TABLE_6_4.items())
def test_thesis_table_6_4_unchanged(n: int, expected_L1: float) -> None:
    """Regression guard: the v0.2.0 changes must not perturb Table 6.4."""
    r = solve_advection_singular_CR(M=n + 1, N=1001, T_final=0.5, scheme="weno5")
    np.testing.assert_allclose(r["L1_err_face"], expected_L1, rtol=0.01)


def test_first_order_at_jump_for_every_scheme() -> None:
    """
    The stationary jump at x = 1/3 caps every reconstruction at first
    order in L1 -- the central observation of Section 6.1.3 of the thesis.
    """
    for scheme in ["weno3", "weno5", "weno5z"]:
        errs = [
            solve_advection_singular_CR(
                M=n + 1, N=1001, T_final=0.5, scheme=scheme
            )["L1_err_face"]
            for n in (20, 80, 320)
        ]
        for a, b in zip(errs, errs[1:]):
            order = np.log(a / b) / LOG4
            assert 0.9 < order < 1.2, f"{scheme}: order {order:.3f} not first order"


# ----------------------------------------------------------------------
# WENO-Z weights
# ----------------------------------------------------------------------

def test_weno_z_weights_sum_to_one_and_are_nonnegative() -> None:
    n = 80
    h = 2.0 / n
    x = np.linspace(h / 2, 2.0 - h / 2, n)
    u = (np.cos(np.pi * (x - h / 2)) - np.cos(np.pi * (x + h / 2))) / (np.pi * h)
    omega = nonlinear_weights_z(smoothness_indicators(u))
    np.testing.assert_allclose(omega.sum(axis=0), 1.0, atol=1e-14)
    assert (omega >= 0.0).all()


def test_weno_z_is_fifth_order_on_smooth_data() -> None:
    """WENO-Z must retain the design order on smooth solutions."""
    rates: list[float] = []
    prev = None
    for n in [40, 80, 160, 320]:
        h = 2.0 / n
        x = np.linspace(h / 2, 2.0 - h / 2, n)
        u = (np.cos(np.pi * (x - h / 2)) - np.cos(np.pi * (x + h / 2))) / (np.pi * h)
        err = float(np.max(np.abs(reconstruct(u, variant="z")[0]
                                  - np.sin(np.pi * (x + h / 2)))))
        if prev is not None:
            rates.append(np.log2(prev / err))
        prev = err
    assert all(r > 4.5 for r in rates), f"WENO-Z rates: {rates}"


def test_weno_z_beats_js_on_smooth_data() -> None:
    """On smooth data the Z weights approach the optimal ones faster."""
    n = 80
    h = 2.0 / n
    x = np.linspace(h / 2, 2.0 - h / 2, n)
    u = (np.cos(np.pi * (x - h / 2)) - np.cos(np.pi * (x + h / 2))) / (np.pi * h)
    exact = np.sin(np.pi * (x + h / 2))
    err_js = np.max(np.abs(reconstruct(u, variant="js")[0] - exact))
    err_z = np.max(np.abs(reconstruct(u, variant="z")[0] - exact))
    assert err_z < err_js


def test_weno_z_weights_approach_optimal_on_constant_data() -> None:
    """On perfectly smooth (constant) data both weightings give gamma."""
    u = np.full(40, 2.5)
    omega = nonlinear_weights_z(smoothness_indicators(u))
    np.testing.assert_allclose(omega, np.tile(GAMMA[:, None], (1, 40)), atol=1e-12)


# ----------------------------------------------------------------------
# Delta-cell placement
# ----------------------------------------------------------------------

@pytest.mark.parametrize("n", [20, 80, 320])
def test_delta_cell_matches_thesis_when_xi_is_interior(n: int) -> None:
    """
    When xi lies strictly inside a cell -- as in every uniform-mesh
    experiment of the thesis -- the cell index must be floor(xi / h),
    and both tie-breaking policies must agree.
    """
    xi = 1.0 / 3.0
    j_expected = int(np.floor(xi * n))
    for policy in ("downwind", "upwind"):
        j = find_delta_cell(xi, 0.0, n + 1, 1.0, policy)
        assert j == j_expected
    h = 1.0 / n
    assert j_expected * h < xi < (j_expected + 1) * h


@pytest.mark.parametrize("n", [30, 60, 120, 240])
def test_delta_cell_tie_breaking_on_interface(n: int) -> None:
    """When 3 | n, xi = 1/3 sits on an interface; the policies must differ
    by exactly one cell, with downwind picking the cell [xi, xi + h]."""
    xi = 1.0 / 3.0
    j_dn = find_delta_cell(xi, 0.0, n + 1, 1.0, "downwind")
    j_up = find_delta_cell(xi, 0.0, n + 1, 1.0, "upwind")
    assert j_dn == j_up + 1
    h = 1.0 / n
    assert j_dn * h == pytest.approx(xi)          # xi is the LEFT face of j_dn


def test_bad_policy_raises() -> None:
    with pytest.raises(ValueError, match="on_interface"):
        find_delta_cell(1.0 / 3.0, 0.0, 61, 1.0, "sideways")


def test_downwind_placement_converges_in_max_norm() -> None:
    """
    The headline result of Example 5.  With the source in the upwind cell
    the max-norm error stalls at the full jump height (~1) no matter how
    fine the mesh; with the downwind cell it converges.
    """
    linf_up, linf_dn, l1_dn = [], [], []
    for n in (60, 120, 240):
        N = int(round(0.5 / (0.1 / n))) + 1
        linf_up.append(
            solve_advection_singular(
                M=n + 1, N=N, T_final=0.5, on_interface="upwind"
            )["max_err_inf"]
        )
        r = solve_advection_singular(
            M=n + 1, N=N, T_final=0.5, on_interface="downwind"
        )
        linf_dn.append(r["max_err_inf"])
        l1_dn.append(r["L1_err_cell"])

    # Upwind: no convergence at all -- the error is the jump height.
    assert all(e > 0.9 for e in linf_up)
    # Downwind: monotone convergence, and far smaller.
    assert linf_dn[0] > linf_dn[1] > linf_dn[2]
    assert linf_dn[-1] < 0.02
    # ... and better than first order in L1.
    assert np.log2(l1_dn[0] / l1_dn[-1]) / 2 > 1.4


def test_default_mesh_is_not_interface_aligned() -> None:
    """The v0.1.0 default (M = 181, i.e. 180 cells) put xi exactly on a
    face.  The v0.2.0 default must not."""
    r = solve_advection_singular()
    n = len(r["x"])
    assert n % 3 != 0, "default mesh must not align xi = 1/3 with a face"
    assert r["max_err_inf"] < 0.5
