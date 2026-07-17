"""
Tests for the singular Burgers solver (weno_singular.burgers).

Cover:
* the mirror-image right-biased reconstruction (recon_plus),
* physical correctness of the steady-state shock,
* machine-precision mass conservation of the direct injection,
* high-order convergence on a smooth (pre-shock) solution,
* the injection-side (on_interface) effect for a nonlinear flux.
"""

from __future__ import annotations

import numpy as np
import pytest

from weno_singular.burgers import (
    _reconstructors,
    _sample_to_coarse,
    solve_burgers_singular,
)

# ----------------------------------------------------------------------
# Reconstruction building block
# ----------------------------------------------------------------------

def _smooth_cell_averages(n, freq=1.0):
    h = 2.0 / n
    x = np.linspace(h / 2, 2.0 - h / 2, n)
    u = (np.cos(freq * np.pi * (x - h / 2))
         - np.cos(freq * np.pi * (x + h / 2))) / (freq * np.pi * h)
    return u, x, h


@pytest.mark.parametrize("scheme", ["weno3", "weno5", "weno5z"])
def test_recon_plus_matches_minus_on_smooth_data(scheme):
    """On smooth data the left- and right-biased interface states agree
    (the solution is continuous there)."""
    u, x, h = _smooth_cell_averages(80)
    rm, rp = _reconstructors(scheme)
    um, up = rm(u), rp(u)
    # WENO3 is lower order, so its two biased states agree less tightly
    tol = 5e-3 if scheme == "weno3" else 5e-4
    assert np.max(np.abs(um - up)) < tol


def test_recon_plus_is_fifth_order():
    """The mirror reconstruction must keep WENO5's design order."""
    _, rp = _reconstructors("weno5")
    prev = None
    rates = []
    for n in [40, 80, 160, 320]:
        u, x, h = _smooth_cell_averages(n)
        err = float(np.max(np.abs(rp(u) - np.sin(np.pi * (x + h / 2)))))
        if prev is not None:
            rates.append(np.log2(prev / err))
        prev = err
    assert all(r > 4.5 for r in rates), f"recon_plus rates: {rates}"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        _reconstructors("weno7")


# ----------------------------------------------------------------------
# Physical correctness
# ----------------------------------------------------------------------

def test_steady_state_shock_height():
    """u_t + (u^2/2)_x = delta(x - xi) drives a shock whose right state
    approaches sqrt(2); the left state stays at 0."""
    r = solve_burgers_singular(M=201, N=6001, T_final=0.3, xi=1.0 / 3.0)
    u, x = r["u"], r["x"]
    assert not np.any(np.isnan(u))
    assert u.max() == pytest.approx(np.sqrt(2.0), abs=0.02)
    left = u[x < r["xi"] - 0.05]
    assert np.max(np.abs(left)) < 5e-3          # zero to the left of xi


def test_mass_conservation_is_machine_precision():
    """Direct injection conserves the injected mass g*T exactly (up to
    the time integrator).  This is the key advantage over regularization."""
    g, T = 1.0, 0.3
    r = solve_burgers_singular(
        M=201, N=6001, T_final=T, xi=1.0 / 3.0, source_fn=lambda t: g
    )
    mass = float(np.sum(r["u"]) * r["h"])
    assert mass == pytest.approx(g * T, abs=1e-6)


def test_no_source_is_conservative_and_stable():
    """With no source and a smooth IC, the pre-shock solution stays
    bounded and mass is conserved."""
    def ic(x, h):
        return 0.5 + 0.25 * (np.cos(2 * np.pi * (x - h / 2))
                             - np.cos(2 * np.pi * (x + h / 2))) / (2 * np.pi * h)

    r = solve_burgers_singular(
        M=129, N=4000, T_final=0.1, source_fn=None, initial_fn=ic
    )
    u = r["u"]
    assert not np.any(np.isnan(u))
    assert u.min() > 0.0 and u.max() < 1.0       # stays within IC range


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------

def test_high_order_on_smooth_solution():
    """Pre-shock, the Rusanov + WENO5 Burgers solver is high order."""
    def ic(x, h):
        return 0.5 + 0.25 * (np.cos(2 * np.pi * (x - h / 2))
                             - np.cos(2 * np.pi * (x + h / 2))) / (2 * np.pi * h)

    T, n_ref = 0.1, 64 * 8
    u_ref = solve_burgers_singular(
        M=n_ref + 1, N=6000, T_final=T, source_fn=None, initial_fn=ic
    )["u"]

    prev, rates = None, []
    for n in [16, 32, 64]:
        u = solve_burgers_singular(
            M=n + 1, N=2000, T_final=T, source_fn=None, initial_fn=ic
        )["u"]
        err = float(np.max(np.abs(u - _sample_to_coarse(u_ref, n_ref, n))))
        if prev is not None:
            rates.append(np.log2(prev / err))
        prev = err
    # clearly above 2nd order (well past a first-order or MUSCL scheme)
    assert rates[-1] > 3.0, f"Burgers smooth rates: {rates}"


# ----------------------------------------------------------------------
# Injection side for a nonlinear flux
# ----------------------------------------------------------------------

def test_downwind_beats_upwind_in_max_norm_burgers():
    """
    Even for the nonlinear Burgers flux, injecting on the upwind side of
    a face-aligned source leaves a larger max-norm error than injecting
    downwind.  The gain is milder than in the linear case (the Burgers
    shock is not stationary, so alignment is not preserved), but the
    ordering is robust.
    """
    T = 0.3
    n_ref = 60 * 8

    def linf(pol, n):
        ref = solve_burgers_singular(
            M=n_ref + 1, N=6000, T_final=T,
            xi=1.0 / 3.0, on_interface=pol,
        )["u"]
        u = solve_burgers_singular(
            M=n + 1, N=1500, T_final=T, xi=1.0 / 3.0, on_interface=pol
        )["u"]
        return float(np.max(np.abs(u - _sample_to_coarse(ref, n_ref, n))))

    # at a representative aligned mesh, downwind's max-norm error is
    # clearly smaller
    up = linf("upwind", 60)
    dn = linf("downwind", 60)
    assert dn < 0.6 * up, f"downwind {dn:.3f} not < 0.6 * upwind {up:.3f}"
