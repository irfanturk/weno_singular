"""
Example 2 — Singular source advection (explicit RK3).

Solves the canonical singular-source test problem from
Türk (2016, Example 6.1.3):

    u_t + u_x = sin(pi t) * delta(x - 1/3),    0 <= x <= 1, periodic,
    u(x, 0) = 0.

The exact solution is

    u(x, t) = sin(pi*(1/3 + t - x))    for  1/3 <= x <= 1/3 + t,
            = 0                        otherwise.

The solution has a STATIONARY jump discontinuity at x = 1/3, so WENO5
can deliver
at best first-order convergence in L1 (Table 6.4 of the thesis).

This example reproduces Table 6.4 to four significant digits using an
explicit SSP-RK3 time integrator with a small time step.

Run:
    python examples/02_singular_source_explicit.py
"""

from __future__ import annotations

import numpy as np

from weno_singular.advection import solve_advection_singular


def main() -> None:
    print("=" * 75)
    print("Example 2: singular source, EXPLICIT RK3")
    print("u_t + u_x = sin(pi*t) * delta(x - 1/3),  u(x, 0) = 0,  T = 0.5")
    print("=" * 75)
    print(f"{'M':>5}  {'L_inf':>13}  {'L1':>13}  {'rate L1':>8}  thesis L1")
    print("-" * 75)

    # Thesis Table 6.4 (Türk 2016).
    thesis = {21: 3.54e-2, 81: 8.56e-3, 321: 2.10e-3}

    prev_L1: float | None = None
    for M in [21, 81, 321]:
        # Use a small dt so the time-error never dominates the spatial
        # discretization error at the discontinuity.
        result = solve_advection_singular(
            M=M, N=5001, T_final=0.5,
            x_left=0.0, x_right=1.0, xi=1.0 / 3.0,
        )
        L_inf = result["max_err_inf"]
        L1 = result["L1_err_face"]
        if prev_L1 is None:
            rate_str = "    -"
        else:
            rate_str = f"{np.log(prev_L1 / L1) / np.log(4.0):>5.2f}"
        print(f"{M:>5}  {L_inf:>13.4e}  {L1:>13.4e}  {rate_str}    {thesis[M]:.2e}")
        prev_L1 = L1

    print()
    print("Each refinement (M -> 4 M) should reduce L1 by a factor close to 4")
    print("(first-order convergence at the stationary jump discontinuity).")


if __name__ == "__main__":
    main()
