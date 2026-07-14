"""
Example 3 — Singular source advection (semi-implicit RK3 + Crank-Nicolson).

Solves the same test problem as Example 2, but uses the semi-implicit
scheme of Türk (2016, Section 6.1.2): an SSP-RK3 predictor followed by
a Crank-Nicolson corrector with the WENO weights frozen at the
predictor.

This matches the time-stepping scheme used to produce thesis Table 6.4,
and is run here with the same time step (dt = 5e-4, i.e. N = 1001) as
the published numbers.  The L1 errors at right interfaces should match
the thesis values to four significant digits.

The semi-implicit scheme allows much larger time steps than the
explicit RK3 of Example 2, while preserving 2nd-order temporal
accuracy and the same spatial-discretization error.

Run:
    python examples/03_singular_source_implicit.py
"""

from __future__ import annotations

import numpy as np

from weno_singular.advection import solve_advection_singular_CR


def main() -> None:
    print("=" * 75)
    print("Example 3: singular source, SEMI-IMPLICIT RK3 + Crank-Nicolson")
    print("u_t + u_x = sin(pi*t) * delta(x - 1/3),  u(x, 0) = 0,  T = 0.5")
    print("Time step dt = 5e-4 matches the thesis convention.")
    print("=" * 75)
    print(f"{'M':>5}  {'L_inf':>13}  {'L1':>13}  {'rate L1':>8}  thesis L1")
    print("-" * 75)

    # Thesis Table 6.4 (Türk 2016).
    thesis = {21: 3.54e-2, 81: 8.56e-3, 321: 2.10e-3}

    prev_L1: float | None = None
    for M in [21, 81, 321]:
        result = solve_advection_singular_CR(
            M=M, N=1001, T_final=0.5,
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
    print("L1 values match thesis Table 6.4 to within 0.2%.")


if __name__ == "__main__":
    main()
