"""
Example 4 — Reconstruction schemes compared on the singular-source problem.

Solves

    u_t + u_x = sin(pi t) * delta(x - 1/3),   0 <= x <= 1, periodic,
    u(x, 0) = 0,   T = 0.5,

with three spatial reconstructions on the same mesh, the same time
integrator (RK3 + Crank-Nicolson) and the same time step, so that the
reconstruction is the only variable:

    weno3   WENO3-JS  (Liu, Osher & Chan 1994)
    weno5   WENO5-JS  (Jiang & Shu 1996)
    weno5z  WENO5-Z   (Borges et al. 2008)

The first two columns reproduce Tables 6.3 and 6.4 of Türk (2016).

Take-away: the exact solution has a stationary jump at x = 1/3, so every
scheme is limited to first-order convergence in L1 there.  Improving the
*reconstruction* (JS -> Z, or 3rd -> 5th order) buys almost nothing: the
error is controlled by the discontinuity, not by the stencil.

Run:
    python examples/04_scheme_comparison.py
"""

from __future__ import annotations

import numpy as np

from weno_singular.advection import solve_advection_singular_CR

LOG4 = np.log(4.0)

#: Published L1 errors (Türk 2016), face-based metric of eq. (6.1.19).
THESIS = {
    "weno3": {20: 3.74e-2, 80: 9.12e-3, 320: 2.22e-3},   # Table 6.3
    "weno5": {20: 3.54e-2, 80: 8.56e-3, 320: 2.10e-3},   # Table 6.4
}


def main() -> None:
    schemes = ["weno3", "weno5", "weno5z"]
    cells = [20, 80, 320]

    print("=" * 78)
    print("Example 4: reconstruction schemes, singular source, RK3 + Crank-Nicolson")
    print("u_t + u_x = sin(pi t) delta(x - 1/3),  u(x,0) = 0,  T = 0.5,  dt = 5e-4")
    print("=" * 78)
    print(f"{'scheme':>7} {'cells':>6} {'L1 (face)':>12} {'order':>6} "
          f"{'L1 (cell)':>12} {'order':>6} {'thesis':>10}")
    print("-" * 78)

    for scheme in schemes:
        prev_f = prev_c = None
        for n in cells:
            r = solve_advection_singular_CR(
                M=n + 1, N=1001, T_final=0.5, scheme=scheme
            )
            ef, ec = r["L1_err_face"], r["L1_err_cell"]
            of = "     -" if prev_f is None else f"{np.log(prev_f / ef) / LOG4:>6.2f}"
            oc = "     -" if prev_c is None else f"{np.log(prev_c / ec) / LOG4:>6.2f}"
            ref = THESIS.get(scheme, {}).get(n)
            ref_s = f"{ref:>10.2e}" if ref else f"{'-':>10}"
            print(f"{scheme:>7} {n:>6} {ef:>12.4e} {of} {ec:>12.4e} {oc} {ref_s}")
            prev_f, prev_c = ef, ec
        print()

    print("All three schemes converge at first order: the stationary jump at")
    print("x = 1/3, not the reconstruction stencil, sets the error.")


if __name__ == "__main__":
    main()
