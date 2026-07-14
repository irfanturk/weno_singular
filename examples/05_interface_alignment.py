"""
Example 5 — Where should a Dirac source sitting on a cell face be injected?

Türk (2016) places the singular source in "the cell which contains the
point x = xi".  On a uniform mesh of n cells over [0, 1] with xi = 1/3,
that cell is unambiguous **unless** n is a multiple of 3, in which case
xi falls exactly on an interface and two cells share the point.

The choice matters.  For u_t + u_x = g(t) delta(x - xi) the exact
solution vanishes for x < xi and jumps to g at xi+.  Injecting into the
*upwind* cell [xi - h, xi] deposits mass where the exact solution is
identically zero; the characteristics do carry it out again, but a
residual O(1) error in the max norm survives every refinement.
Injecting into the *downwind* cell [xi, xi + h] -- the one the
characteristics immediately fill -- removes it.

This example sweeps aligned meshes (3 | n) and reports both policies.

Run:
    python examples/05_interface_alignment.py
"""

from __future__ import annotations

import numpy as np

from weno_singular.advection import solve_advection_singular


def main() -> None:
    cells = [30, 60, 120, 240, 480]
    cfl = 0.1

    print("=" * 76)
    print("Example 5: Dirac source on a cell interface (xi = 1/3, 3 | n)")
    print("WENO5-JS + SSP-RK3.  L1 is the well-posed cell-average metric.")
    print("=" * 76)
    print(f"{'cells':>6} | {'upwind L1':>11} {'ord':>5} {'Linf':>8} "
          f"| {'downwind L1':>12} {'ord':>5} {'Linf':>8}")
    print("-" * 76)

    prev_u = prev_d = None
    for n in cells:
        N = int(round(0.5 / (cfl / n))) + 1
        up = solve_advection_singular(
            M=n + 1, N=N, T_final=0.5, on_interface="upwind"
        )
        dn = solve_advection_singular(
            M=n + 1, N=N, T_final=0.5, on_interface="downwind"
        )
        eu, ed = up["L1_err_cell"], dn["L1_err_cell"]
        ou = "    -" if prev_u is None else f"{np.log2(prev_u / eu):>5.2f}"
        od = "    -" if prev_d is None else f"{np.log2(prev_d / ed):>5.2f}"
        print(f"{n:>6} | {eu:>11.4e} {ou} {up['max_err_inf']:>8.4f} "
              f"| {ed:>12.4e} {od} {dn['max_err_inf']:>8.4f}")
        prev_u, prev_d = eu, ed

    print()
    print("upwind  : L1 order 1, and Linf stalls at ~1.0 (the full jump height)")
    print("          -- the max-norm error never converges.")
    print("downwind: L1 order climbs past 1.5, Linf converges.  Same scheme,")
    print("          same mesh, same time step: only the injection cell differs.")


if __name__ == "__main__":
    main()
