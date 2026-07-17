"""
Example 6 — The injection-side effect for a nonlinear flux (Burgers).

Repeats the interface-alignment study of Example 5, but for the
inviscid Burgers equation

    u_t + (u^2/2)_x = delta(x - 1/3),   u(x, 0) = 0.

Since Burgers has no closed-form solution once the source builds a
shock, accuracy is measured by self-convergence against a fine-mesh
reference (following Suarez, Jacobs & Don 2014).

The contrast with the linear case (Example 5) is the point:

  * Linear advection: the jump sits at x = 1/3 and STAYS there.  When
    the mesh is aligned and the source is injected downwind, order
    reduction disappears entirely and the L1 gain is ~100x.

  * Burgers: the shock MOVES (speed ~ 1/sqrt(2)), so the initial
    alignment is immediately lost.  Injecting downwind still helps --
    the max-norm error is consistently smaller -- but the dramatic
    order recovery of the linear case does not occur.

In other words: the injection side always matters, but the full benefit
of alignment requires the discontinuity to stay aligned.

Run:
    python examples/06_burgers_alignment.py
"""

from __future__ import annotations

from weno_singular.burgers import burgers_self_convergence


def main() -> None:
    print("=" * 72)
    print("Example 6: Burgers, Dirac source on a cell face (xi = 1/3, 3 | n)")
    print("u_t + (u^2/2)_x = delta(x - 1/3),  self-convergence, T = 0.3")
    print("=" * 72)

    cells = [30, 60, 120, 240]
    up = burgers_self_convergence(cells, T_final=0.3, on_interface="upwind")
    dn = burgers_self_convergence(cells, T_final=0.3, on_interface="downwind")

    print(f"{'cells':>6} | {'upwind L1':>11} {'Linf':>8} "
          f"| {'downwind L1':>12} {'Linf':>8} | {'Linf ratio':>10}")
    print("-" * 72)
    for i, n in enumerate(cells):
        ratio = up["Linf"][i] / dn["Linf"][i]
        print(f"{n:>6} | {up['L1'][i]:>11.4e} {up['Linf'][i]:>8.4f} "
              f"| {dn['L1'][i]:>12.4e} {dn['Linf'][i]:>8.4f} | {ratio:>9.1f}x")

    print()
    print("The max-norm error is consistently ~3x smaller downwind. Unlike")
    print("the linear case (Example 5), order reduction is NOT removed: the")
    print("Burgers shock moves, so the source no longer stays aligned with")
    print("the discontinuity. The injection side still matters; alignment")
    print("alone is no longer enough.")


if __name__ == "__main__":
    main()
