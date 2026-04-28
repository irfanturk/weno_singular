"""
Example 1 — Smooth advection: 5th-order convergence study.

Solves the smooth linear advection problem

    u_t + u_x = 0,    0 <= x <= 2,   periodic BC,
    u(x, 0) = sin(pi * x),

on a sequence of refining uniform meshes and reports the L_inf error
between the WENO5 cell-averaged solution at T = 0.1 and the analytical
solution u(x, t) = sin(pi*(x - t)).

Expected behaviour:  the L_inf error should decay at a fifth-order
rate when the time step is refined together with the mesh, reproducing
the well-known result for WENO5 on smooth solutions.

Run:
    python examples/01_smooth_advection.py
"""

from __future__ import annotations

import numpy as np

from weno_singular.advection import solve_advection_singular


def smooth_initial(x: np.ndarray, h: float) -> np.ndarray:
    """Cell averages of u(x, 0) = sin(pi*x)."""
    return (np.cos(np.pi * (x - h / 2))
            - np.cos(np.pi * (x + h / 2))) / (np.pi * h)


def main() -> None:
    print("=" * 68)
    print("Example 1: smooth linear advection, u_t + u_x = 0")
    print("u(x, 0) = sin(pi*x), exact: u(x, t) = sin(pi*(x - t))")
    print("=" * 68)
    print(f"{'M':>5}  {'h':>10}  {'dt':>10}  {'L_inf error':>15}  {'rate':>6}")
    print("-" * 68)

    T_final = 0.1
    M_values = [21, 41, 81, 161]
    prev_err: float | None = None
    for M in M_values:
        n = M - 1
        h = 2.0 / n
        # Scale dt as h^(5/3) so spatial error dominates and rate is clean.
        dt_target = 0.4 * (0.1) * (h / 0.1) ** (5.0 / 3.0)
        N = int(np.ceil(T_final / dt_target)) + 1
        dt = T_final / (N - 1)

        result = solve_advection_singular(
            M=M,
            N=N,
            T_final=T_final,
            x_left=0.0,
            x_right=2.0,
            source_fn=None,           # no source: pure advection
            initial_fn=smooth_initial,
            track_max_error=False,
        )

        # Compute L_inf error vs the analytical cell averages.
        x = result["x"]
        u_exact = (np.cos(np.pi * (x - h / 2 - T_final))
                   - np.cos(np.pi * (x + h / 2 - T_final))) / (np.pi * h)
        err = float(np.max(np.abs(result["u"] - u_exact)))

        if prev_err is None:
            rate_str = "    -"
        else:
            rate_str = f"{np.log2(prev_err / err):>5.2f}"
        print(f"{M:>5}  {h:>10.3e}  {dt:>10.3e}  {err:>15.4e}  {rate_str}")
        prev_err = err

    print()
    print("Asymptotic rate should approach 5.")


if __name__ == "__main__":
    main()
