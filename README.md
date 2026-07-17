# weno_singular

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19865329.svg)](https://doi.org/10.5281/zenodo.19865329)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**WENO solvers for linear hyperbolic conservation laws with singular source terms.**

A Python implementation of finite-volume WENO schemes for the 1D linear
advection equation on uniform periodic meshes, with first-class support
for Dirac-delta point sources:

$$u_t + u_x = g(t)\delta(x - \xi), \qquad u(x,0) = \varphi(x).$$

Such problems arise wherever a conservation law is forced at a point — a
well in a flow, a point release of a contaminant, a localized reaction
site. Their solutions are **discontinuous at the source**, and that is
what makes them awkward: a formally fifth-order scheme is capped at first
order in $L_1$, and the placement of the source relative to the mesh
turns out to matter as much as the reconstruction stencil.

## Features

- **Three reconstructions**, selectable per solve:
  - `weno3` — WENO3-JS (Liu, Osher & Chan, 1994)
  - `weno5` — WENO5-JS (Jiang & Shu, 1996)
  - `weno5z` — WENO5-Z (Borges et al., 2008)
- **Two time integrators**:
  - SSP-RK3 (Shu & Osher, 1988), fully explicit
  - SSP-RK3 predictor + Crank–Nicolson corrector with frozen WENO
    weights (one sparse solve per step), semi-implicit
- **Singular source terms** treated by direct injection into a single
  cell, exactly mass-conserving to machine precision
- **Explicit control over source placement** when $\xi$ lands on a cell
  interface — see [Source placement](#source-placement-on-a-cell-interface)
- Sparse matrix form of the WENO operator (`build_matrix`), the building
  block of the semi-implicit corrector

## Installation

```bash
pip install weno-singular
```

From source:

```bash
git clone https://github.com/irfanturk/weno_singular.git
cd weno_singular
pip install -e ".[dev]"
```

## Quick start

```python
from weno_singular.advection import solve_advection_singular

# Solve  u_t + u_x = sin(pi t) * delta(x - 1/3)
# on [0, 1] with periodic BCs and u(x, 0) = 0.
result = solve_advection_singular(
    M=182,            # number of cell interfaces (M - 1 = 181 cells)
    N=5001,           # number of time levels
    T_final=0.5,
    xi=1.0 / 3.0,     # source location
    scheme="weno5",   # "weno3" | "weno5" | "weno5z"
)

print(f"L_inf error      : {result['max_err_inf']:.4e}")
print(f"L1 error (cells) : {result['L1_err_cell']:.4e}")
print(f"L1 error (faces) : {result['L1_err_face']:.4e}")
```

`M` counts **interfaces**, so a mesh of `n` cells is `M = n + 1`.
Türk (2016) counts cells; the tables below give both.

## Error metrics

| key | definition | when to use |
|---|---|---|
| `L1_err_cell` | over cell averages | **default**; well posed on any mesh |
| `L1_err_face` | over the reconstruction at right interfaces, eq. (6.1.19) of Türk (2016) | to compare against the published tables |

> **Caveat.** When $\xi$ coincides with a cell interface, `L1_err_face`
> is *ill posed*: the analytical solution is right-continuous at $\xi$,
> with value $g(t)$, while an upwind reconstruction at that face necessarily
> returns the left limit, $0$. Both are correct from their own
> side, and the resulting $O(1)$ discrepancy at a single face injects a
> spurious $O(h)$ term into the norm. Prefer `L1_err_cell` on such meshes.

## Source placement on a cell interface

Türk (2016) injects the source into "the cell which contains the point
$x = \xi$". On a uniform mesh that is unambiguous — *unless* $\xi$ falls
exactly on an interface, in which case two cells share the point.

The choice is not cosmetic. The exact solution vanishes for $x < \xi$ and
jumps to $g$ at $\xi^{+}$, so injecting into the **upwind** cell
$[\xi - h,\ \xi]$ deposits mass where the exact solution is identically
zero. A residual $O(1)$ error in the max norm then survives every
refinement. Injecting into the **downwind** cell $[\xi,\ \xi + h]$ — the
one the characteristics immediately fill — removes it:

| cells | upwind $L_1$ | upwind $L_\infty$ | downwind $L_1$ | downwind $L_\infty$ |
|---:|---:|---:|---:|---:|
| 30 | 6.72e-02 | 0.998 | 2.59e-03 | 0.022 |
| 60 | 3.35e-02 | 1.000 | 8.78e-04 | 0.013 |
| 120 | 1.67e-02 | 1.000 | 2.77e-04 | 0.008 |
| 240 | 8.35e-03 | 1.000 | 8.36e-05 | 0.005 |
| 480 | 4.17e-03 | 1.000 | 2.32e-05 | 0.003 |

(WENO5-JS + SSP-RK3, CFL 0.1, $\xi = 1/3$, cell-average $L_1$.) Same
scheme, same mesh, same time step; only the injection cell differs. The
default policy is `on_interface="downwind"`; pass `"upwind"` to recover
the v0.1.0 behaviour. Reproduce with `examples/05_interface_alignment.py`.

## Validation

The package reproduces the **uniform-mesh** tables of Türk (2016) to
within 0.2%:

| thesis table | scheme | cells | published $L_1$ | `weno_singular` |
|---|---|---|---|---|
| 6.3 | WENO3 | 20 / 80 / 320 | 3.74e-2 / 9.12e-3 / 2.22e-3 | 3.744e-2 / 9.121e-3 / 2.217e-3 |
| 6.4 | WENO5 | 20 / 80 / 320 | 3.54e-2 / 8.56e-3 / 2.10e-3 | 3.538e-2 / 8.560e-3 / 2.104e-3 |

Observed convergence orders match the published ones to four digits
(WENO3: 1.0186 / 1.0203 vs 1.0185 / 1.0186; WENO5: 1.0237 / 1.0124 vs
1.0237 / 1.0124).

**Scope.** Tables 6.1 and 6.2 of the thesis use a *non-uniform* mesh,
which this package does not yet implement; they are therefore **not**
reproduced here. Non-uniform mesh support is planned.

Run the test suite with `pytest` (56 tests). Examples in [`examples/`](examples/):

- `01_smooth_advection.py` — fifth-order convergence on a smooth solution
- `02_singular_source_explicit.py` — thesis Table 6.4, explicit RK3
- `03_singular_source_implicit.py` — thesis Table 6.4, RK3 + Crank–Nicolson
- `04_scheme_comparison.py` — WENO3 / WENO5 / WENO5-Z side by side (Tables 6.3 and 6.4)
- `05_interface_alignment.py` — the source-placement study above

## Citation

```bibtex
@software{turk_weno_singular,
  author    = {Türk, İrfan},
  title     = {{weno\_singular}: WENO solvers for linear hyperbolic
               conservation laws with singular source terms},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19865329},
  url       = {https://doi.org/10.5281/zenodo.19865329},
}
```

The DOI above is the *concept* DOI: it always resolves to the latest
release. To cite a specific release, use its version DOI, listed in
[CHANGELOG.md](CHANGELOG.md).

The underlying numerical methods are described in:

- İ. Türk and M. Ashyraliyev, *On the numerical solution of hyperbolic
  equations with singular source terms*, AIP Conf. Proc. **1611** (2014),
  374–379. [doi:10.1063/1.4893863](https://doi.org/10.1063/1.4893863)
- İ. Türk and M. Ashyraliyev, *On the numerical solution of diffusion
  problem with singular source terms*, AIP Conf. Proc. **1470** (2012),
  176–178. [doi:10.1063/1.4747668](https://doi.org/10.1063/1.4747668)
- İ. Türk, *On the numerical solution of advection diffusion reaction
  equations with singular source terms*, Ph.D. thesis, İstanbul
  University, 2016.

## Related work

[PyWENO](https://github.com/memmett/PyWENO) and
[weno4](https://pypi.org/project/weno4/) provide general-purpose WENO
reconstructions on uniform and non-uniform grids, but neither addresses
conservation laws forced by point sources. `weno_singular` complements
them with solvers, error metrics, and source-placement policies specific
to that case.

## Contributing

Issues and pull requests are welcome — bug reports, new test problems,
and requests for additional schemes alike.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

The numerical methods implemented in `weno_singular` were developed in
the author's Ph.D. thesis at İstanbul University and in the two joint
papers cited above. The author is deeply grateful to his thesis
supervisor, **Prof. Maksat Ashyraliyev**, for his mentorship, guidance,
and continuous support throughout the development of these methods.
