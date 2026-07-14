# Changelog

All notable changes to **weno_singular** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **DOIs.** The concept DOI
> [10.5281/zenodo.19865329](https://doi.org/10.5281/zenodo.19865329) always
> resolves to the latest release. Each release below also carries its own
> version DOI.

## [0.2.0] — 2026-07-14

Zenodo version DOI: [10.5281/zenodo.21364566](https://doi.org/10.5281/zenodo.21364566)

Bug-fix and feature release.  Two of the fixes below change numerical
output, so this is a minor rather than a patch release; users comparing
against v0.1.0 results should read "Changed" carefully.

### Added
- **WENO-Z reconstruction** (`variant="z"`, Borges et al. 2008): global
  smoothness indicator `tau5 = |beta_0 - beta_2|` rescaling the Jiang-Shu
  weights.  Verified fifth order on smooth data.
- **Scheme selection in the solvers** via `scheme="weno3" | "weno5" |
  "weno5z"`.  The WENO3 module existed in v0.1.0 but was unreachable
  from `advection.py`, which hard-wired WENO5.  As a result, **thesis
  Table 6.3 (WENO3) is now reproduced** (to within 0.13%), alongside
  Table 6.4.
- **`on_interface` policy** on `find_delta_cell` and both solvers,
  controlling which cell receives the Dirac source when `xi` lands
  exactly on a cell interface.  Default `"downwind"`; `"upwind"`
  reproduces v0.1.0.
- **`L1_err_cell`** (and the alias `L1_err`) in the solver result: an
  `L1` error over cell averages, well posed on any mesh.  The existing
  `L1_err_face` is retained for comparison with the published tables.
- `examples/04_scheme_comparison.py` — WENO3 / WENO5 / WENO5-Z on the
  singular-source problem (reproduces Tables 6.3 and 6.4).
- `examples/05_interface_alignment.py` — the source-placement study.
- 26 new tests (56 total).

### Fixed
- **`find_delta_cell` ignored the domain length.**  The index was
  computed as `ceil((xi - x_left) * (M - 1)) - 1`, which silently assumes
  a domain of length 1.  On `[0, 2]` with `M = 5` and `xi = 1.0` it
  returned cell 3 = `[1.5, 2.0]`, not even adjacent to `xi`.  The cell
  width is now used explicitly, and `x_right` is a parameter.
- **Source injected on the wrong side of an interface.**  When `xi` fell
  exactly on a cell interface, the ceiling convention selected the
  *upwind* cell — where the exact solution is identically zero.  The
  resulting `L_inf` error equals the full jump height and does not
  converge under mesh refinement.  See README for the numbers.
- **The default mesh triggered exactly that case.**  `M = 181` gives 180
  cells, and `xi = 1/3` then lies on an interface.  (The value 180 comes
  from the *non-uniform* mesh of the thesis, but this package uses a
  uniform mesh.)  The default is now `M = 182`.
- **The quick-start example in the README raised `KeyError`.**  It read
  `result['L1_err']`; the key was `L1_err_face`.
- **Examples 02 and 03 printed the wrong convergence order.**  They used
  `log2` while refining the mesh by a factor of 4, reporting ~2.05 for
  what is first-order convergence.  Now `log4`, giving 1.02 / 1.01 and
  matching the thesis (1.0237 / 1.0124).
- **"Moving discontinuity" was wrong.**  The jump in the exact solution
  sits at `x = xi` and is *stationary*; the wavefront at `x = xi + t`
  moves but is continuous there.  Corrected in docstrings and examples.
- `solve_advection_singular_CR` imported `ssp_rk3_step` inside its time
  loop.

### Changed
- Results from `on_interface="downwind"` (the new default) differ from
  v0.1.0 **only on meshes where `xi` lies on a cell interface**.  Every
  uniform-mesh experiment in the thesis (20, 80, 320 cells with
  `xi = 1/3`) has `xi` strictly inside a cell, so Table 6.4 is
  bit-for-bit unaffected; a regression test now guards this.
- Documentation no longer claims to reproduce "the convergence tables"
  of the thesis in general: Tables 6.1 and 6.2 use a non-uniform mesh,
  which is not implemented.  Only Tables 6.3 and 6.4 are reproduced.
- The claim of agreement "to four significant digits" is replaced by the
  measured figure (within 0.2%); the published values carry three
  significant digits.
- Removed the `burgers` keyword from `pyproject.toml`: no Burgers solver
  is provided.

## [0.1.0] — 2026-04-28

Zenodo version DOI: [10.5281/zenodo.19865330](https://doi.org/10.5281/zenodo.19865330)

First public release.

### Added
- WENO5 (Jiang-Shu) reconstruction on uniform periodic mesh.
- WENO3 (Jiang-Shu) reconstruction on uniform periodic mesh.
- SSP-RK3 (Shu-Osher) time stepper.
- Crank-Nicolson corrector with frozen WENO weights for semi-implicit
  time integration.
- Solvers for the linear scalar advection equation
  `u_t + u_x = g(t) * delta(x - xi)` with arbitrary source amplitudes.
- 30 unit and regression tests, including a reproduction of thesis
  Table 6.4 (Türk, 2016).
- Three runnable examples: smooth advection convergence, singular
  source explicit, singular source semi-implicit.
- GitHub Actions CI on Linux/macOS/Windows × Python 3.9-3.12.
