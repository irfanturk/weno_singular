# Changelog

All notable changes to **weno_singular** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-04-28

First public release.  Archived on Zenodo with DOI
[10.5281/zenodo.19865330](https://doi.org/10.5281/zenodo.19865330).

### Added
- WENO5 (Jiang-Shu) reconstruction on uniform periodic mesh.
- WENO3 (Jiang-Shu) reconstruction on uniform periodic mesh.
- SSP-RK3 (Shu-Osher) time stepper.
- Crank-Nicolson corrector with frozen WENO weights for semi-implicit
  time integration.
- Solvers for the linear scalar advection equation
  `u_t + u_x = g(t) * delta(x - xi)` with arbitrary source amplitudes.
- 30 unit and regression tests, including reproductions of thesis
  Table 6.4 (Türk, 2016) to four significant digits.
- Three runnable examples: smooth advection convergence, singular
  source explicit, singular source semi-implicit.
- GitHub Actions CI on Linux/macOS/Windows × Python 3.9-3.12.
