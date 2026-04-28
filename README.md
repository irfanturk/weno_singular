# weno_singular

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19865330.svg)](https://doi.org/10.5281/zenodo.19865330)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**WENO solvers for linear hyperbolic conservation laws with singular source terms.**

A specialized Python implementation of WENO3 and WENO5 finite-volume
schemes for 1D linear scalar advection on uniform meshes, with first-class
support for problems involving Dirac-delta point sources.

The library was developed as a companion to the convergence and validation
tables in Türk (2016).

---

## Features

- **WENO3 and WENO5** finite-volume reconstructions on uniform periodic meshes
- **Two time integrators**:
  - SSP-RK3 (Shu–Osher) for fully explicit time stepping
  - RK3 predictor + Crank–Nicolson corrector for semi-implicit time stepping
- **Singular source terms** (Dirac-delta in space, smooth or stationary in time)
  treated through a consistent direct-injection discretization
- Validated against analytical solutions (5th-order convergence on smooth
  problems, 1st-order convergence at discontinuities — matching theory)

## Installation

```bash
pip install weno_singular
```

Or, from source:

```bash
git clone https://github.com/irfanturk/weno_singular.git
cd weno_singular
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
from weno_singular.advection import solve_advection_singular

# Solve  u_t + u_x = sin(pi t) * delta(x - 1/3)
# on [0, 1] with periodic BCs and u(x, 0) = 0
result = solve_advection_singular(
    M=181,            # number of cell interfaces
    N=5001,           # number of time levels
    T_final=0.5,
    xi=1.0/3.0,       # source location
)

print(f"L_inf error : {result['max_err_inf']:.4e}")
print(f"L1 error    : {result['L1_err']:.4e}")
```

See [`examples/`](examples/) for more, including:

- `01_smooth_advection.py` — convergence study reproducing 5th order
- `02_singular_source_explicit.py` — Example 6.1.3 from Türk (2016), RK3
- `03_singular_source_implicit.py` — same problem, RK3 + Crank–Nicolson

## Validation

`weno_singular` reproduces the convergence tables of Türk (2016) to four
significant digits. See `tests/` for the regression suite and
`docs/validation.md` for full tables.

## Citation

If you use `weno_singular` in your research, please cite:

```bibtex
@software{turk_weno_singular,
  author    = {Türk, İrfan},
  title     = {{weno\_singular}: WENO solvers for linear hyperbolic
               conservation laws with singular source terms},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19865330},
  url       = {https://doi.org/10.5281/zenodo.19865330},
}
```

## Related work

For general-purpose WENO reconstructions on uniform and non-uniform grids,
see [PyWENO](https://github.com/memmett/PyWENO) and
[weno4](https://pypi.org/project/weno4/). `weno_singular` complements
these libraries with a focus on the specific case of conservation laws
with point-source forcing.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

The numerical methods implemented in `weno_singular` are based on the
algorithms developed in the author's PhD thesis at İstanbul University.
The author is deeply grateful to his thesis supervisor,
**Prof. Maksat Ashyraliyev**, for his mentorship, guidance, and
continuous support throughout the development of these methods.
