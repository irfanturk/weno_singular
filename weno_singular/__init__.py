"""
weno_singular: WENO solvers for hyperbolic conservation laws with singular
source terms.

Top-level modules
-----------------
- ``weno_singular.weno5``         : WENO5 reconstruction on uniform mesh
- ``weno_singular.weno3``         : WENO3 reconstruction on uniform mesh
- ``weno_singular.flux``          : numerical flux functions (Lax–Friedrichs)
- ``weno_singular.time_steppers`` : SSP-RK3 and Crank–Nicolson integrators
- ``weno_singular.advection``     : solver for u_t + u_x = source terms
- ``weno_singular.burgers``       : inviscid Burgers' equation solver
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
