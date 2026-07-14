"""
weno_singular: WENO solvers for linear hyperbolic conservation laws with
singular source terms.

Top-level modules
-----------------
- ``weno_singular.weno5``         : WENO5 reconstruction (Jiang-Shu and WENO-Z)
- ``weno_singular.weno3``         : WENO3 reconstruction on uniform mesh
- ``weno_singular.time_steppers`` : SSP-RK3 and Crank-Nicolson integrators
- ``weno_singular.advection``     : solver for u_t + u_x = source terms
"""

__version__ = "0.2.0"

__all__ = [
    "__version__",
]
