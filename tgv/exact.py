"""
Exact analytical solution of the 2D Taylor–Green vortex (TGV).

    u(x, y, t) =  sin(x) cos(y) e^{-2νt}
    v(x, y, t) = -cos(x) sin(y) e^{-2νt}
    p(x, y, t) = -(1/4)[cos(2x) + cos(2y)] e^{-4νt}

Kinetic energy per unit volume:  E(t) = (1/2) ⟨u²+v²⟩ = (1/2) e^{-4νt}

The grid indexing convention matches grid.py:
    cell (i, j)  →  flat index k = i + N*j
    x = grid.x[i],  y = grid.y[j]
"""

import numpy as np
from .grid import Grid


def tgv_exact(grid: Grid, t: float, nu: float = 0.1):
    """Return (u, v, p) as flat arrays of length N²."""
    X, Y = grid.XX.ravel(), grid.YY.ravel()
    decay2 = np.exp(-2 * nu * t)
    decay4 = decay2 ** 2

    u =  np.sin(X) * np.cos(Y) * decay2
    v = -np.cos(X) * np.sin(Y) * decay2
    p = -0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * decay4

    return u, v, p


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    """Discrete kinetic energy density ½ ⟨u²+v²⟩."""
    return 0.5 * float(np.mean(u**2 + v**2))


def kinetic_energy_exact(t: float, nu: float = 0.1) -> float:
    """Exact volume-averaged kinetic energy for TGV: ¼ e^{-4νt}.

    From ½⟨u²+v²⟩ = ½ · ½ · e^{-4νt} = ¼ e^{-4νt}, since
    mean(sin²·cos²) = mean(cos²·sin²) = ¼ on [0,2π]².
    """
    return 0.25 * np.exp(-4 * nu * t)
