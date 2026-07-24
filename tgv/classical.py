"""
Classical Chorin projection (fractional-step) solver for 2D incompressible NS.

The projection step is carried out entirely in Fourier space, which gives
machine-precision divergence freedom after each step.  Advection and viscous
terms are computed with 2nd-order central differences in physical space.

Axis convention: flat index k = i + N*j  (i = x-column, j = y-row).
After reshape(N, N), axis 0 is j (y) and axis 1 is i (x).
"""

import numpy as np
import scipy.sparse as sp
from .grid import Grid
from .exact import tgv_exact, kinetic_energy


# ── Spectral Poisson / projection ─────────────────────────────────────────────

def _wavenumbers(N: int, L: float):
    """Physical wavenumber arrays Kx, Ky (shape N×N) for a domain of length L."""
    m = np.fft.fftfreq(N, d=1.0 / N)      # integer indices [0,1,...,N/2-1,-N/2,...,-1]
    My, Mx = np.meshgrid(m, m, indexing='ij')   # [j, i] layout matching reshape(N,N)
    Kx = Mx * (2 * np.pi / L)
    Ky = My * (2 * np.pi / L)
    return Kx, Ky


def spectral_project(u_star: np.ndarray, v_star: np.ndarray,
                     dt: float, grid: Grid):
    """Project (u*, v*) onto the divergence-free manifold via spectral Poisson.

    Returns (u_new, v_new, p) with div_spectral(u_new) = 0 to machine ε.

    Uses continuous Fourier eigenvalues -(kx²+ky²), fully consistent with the
    spectral div and grad, so the projection is exact.
    """
    N, L = grid.N, grid.L
    Kx, Ky = _wavenumbers(N, L)
    K2 = Kx**2 + Ky**2

    # FFT of intermediate velocity (arrays are [j, i] after reshape)
    us_hat = np.fft.fft2(u_star.reshape(N, N))
    vs_hat = np.fft.fft2(v_star.reshape(N, N))

    # Spectral divergence of u*
    div_hat = 1j * (Kx * us_hat + Ky * vs_hat)

    # Poisson solve: -K² p̂ = (1/dt) div_hat
    safe_K2 = np.where(K2 > 1e-14, K2, 1.0)
    p_hat   = np.where(K2 > 1e-14, -div_hat / (dt * safe_K2), 0.0)

    # Pressure gradient and projection
    un_hat = us_hat - dt * (1j * Kx * p_hat)
    vn_hat = vs_hat - dt * (1j * Ky * p_hat)

    u_new = np.fft.ifft2(un_hat).real.ravel()
    v_new = np.fft.ifft2(vn_hat).real.ravel()
    p     = np.fft.ifft2(p_hat).real.ravel()
    p    -= p.mean()
    return u_new, v_new, p


def solve_pressure_fft(rhs: np.ndarray, grid: Grid) -> np.ndarray:
    """Solve ∇²p = rhs using the 5-point discrete Laplacian eigenvalues.

    Used by the quantum solver comparison path (HHL targets this operator).
    """
    N, h = grid.N, grid.h
    m = np.fft.fftfreq(N, d=1.0 / N)
    My, Mx = np.meshgrid(m, m, indexing='ij')
    evals = -(4.0 / h**2) * (np.sin(np.pi * Mx / N)**2 + np.sin(np.pi * My / N)**2)

    rhs_hat = np.fft.fft2(rhs.reshape(N, N))
    safe    = np.where(np.abs(evals) > 1e-14, evals, 1.0)
    p_hat   = np.where(np.abs(evals) > 1e-14, rhs_hat / safe, 0.0)

    p = np.fft.ifft2(p_hat).real.ravel()
    p -= p.mean()
    return p


# ── Physical-space operators ──────────────────────────────────────────────────

def advection(u: np.ndarray, v: np.ndarray, grid: Grid) -> tuple:
    """−(u·∇)u via 2nd-order central differences.

    Axis convention: reshape(N,N) gives [j, i]; x-derivative → axis 1, y → axis 0.
    """
    N, h = grid.N, grid.h

    def rx(f, d):   # shift in x (axis 1 = i-axis)
        return np.roll(f.reshape(N, N), d, axis=1).ravel()

    def ry(f, d):   # shift in y (axis 0 = j-axis)
        return np.roll(f.reshape(N, N), d, axis=0).ravel()

    dudx = (rx(u, -1) - rx(u, +1)) / (2 * h)
    dudy = (ry(u, -1) - ry(u, +1)) / (2 * h)
    dvdx = (rx(v, -1) - rx(v, +1)) / (2 * h)
    dvdy = (ry(v, -1) - ry(v, +1)) / (2 * h)

    return -(u * dudx + v * dudy), -(u * dvdx + v * dvdy)


def spectral_divergence(u: np.ndarray, v: np.ndarray, grid: Grid) -> np.ndarray:
    """Return the spectral divergence field ∂u/∂x + ∂v/∂y."""
    N, L = grid.N, grid.L
    Kx, Ky = _wavenumbers(N, L)
    div_hat = (1j * Kx * np.fft.fft2(u.reshape(N, N)) +
               1j * Ky * np.fft.fft2(v.reshape(N, N)))
    return np.fft.ifft2(div_hat).real.ravel()


# ── Solver class ──────────────────────────────────────────────────────────────

class ClassicalSolver:
    """Chorin projection solver for 2D incompressible NS (spectral projection)."""

    def __init__(self, N: int, nu: float = 0.1, dt: float = None):
        self.grid = Grid(N)
        self.nu   = nu
        self.N    = N

        if dt is None:
            dt = 0.4 * self.grid.h          # CFL-safe for |u_max|≤1
        self.dt = dt

        self.lap  = self.grid.laplacian_operator()
        self.div  = self.grid.div_operator()   # kept for reference / quantum path
        self.grad = self.grid.grad_operator()

    def step(self, u: np.ndarray, v: np.ndarray):
        """One Chorin projection step. Returns (u_new, v_new, p)."""
        dt, nu = self.dt, self.nu

        # Step 1: intermediate velocity
        adv_u, adv_v = advection(u, v, self.grid)
        visc_u = self.lap @ u
        visc_v = self.lap @ v

        u_star = u + dt * (adv_u + nu * visc_u)
        v_star = v + dt * (adv_v + nu * visc_v)

        # Step 2+3: spectral projection → divergence-free u_new
        u_new, v_new, p = spectral_project(u_star, v_star, dt, self.grid)
        return u_new, v_new, p

    def run(self, n_steps: int, nu: float = None, verbose: bool = False):
        """Run from exact TGV initial condition. Returns list of diagnostic dicts."""
        if nu is not None:
            self.nu = nu

        grid = self.grid
        t    = 0.0
        u, v, _ = tgv_exact(grid, t=0.0, nu=self.nu)

        history = []
        for step in range(n_steps + 1):
            u_ex, v_ex, _ = tgv_exact(grid, t=t, nu=self.nu)
            from .metrics import velocity_l2
            l2      = velocity_l2(u, v, u_ex, v_ex)
            ke      = kinetic_energy(u, v)
            div_arr = spectral_divergence(u, v, grid)
            div_err = float(np.max(np.abs(div_arr)))
            history.append({
                "step": step, "t": t, "l2": l2,
                "ke": ke, "div": div_err,
            })
            if verbose and step % 10 == 0:
                print(f"  step {step:4d}  t={t:.4f}  L2={l2:.3e}  "
                      f"KE={ke:.4f}  div={div_err:.2e}")
            if step < n_steps:
                u, v, _ = self.step(u, v)
                t += self.dt

        return history
