"""
Collocated cell-centred finite-volume grid on [0, 2π]² with periodic BCs.

Index map: cell (i, j) → k = i + N*j   (i = x-index, j = y-index)
This is the little-endian ordering that matches the qubit register:
    qubit-register index k  ↔  cell (k % N,  k // N)

Operators (scipy.sparse, shape N²×N²):
    div   : R^{2N²} → R^{N²}   divergence of a 2-component velocity field
    grad  : R^{N²} → R^{2N²}  gradient of a scalar field
    laplacian : R^{N²} → R^{N²}  5-point periodic Laplacian  (singular: zero row-sum)
"""

import numpy as np
import scipy.sparse as sp


def cell_to_flat(i, j, N):
    return int(i % N) + N * int(j % N)


def flat_to_cell(k, N):
    return k % N, k // N


class Grid:
    def __init__(self, N: int, L: float = 2 * np.pi):
        self.N = N
        self.L = L
        self.h = L / N                      # cell size
        self.n = N * N                       # total cells

        # Cell-centre coordinates
        self.x = np.linspace(self.h / 2, L - self.h / 2, N)
        self.y = np.linspace(self.h / 2, L - self.h / 2, N)
        self.XX, self.YY = np.meshgrid(self.x, self.y, indexing='ij')

    # ── Differential operators (central differences, 2nd order) ──────────────

    def _shift(self, axis: int, delta: int) -> sp.csr_matrix:
        """Sparse shift operator: maps f(i,j) → f(i+delta, j) or f(i, j+delta)."""
        N, n = self.N, self.n
        rows, cols = [], []
        for k in range(n):
            i, j = flat_to_cell(k, N)
            if axis == 0:
                ki = cell_to_flat((i + delta) % N, j, N)
            else:
                ki = cell_to_flat(i, (j + delta) % N, N)
            rows.append(k); cols.append(ki)
        data = np.ones(n)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def div_operator(self) -> sp.csr_matrix:
        """Divergence (∂u/∂x + ∂v/∂y) as N² × 2N² sparse matrix.

        Input vector: [u_flat (N²), v_flat (N²)]
        """
        h = self.h
        Sp_x = self._shift(0, +1)
        Sm_x = self._shift(0, -1)
        Sp_y = self._shift(1, +1)
        Sm_y = self._shift(1, -1)

        dfdx = (Sp_x - Sm_x) / (2 * h)   # central diff in x
        dfdy = (Sp_y - Sm_y) / (2 * h)   # central diff in y

        return sp.hstack([dfdx, dfdy])     # N² × 2N²

    def grad_operator(self) -> sp.csr_matrix:
        """Gradient (∂p/∂x, ∂p/∂y) as 2N² × N² sparse matrix."""
        h = self.h
        Sp_x = self._shift(0, +1)
        Sm_x = self._shift(0, -1)
        Sp_y = self._shift(1, +1)
        Sm_y = self._shift(1, -1)

        dpdx = (Sp_x - Sm_x) / (2 * h)
        dpdy = (Sp_y - Sm_y) / (2 * h)

        return sp.vstack([dpdx, dpdy])     # 2N² × N²

    def laplacian_operator(self) -> sp.csr_matrix:
        """5-point periodic Laplacian (second-order, singular: zero row-sum)."""
        h = self.h
        Sp_x = self._shift(0, +1)
        Sm_x = self._shift(0, -1)
        Sp_y = self._shift(1, +1)
        Sm_y = self._shift(1, -1)
        I    = sp.eye(self.n, format='csr')

        return (Sp_x + Sm_x + Sp_y + Sm_y - 4 * I) / (h * h)

    def laplacian_eigenvalues(self) -> np.ndarray:
        """Analytical eigenvalues of the periodic Laplacian.

        λ(k, l) = (4/h²)[sin²(πk/N) + sin²(πl/N)],  k,l = 0..N-1
        Returned as flat array of length N², ordered by flat index k = kx + N*ky.
        """
        h, N = self.h, self.N
        kx = np.arange(N)
        ky = np.arange(N)
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        evals = (4 / h**2) * (np.sin(np.pi * Kx / N)**2 + np.sin(np.pi * Ky / N)**2)
        return evals.ravel()  # shape (N²,), index k = kx + N*ky

    def cell_coords(self, k: int):
        """Return (x, y) centre-coordinates for flat cell index k."""
        i, j = flat_to_cell(k, self.N)
        return self.x[i], self.y[j]
