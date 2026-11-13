"""2D frequency-domain (FDFD) Helmholtz solver for the out-of-plane field Ez, with stretched-coordinate PML.

Units: lengths in um, k0 = 2*pi/lambda in 1/um, time dependence exp(-i w t).
Equation on a uniform grid of cell-centred Ez samples (index i*ny + j, C order):

    (Dxb Dxf + Dyb Dyf + k0^2 eps_r) Ez = b

with Dxf/Dxb forward/backward differences divided by the PML stretch s = 1 + i a(d), which makes outgoing
waves exp(i k x) decay as exp(-k int a dx) inside the PML. Dirichlet zero beyond the outer boundary.

The matrix is complex symmetric only away from the PML corners, so adjoint solves use the transpose explicitly.
Dependencies: numpy, scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

PML_DECAY = 18.0  # target ln(1/R) for a normally incident wave in the cladding


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float          # um, same spacing in x and y
    npml: int          # PML thickness in cells on every side

    @property
    def n(self) -> int:
        return self.nx * self.ny


def _stretch(n: int, npml: int, dx: float, k0: float, half: bool) -> np.ndarray:
    """PML stretch factors on n samples; half=True evaluates at the i+1/2 points used by forward differences."""
    pos = np.arange(n) + (1.0 if half else 0.5)  # in cells, measured from the low edge
    L = npml * dx
    d = np.clip(np.maximum(npml - pos, pos - (n - npml)) / max(npml, 1), 0.0, 1.0)  # depth into the PML, 0..1
    a_max = PML_DECAY * 4.0 / (k0 * L) if npml else 0.0  # int_0^L a_max (d/L)^3 dx = a_max L / 4
    return 1.0 + 1j * a_max * d**3


def _diff1d(n: int, dx: float, s_f: np.ndarray, s_b: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    e = np.ones(n)
    df = sp.diags([-e, e[:-1]], [0, 1], shape=(n, n)) / dx
    db = sp.diags([e, -e[:-1]], [0, -1], shape=(n, n)) / dx
    return sp.diags(1.0 / s_f) @ df, sp.diags(1.0 / s_b) @ db


class Laplacian:
    """PML-stretched Laplacian for one wavelength; reused across every permittivity at that wavelength."""

    def __init__(self, grid: Grid, wavelength: float):
        self.grid = grid
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        g = grid

        def ops(n):
            return _diff1d(n, g.dx, *(_stretch(n, g.npml, g.dx, self.k0, half) for half in (True, False)))

        (dxf, dxb), (dyf, dyb) = ops(g.nx), ops(g.ny)
        ix, iy = sp.identity(g.nx), sp.identity(g.ny)
        self.L = (sp.kron(dxb @ dxf, iy) + sp.kron(ix, dyb @ dyf)).tocsc()

    def system(self, eps: np.ndarray) -> sp.csc_matrix:
        return (self.L + sp.diags(self.k0**2 * eps.ravel())).tocsc()


class Solution:
    """One LU factorization of A(eps): forward solves and adjoint (transpose) solves share it."""

    def __init__(self, lap: Laplacian, eps: np.ndarray):
        self.lu = spl.splu(lap.system(eps), permc_spec="COLAMD")

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.ascontiguousarray(b, dtype=complex))

    def solve_T(self, g: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.ascontiguousarray(g, dtype=complex), trans="T")


class Condensed:
    """Exact static condensation onto a design window, for when only eps inside the window changes.

    Split unknowns into design (d) and outside (o). With A_oo fixed (eps_bg outside), x_d solves the Schur system
        S x_d = r,   S = A_dd - A_do A_oo^-1 A_od,   r = b_d - A_do A_oo^-1 b_o.
    A_od only touches the ~4N ring pixels, so the correction is a dense ring x ring block built once per wavelength.
    Each iteration then solves a 6.4k-unknown system instead of the full grid. SuperLU is slow to refactor even that,
    so S is solved by GMRES preconditioned with the LU of an earlier S, refactored when GMRES needs > `max_iters`.
    Quantities outside the window are linear in x_d: a monitor amplitude c.x = u.b_o - (A_od^T u).x_d, u = A_oo^-T c_o.
    """

    max_iters = 20

    def __init__(self, lap: Laplacian, eps_bg: np.ndarray, design: tuple[slice, slice]):
        g = lap.grid
        self.lap, self.k2 = lap, lap.k0**2
        mask = np.zeros((g.nx, g.ny), bool)
        mask[design] = True
        self.di = np.flatnonzero(mask.ravel())
        self.oi = np.flatnonzero(~mask.ravel())
        A = lap.system(eps_bg).tocsr()
        Aoo = A[self.oi][:, self.oi].tocsc()
        self.Aod = A[self.oi][:, self.di].tocsc()
        self.Ado = A[self.di][:, self.oi].tocsr()
        L_dd = lap.L.tocsr()[self.di][:, self.di]
        self.lu_oo = spl.splu(Aoo, permc_spec="COLAMD")
        ring = np.unique(self.Aod.nonzero()[1])
        rows = np.unique(self.Ado.nonzero()[0])
        X = self.lu_oo.solve(self.Aod[:, ring].toarray().astype(complex))
        C = self.Ado[rows] @ X
        corr = sp.coo_matrix((C.ravel(), (np.repeat(rows, ring.size), np.tile(ring, rows.size))),
                             shape=L_dd.shape)
        self.S0 = (L_dd - corr).tocsr()
        self.pre = None
        self.factorizations = 0

    def system(self, eps_d: np.ndarray) -> sp.csr_matrix:
        return (self.S0 + sp.diags(self.k2 * eps_d)).tocsr()

    def _factor(self, S: sp.csr_matrix):
        self.pre = spl.splu(S.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                            options=dict(SymmetricMode=True))
        self.factorizations += 1

    def solve(self, S: sp.csr_matrix, rhs: np.ndarray, trans: bool = False) -> np.ndarray:
        if self.pre is None:
            self._factor(S)
        op = S.T.tocsr() if trans else S
        t = "T" if trans else "N"
        M = spl.LinearOperator(S.shape, matvec=lambda v: self.pre.solve(v, trans=t), dtype=complex)
        x, info = spl.gmres(op, rhs, M=M, rtol=1e-9, restart=self.max_iters, maxiter=1)
        if info != 0:  # preconditioner too stale: refactor at the current eps, solve directly
            self._factor(S)
            x = self.pre.solve(np.ascontiguousarray(rhs), trans=t)
        return x

    def source(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(r, b_o) for a full-grid source vector b."""
        bo = b[self.oi]
        return b[self.di] - self.Ado @ self.lu_oo.solve(bo), bo

    def monitor(self, c: np.ndarray, bo: np.ndarray) -> tuple[complex, np.ndarray]:
        """(a0, w) with amplitude c.x = a0 + w.x_d; requires c to vanish on the design window."""
        assert not np.any(c[self.di]), "monitor overlaps the design window"
        u = self.lu_oo.solve(np.ascontiguousarray(c[self.oi]), trans="T")
        return complex(u @ bo), -(self.Aod.T @ u)

    def full(self, bo: np.ndarray, xd: np.ndarray) -> np.ndarray:
        x = np.empty(self.lap.grid.n, dtype=complex)
        x[self.di] = xd
        x[self.oi] = self.lu_oo.solve(bo - self.Aod @ xd)
        return x


# ---------------------------------------------------------------- ports and modes

@dataclass(frozen=True)
class Port:
    """A line through the grid where a mode is launched or measured.

    axis 'x': the line is a column at x index `index`, spanning y indices [lo, hi); light travels along x.
    axis 'y': the line is a row at y index `index`, spanning x indices [lo, hi); light travels along y.
    """
    name: str
    axis: str
    index: int
    lo: int
    hi: int

    def flat(self, grid: Grid) -> np.ndarray:
        span = np.arange(self.lo, self.hi)
        if self.axis == "x":
            return self.index * grid.ny + span
        return span * grid.ny + self.index

    def eps_line(self, eps: np.ndarray) -> np.ndarray:
        return eps[self.index, self.lo:self.hi] if self.axis == "x" else eps[self.lo:self.hi, self.index]


def modes(eps_line: np.ndarray, dx: float, k0: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Guided modes of a 1D slab cross-section: (d2/dy2 + k0^2 eps) m = beta^2 m, Dirichlet ends.

    Returns (beta[count], profiles[count, n]) sorted by descending beta, profiles normalized to sum m^2 dx = 1
    and signed so the largest lobe is positive.
    """
    n = eps_line.size
    e = np.ones(n)
    d2 = (np.diag(-2 * e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)) / dx**2
    w, v = np.linalg.eigh(d2 + np.diag(k0**2 * eps_line.real))
    order = np.argsort(w)[::-1][:count]
    beta = np.sqrt(np.maximum(w[order], 0.0))
    prof = v[:, order].T / np.sqrt(dx)
    for p in prof:
        if p[np.argmax(np.abs(p))] < 0:
            p *= -1
    return beta, prof


@dataclass
class PortMode:
    port: Port
    mode: int
    beta: float
    profile: np.ndarray
    idx: np.ndarray       # flat grid indices of the port line

    def source(self, grid: Grid) -> np.ndarray:
        b = np.zeros(grid.n, dtype=complex)
        b[self.idx] = self.profile / grid.dx  # line current density; the 1/dx keeps amplitude grid-independent
        return b

    def overlap_vec(self, grid: Grid) -> np.ndarray:
        """c with amplitude = c . Ez (unconjugated dot), so the holomorphic derivative is simply c."""
        c = np.zeros(grid.n, dtype=complex)
        c[self.idx] = self.profile * grid.dx
        return c


def port_mode(port: Port, grid: Grid, eps: np.ndarray, k0: float, mode: int) -> PortMode:
    beta, prof = modes(port.eps_line(eps), grid.dx, k0, mode + 1)
    return PortMode(port, mode, float(beta[mode]), prof[mode], port.flat(grid))
