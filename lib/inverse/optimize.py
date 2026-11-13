"""Adjoint topology optimization on the FDFD solver.

Pipeline per iteration:
    rho (design pixels, [0,1]) -> conic filter W (min feature) -> tanh projection (beta, eta) -> eps
    -> one LU per (wavelength, source) -> forward solve x, mode overlaps -> T
    -> loss = sum w (T - target)^2 -> one adjoint solve A^T lam = dL/dx per (wavelength, source)
    -> dL/deps_i = -2 Re(lam_i k0^2 x_i) -> chain rule back to rho -> Adam step, clip to [0,1].

T is the modal power fraction: T = beta_out |c_out|^2 / (beta_in |c_ref|^2), with c_ref measured in a straight
copy of the input guide (no device), so T = 1 for a lossless, reflectionless, mode-preserving device.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from devices import Device, Term
from fdfd import Condensed, Laplacian, PortMode, Solution, Port, port_mode


# ---------------------------------------------------------------- parametrization

def conic_filter(shape: tuple[int, int], radius_px: float) -> sp.csr_matrix:
    """Row-normalized conic (hat) filter on the design pixels; zero-flux at the window edge (weights renormalized)."""
    nx, ny = shape
    if radius_px <= 1.0:
        return sp.identity(nx * ny, format="csr")
    r = int(np.ceil(radius_px))
    offs = [(a, b, 1.0 - np.hypot(a, b) / radius_px) for a in range(-r, r + 1) for b in range(-r, r + 1)
            if np.hypot(a, b) < radius_px]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rows, cols, vals = [], [], []
    for a, b, wgt in offs:
        i2, j2 = ii + a, jj + b
        ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny)
        rows.append((ii * ny + jj)[ok])
        cols.append((i2 * ny + j2)[ok])
        vals.append(np.full(ok.sum(), wgt))
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    return sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W


def project(x: np.ndarray, beta: float, eta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """tanh threshold projection and its derivative."""
    if beta <= 0:
        return x, np.ones_like(x)
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    t = np.tanh(beta * (x - eta))
    return (np.tanh(beta * eta) + t) / den, beta * (1 - t**2) / den


# ---------------------------------------------------------------- objective

@dataclass
class _Sim:
    lap: Laplacian
    src: PortMode
    p_in: float                                   # beta_in |c_ref|^2
    cond: Condensed
    r: np.ndarray                                 # condensed source
    bo: np.ndarray                                # source outside the window (for full fields)
    mons: list[tuple[Term, PortMode, complex, np.ndarray]]   # term, mode, a0, w


class Objective:
    """All solves for one Device. Laplacians, mode profiles and input-power references are built once."""

    def __init__(self, dev: Device, threads: int = 4):
        self.dev = dev
        g = dev.grid
        laps = {lam: Laplacian(g, lam) for lam in sorted({t.wavelength for t in dev.terms})}
        self.sims: list[_Sim] = []
        for (lam, src, sm) in sorted({(t.wavelength, t.src, t.src_mode) for t in dev.terms}):
            lap = laps[lam]
            sport = dev.ports[src]
            spm = port_mode(sport, g, dev.eps_bg, lap.k0, sm)
            cond = Condensed(lap, dev.eps_bg, dev.design)
            r, bo = cond.source(spm.source(g))
            mons = []
            for t in dev.terms:
                if (t.wavelength, t.src, t.src_mode) == (lam, src, sm):
                    pm = port_mode(dev.ports[t.mon], g, dev.eps_bg, lap.k0, t.mon_mode)
                    mons.append((t, pm, *cond.monitor(pm.overlap_vec(g), bo)))
            self.sims.append(_Sim(lap, spm, self._reference(lap, spm), cond, r, bo, mons))
        self.pool = ThreadPoolExecutor(max(1, min(threads, len(self.sims))))

    def _reference(self, lap: Laplacian, spm: PortMode) -> float:
        """Input power: launch into a straight copy of the input guide and project a few cells downstream."""
        g, sport = self.dev.grid, spm.port
        line = self.dev.eps_bg[sport.index, :] if sport.axis == "x" else self.dev.eps_bg[:, sport.index]
        eps_ref = np.broadcast_to(line, (g.nx, g.ny)).copy() if sport.axis == "x" \
            else np.broadcast_to(line[:, None], (g.nx, g.ny)).copy()
        mon = Port("ref", sport.axis, sport.index + 10, sport.lo, sport.hi)
        mpm = port_mode(mon, g, eps_ref, lap.k0, spm.mode)
        x = Solution(lap, eps_ref).solve(spm.source(g))
        return spm.beta * abs(mpm.overlap_vec(g) @ x) ** 2

    def _one(self, sim: _Sim, eps_d: np.ndarray, grad: bool):
        S = sim.cond.system(eps_d)
        xd = sim.cond.solve(S, sim.r)
        loss, Ts, gx = 0.0, [], np.zeros(xd.size, dtype=complex)
        for t, pm, a0, w in sim.mons:
            a = a0 + w @ xd
            T = pm.beta * abs(a) ** 2 / sim.p_in
            loss += t.weight * (T - t.target) ** 2
            Ts.append((t, T))
            # holomorphic dL/dx_d = dL/dT * beta/p_in * conj(a) * w
            gx += 2 * t.weight * (T - t.target) * pm.beta / sim.p_in * np.conj(a) * w
        d_eps = None
        if grad:
            lam = sim.cond.solve(S, gx, trans=True)
            # dS/deps_i = k0^2 e_i e_i^T;  dL/deps = 2 Re(gx . dx/deps) = -2 Re(lam * k0^2 * x)
            d_eps = -2.0 * np.real(lam * sim.cond.k2 * xd)
        return loss, Ts, d_eps, xd

    def __call__(self, eps_d: np.ndarray, grad: bool = True):
        """eps_d: permittivity on the design window (design_shape). Returns loss, [(term, T)], dL/deps_d, [x_d]."""
        e = eps_d.ravel()
        res = list(self.pool.map(lambda s: self._one(s, e, grad), self.sims))
        loss = sum(r[0] for r in res)
        Ts = [tt for r in res for tt in r[1]]
        d_eps = sum(r[2] for r in res).reshape(eps_d.shape) if grad else None
        return loss, Ts, d_eps, [r[3] for r in res]

    def retarget(self, terms: list[Term]):
        """Swap in new targets/weights for the same transmissions (no new solves or operators)."""
        key = lambda t: (t.wavelength, t.src, t.src_mode, t.mon, t.mon_mode)  # noqa: E731
        new = {key(t): t for t in terms}
        for s in self.sims:
            s.mons = [(new.get(key(t), t), pm, a0, w) for t, pm, a0, w in s.mons]

    def field(self, k: int, xd: np.ndarray) -> np.ndarray:
        """Full-grid Ez for simulation k from its design-window solution."""
        s = self.sims[k]
        return s.cond.full(s.bo, xd).reshape(self.dev.grid.nx, self.dev.grid.ny)


# ---------------------------------------------------------------- optimizer

class Optimizer:
    """Adam on the design density with live-adjustable filter radius, projection beta and step size."""

    def __init__(self, dev: Device, radius: float = 0.12, beta: float = 4.0, lr: float = 0.03,
                 init: str = "uniform", seed: int = 0, threads: int = 4, obj: Objective | None = None):
        self.dev = dev
        self.obj = obj or Objective(dev, threads)
        self.beta, self.lr = beta, lr
        self.set_radius(radius)
        self.reset(init, seed)

    def reset(self, init: str = "uniform", seed: int = 0):
        shp = self.dev.design_shape
        rng = np.random.default_rng(seed)
        self.rho = {"uniform": np.full(shp, 0.5),
                    "random": 0.5 + 0.15 * rng.standard_normal(shp),
                    "empty": np.full(shp, 0.25)}[init].clip(0, 1)
        self._restart()

    def _restart(self):
        self.m = np.zeros_like(self.rho)
        self.v = np.zeros_like(self.rho)
        self.t = 0
        self.history: list[dict] = []

    def set_device(self, dev: Device, obj: Objective):
        """Switch device/objective, warm-starting from the current density (resampled if the window changed)."""
        old = self.rho
        self.dev, self.obj = dev, obj
        self.__dict__.pop("last", None)  # its fields belong to the previous objective
        if old.shape != dev.design_shape:
            from scipy.ndimage import zoom
            f = (dev.design_shape[0] / old.shape[0], dev.design_shape[1] / old.shape[1])
            self.rho = np.clip(zoom(old, f, order=1), 0, 1)[:dev.design_shape[0], :dev.design_shape[1]]
            self.set_radius(self.radius)
        self._restart()

    def set_radius(self, radius_um: float):
        self.radius = radius_um
        self.W = conic_filter(self.dev.design_shape, radius_um / self.dev.grid.dx)

    def density(self, rho: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.rho if rho is None else rho
        rt = (self.W @ r.ravel()).reshape(r.shape)
        rp, drp = project(rt, self.beta)
        return rt, rp, drp

    def eps(self, rp: np.ndarray) -> np.ndarray:
        return self.dev.eps(rp)

    def eps_full(self, rp: np.ndarray) -> np.ndarray:
        return self.dev.eps_full(rp)

    def loss_grad(self, rho: np.ndarray):
        _, rp, drp = self.density(rho)
        loss, Ts, d_eps, fields = self.obj(self.eps(rp))
        g_rp = d_eps * (self.dev.eps_max - self.dev.eps_min)
        g_rho = (self.W.T @ (g_rp * drp).ravel()).reshape(rho.shape)
        return loss, Ts, g_rho, fields, rp

    def step(self) -> dict:
        t0 = time.perf_counter()
        loss, Ts, g, fields, rp = self.loss_grad(self.rho)
        self.t += 1
        b1, b2 = 0.9, 0.999
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mh = self.m / (1 - b1**self.t)
        vh = self.v / (1 - b2**self.t)
        self.rho = np.clip(self.rho - self.lr * mh / (np.sqrt(vh) + 1e-12), 0.0, 1.0)
        rec = dict(it=self.t, loss=float(loss), T={t.label: float(T) for t, T in Ts},
                   gray=float(np.mean(4 * rp * (1 - rp))), ms=1e3 * (time.perf_counter() - t0))
        self.history.append(rec)
        del self.history[:-64]  # Continuation only looks back `window` steps
        self.last = dict(rp=rp, fields=fields, **rec)
        return rec

    def export(self) -> dict:
        """The design as meep_check.py and the live page's download read it."""
        _, rp, _ = self.density()
        d = self.dev
        return dict(kind=d.kind, params=d.params, dx_um=d.grid.dx,
                    design_box=[d.design[0].start, d.design[0].stop, d.design[1].start, d.design[1].stop],
                    eps_min=d.eps_min, eps_max=d.eps_max, radius=self.radius, beta=self.beta,
                    rho=self.rho.round(5).tolist(), density=rp.round(5).tolist(),
                    last=dict(loss=self.last["loss"], T=self.last["T"]) if hasattr(self, "last") else None,
                    binary=self.binary_check())

    def binary_check(self) -> dict:
        """T of the design thresholded at 0.5 (what would be fabricated), no gradient."""
        _, rp, _ = self.density()
        # The binary design is far from the current one, so GMRES would refactor every preconditioner at it and the
        # next step would refactor them all back; keep the step's preconditioners instead.
        saved = [s.cond.pre for s in self.obj.sims]
        loss, Ts, _, _ = self.obj(self.eps((rp > 0.5).astype(float)), grad=False)
        for s, pre in zip(self.obj.sims, saved):
            s.cond.pre = pre
        return dict(loss=float(loss), T={t.label: float(T) for t, T in Ts})


class Continuation:
    """Stall detection and beta continuation, shared by the live server and run_design.py.

    Every step: if the loss is already small (every T within ~1 % of its target) or fell by < 1 % over the last
    `window` steps, raise beta by `factor` (up to `beta_max`) when auto_beta is on; once beta is at the cap, report
    converged. Returns "running", "raised" or "converged".
    """

    def __init__(self, window: int = 12, factor: float = 1.6, beta_max: float = 64.0):
        self.window, self.factor, self.beta_max = window, factor, beta_max
        self.since = 0

    def reset(self):
        self.since = 0

    def update(self, opt: "Optimizer", rec: dict, auto_beta: bool = True) -> str:
        self.since += 1
        h = opt.history
        if len(h) < self.window or self.since < self.window:
            return "running"
        prev = h[-self.window]["loss"]
        # "small" is per transmission, so it does not depend on how many terms there are or how they are weighted
        small = max(abs(rec["T"][t.label] - t.target) for t in opt.dev.terms) < 1e-2
        stalled = small or prev - rec["loss"] <= 1e-2 * prev
        if not stalled:
            return "running"
        if auto_beta and opt.beta < self.beta_max:
            opt.beta = float(min(self.beta_max, opt.beta * self.factor))
            self.since = 0
            return "raised"
        return "converged"
