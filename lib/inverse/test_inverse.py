"""Standalone PASS/FAIL checks for the inverse-design solver.

1. A straight 500 nm Si guide through the window transmits T = 1 (normalization + PML).
2. The condensed design-window solve equals a full-grid LU solve.
3. Adjoint gradient dL/drho matches central finite differences (filter + projection + adjoint chain).
4. A 60-iteration bend run improves T and reports the per-iteration wall time (reported, not gated).

Run: ~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/test_inverse.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devices  # noqa: E402
import optimize  # noqa: E402
from fdfd import Solution  # noqa: E402

fails = 0


def check(name: str, ok: bool | None, detail: str):
    global fails
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    fails += ok is False
    print(f"[{tag}] {name}: {detail}")


def straight():
    d, rp = devices.straight()
    o = optimize.Optimizer(d, beta=0, radius=0)
    _, Ts, _, xds = o.obj(o.eps(rp), grad=False)
    T = Ts[0][1]
    check("straight guide T", abs(T - 1) < 1e-3, f"T = {T:.6f} (|T-1| < 1e-3)")

    s = o.obj.sims[0]
    x_full = Solution(s.lap, o.eps_full(rp)).solve(s.src.source(d.grid))
    x_cond = o.obj.field(0, xds[0]).ravel()
    err = np.abs(x_cond - x_full).max() / np.abs(x_full).max()
    check("condensed == full solve", err < 1e-7, f"max rel err {err:.1e} (< 1e-7; GMRES rtol 1e-9)")


def gradient():
    d = devices.build("splitter", design=1.6, ratio=0.7)
    o = optimize.Optimizer(d, beta=6, radius=0.1, init="random", seed=3)
    o.rho = np.clip(o.rho + 0.15 * np.random.default_rng(4).standard_normal(o.rho.shape), 0, 1)
    rho = o.rho.copy()
    _, _, g, _, _ = o.loss_grad(rho)
    rng = np.random.default_rng(5)
    errs = []
    for _ in range(6):
        v = rng.standard_normal(rho.shape)
        h = 1e-4
        lp = o.loss_grad(rho + h * v)[0]
        lm = o.loss_grad(rho - h * v)[0]
        fd = (lp - lm) / (2 * h)
        ad = float(np.sum(g * v))
        errs.append(abs(fd - ad) / max(abs(fd), 1e-12))
    check("adjoint vs finite difference", max(errs) < 1e-4,
          f"max rel err {max(errs):.1e} over 6 random directions (< 1e-4)")


def run():
    for kind, n in [("bend", 60), ("splitter", 20), ("mode_converter", 20), ("demux", 20)]:
        o = optimize.Optimizer(devices.build(kind))
        t0 = time.perf_counter()
        first = o.step()
        recs = [o.step() for _ in range(n - 1)]
        wall = (time.perf_counter() - t0) / n * 1e3
        nfac = sum(s.cond.factorizations for s in o.obj.sims)
        Ts = ", ".join(f"{k} {first['T'][k]:.2f}->{v:.2f}" for k, v in recs[-1]["T"].items())
        check(f"{kind} timing", None, f"{wall:.0f} ms/iter over {n} iters, {nfac} LU refactors; {Ts}")
        if kind == "bend":
            T0, T1 = first["T"]["in0->out0 @1550nm"], recs[-1]["T"]["in0->out0 @1550nm"]
            check("bend improves", T1 > 0.9, f"T {T0:.3f} -> {T1:.3f} in {n} iters (> 0.9)")


if __name__ == "__main__":
    straight()
    gradient()
    run()
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
