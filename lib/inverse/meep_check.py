"""Independent check of an exported inverse design with MEEP (2D FDTD, Ez / ODD_Z), plus an FDFD sweep of the same structure.

    ~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/meep_check.py DESIGN.json [--res 50] [--out DIR]
    ~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/meep_check.py --selftest

DESIGN.json is what the live page's "Download design" writes (kind, params, design box, density). The structure checked
is the fabricable one: density thresholded at 0.5 on the 40 nm FDFD pixels. MEEP samples those pixels exactly
(eps_averaging off) at a finer grid (default 50 px/um = 20 nm, 2x the optimizer's), so the comparison tests the
solver and its discretization, not a re-interpretation of the geometry.

One broadband pulse gives T(lambda) for every port pair. Each port uses eigenmode decomposition, and T is normalized by the
forward input-mode power measured just downstream of the source in the same run, so no reference run is needed.
The FDFD sweep of the identical structure is plotted against it. The gate compares the two at the design wavelengths.

Writes DIR/meep_check.json and DIR/meep_check.png. The exit code is 1 if any design-wavelength |T_meep - T_fdfd| > --tol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devices  # noqa: E402
import optimize  # noqa: E402


def load_structure(design: dict) -> tuple[devices.Device, np.ndarray, list[devices.Term]]:
    dev = devices.build(design["kind"], dx=design.get("dx_um", 0.04), **design["params"])
    rp = np.asarray(design["density"], dtype=float)
    if rp.shape != dev.design_shape:
        raise SystemExit(f"density {rp.shape} does not match the rebuilt design window {dev.design_shape}")
    eps = dev.eps_full((rp > 0.5).astype(float))
    terms = [devices.Term(**t) for t in design["terms"]] if "terms" in design else dev.terms
    return dev, eps, terms


def path_of(t: devices.Term) -> tuple[str, int, str, int]:
    return t.src, t.src_mode, t.mon, t.mon_mode


def paths(terms: list[devices.Term]) -> list[tuple[str, int, str, int]]:
    return sorted({path_of(t) for t in terms})


def key(path: tuple[str, int, str, int]) -> str:
    s, sm, m, mm = path
    return f"{s}{sm}->{m}{mm}"


# ---------------------------------------------------------------- MEEP

def run_meep(dev: devices.Device, eps: np.ndarray, terms, res: float, lam_lo: float, lam_hi: float, nfreq: int):
    import meep as mp
    mp.verbosity(0)

    g = dev.grid
    Lx, Ly = g.nx * g.dx, g.ny * g.dx
    # eps is indexed [i(x), j(y)]; cell i covers [i dx, (i+1) dx)
    # Pixel edges lie on MEEP grid planes (40 nm is a whole number of cells), so a sample could sit exactly on an
    # edge and fall on either side. Looking up a quarter cell away puts every sample strictly inside one pixel, which
    # shifts the whole structure uniformly by dx_meep/4 (harmless: ports move with it).
    q = 0.25 / res

    def material(p):
        i = min(g.nx - 1, max(0, int(np.floor((p.x + q + Lx / 2) / g.dx))))
        j = min(g.ny - 1, max(0, int(np.floor((p.y + q + Ly / 2) / g.dx))))
        return mp.Medium(epsilon=float(eps[i, j]))

    def centre(port, offset=0):
        """(centre, size, outward sign) of a port line in MEEP coordinates."""
        mid = (port.lo + port.hi) / 2 * g.dx
        span = (port.hi - port.lo) * g.dx
        pos = (port.index + offset + 0.5) * g.dx
        if port.axis == "x":
            return mp.Vector3(pos - Lx / 2, mid - Ly / 2), mp.Vector3(0, span), mp.X
        return mp.Vector3(mid - Lx / 2, pos - Ly / 2), mp.Vector3(span, 0), mp.Y

    f_lo, f_hi = 1 / lam_hi, 1 / lam_lo
    fcen, band = (f_lo + f_hi) / 2, f_hi - f_lo
    src_names = sorted({p[0] for p in paths(terms)})
    assert len(src_names) == 1, "one source port per check"
    sport = dev.ports[src_names[0]]
    def side(port):
        """+1 if the port sits in the upper half of its axis (light leaves along +normal), else -1."""
        return 1 if port.index > (g.nx if port.axis == "x" else g.ny) / 2 else -1

    sc, ssz, _ = centre(sport)
    sign_in = -side(sport)  # the source launches inward
    src_mode = paths(terms)[0][1]
    sources = [mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=1.5 * band), center=sc, size=ssz,
                                  eig_band=src_mode + 1, eig_parity=mp.ODD_Z,
                                  direction=mp.NO_DIRECTION,
                                  eig_kpoint=mp.Vector3(sign_in, 0) if sport.axis == "x" else mp.Vector3(0, sign_in))]
    sim = mp.Simulation(cell_size=mp.Vector3(Lx, Ly), resolution=res, boundary_layers=[mp.PML(g.npml * g.dx)],
                        default_material=material, eps_averaging=False, sources=sources)
    ic, isz, _ = centre(sport, offset=10 * sign_in)
    mons = {"__in": sim.add_mode_monitor(fcen, band, nfreq, mp.ModeRegion(center=ic, size=isz))}
    for name in sorted({p[2] for p in paths(terms)}):
        c, sz, _ = centre(dev.ports[name])
        mons[name] = sim.add_mode_monitor(fcen, band, nfreq, mp.ModeRegion(center=c, size=sz))
    t0 = time.perf_counter()
    sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))
    wall = time.perf_counter() - t0
    freqs = np.array(mp.get_flux_freqs(mons["__in"]))

    def coeffs(mon, mode):
        r = sim.get_eigenmode_coefficients(mon, [mode + 1], eig_parity=mp.ODD_Z)
        return r.alpha[0, :, 0], r.alpha[0, :, 1]  # amplitude along +normal, along -normal

    fwd, bwd = coeffs(mons["__in"], src_mode)
    a_in, a_ref = (fwd, bwd) if sign_in > 0 else (bwd, fwd)
    p_in = np.abs(a_in) ** 2
    out = {"R_" + sport.name: (np.abs(a_ref) ** 2 / p_in).tolist()}
    for (s, sm, m, mm) in paths(terms):
        f, b = coeffs(mons[m], mm)
        out[f"{s}{sm}->{m}{mm}"] = (np.abs(f if side(dev.ports[m]) > 0 else b) ** 2 / p_in).tolist()
    return 1 / freqs, out, wall


# ---------------------------------------------------------------- FDFD sweep of the same structure

def run_fdfd(dev: devices.Device, eps: np.ndarray, terms, lams: np.ndarray):
    sweep = [devices.Term(float(l), *path, 0.0) for l in lams for path in paths(terms)]
    _, Ts, _, _ = optimize.Objective(replace(dev, terms=sweep), threads=4)(eps[dev.design], grad=False)
    T = {(t.wavelength, *path_of(t)): v for t, v in Ts}
    return {key(p): [T[(float(l), *p)] for l in lams] for p in paths(terms)}


# ---------------------------------------------------------------- main

def check(design: dict, out_dir: Path, res: float, tol: float, nfreq: int = 101, n_fdfd: int = 9) -> bool:
    dev, eps, terms = load_structure(design)
    design_lams = sorted({t.wavelength for t in terms})
    lam_lo = min(1.2, min(design_lams) - 0.1)
    lam_hi = max(1.7, max(design_lams) + 0.1)
    lam_m, Tm, wall = run_meep(dev, eps, terms, res, lam_lo, lam_hi, nfreq)
    lam_f = np.unique(np.concatenate([np.linspace(lam_lo, lam_hi, n_fdfd), design_lams]))
    t0 = time.perf_counter()
    Tf = run_fdfd(dev, eps, terms, lam_f)
    wall_f = time.perf_counter() - t0

    rows, ok = [], True
    order = np.argsort(lam_m)
    targets = {(t.wavelength, *path_of(t)): t.target for t in terms}
    col = {lam: i for i, lam in enumerate(lam_f)}
    for p in paths(terms):
        k = key(p)
        for lam in design_lams:
            tm = float(np.interp(lam, lam_m[order], np.asarray(Tm[k])[order]))
            tf = Tf[k][col[lam]]
            target = targets.get((lam, *p))
            good = bool(abs(tm - tf) <= tol)
            ok &= good
            row = dict(path=k, wavelength=lam, target=target, T_meep=tm, T_fdfd=tf, ok=good)
            if target is not None and target >= 0.5:
                # passband peak within +-50 nm: separates "detuned by grid dispersion" from "lost"
                win = np.abs(lam_m - lam) <= 0.05
                i = np.flatnonzero(win)[np.argmax(np.asarray(Tm[k])[win])]
                row.update(T_meep_peak=float(Tm[k][i]), peak_wavelength=float(lam_m[i]))
            rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = dict(kind=design["kind"], params=design["params"], meep_res_px_per_um=res, fdfd_dx_um=dev.grid.dx,
               tol=tol, meep_wall_s=wall, fdfd_wall_s=wall_f, rows=rows,
               meep=dict(wavelength=lam_m.tolist(), T=Tm), fdfd=dict(wavelength=lam_f.tolist(), T=Tf))
    (out_dir / "meep_check.json").write_text(json.dumps(rep, indent=1))
    plot(rep, out_dir / "meep_check.png")
    for r in rows:
        peak = (f"; MEEP peak {r['T_meep_peak']:.3f} at {r['peak_wavelength'] * 1e3:.0f} nm"
                if "T_meep_peak" in r else "")
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['path']} @{r['wavelength'] * 1e3:.0f} nm: "
              f"MEEP {r['T_meep']:.3f}  FDFD {r['T_fdfd']:.3f}  (target {r['target']}, |diff| <= {tol}){peak}")
    print(f"MEEP {wall:.1f} s at {res:g} px/um; FDFD sweep {wall_f:.1f} s; wrote {out_dir}")
    return ok


def plot(rep: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2f6fd6", "#d0582a", "#2a9d6f", "#8e5bd0", "#7a7a7a"]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    lm = np.asarray(rep["meep"]["wavelength"]) * 1e3
    o = np.argsort(lm)
    for n, key in enumerate(k for k in rep["meep"]["T"] if not k.startswith("R_")):
        c = cols[n % len(cols)]
        ax.plot(lm[o], np.asarray(rep["meep"]["T"][key])[o], color=c, lw=1.8, label=f"{key} MEEP")
        ax.plot(np.asarray(rep["fdfd"]["wavelength"]) * 1e3, rep["fdfd"]["T"][key], "o", color=c, mfc="none",
                ms=5, label=f"{key} FDFD")
    rk = next(k for k in rep["meep"]["T"] if k.startswith("R_"))
    ax.plot(lm[o], np.asarray(rep["meep"]["T"][rk])[o], color=cols[-1], lw=1, ls="--", label="reflection (MEEP)")
    for lam in sorted({r["wavelength"] for r in rep["rows"]}):
        ax.axvline(lam * 1e3, color="0.6", lw=0.8, ls=":")
    ax.set(xlabel="wavelength (nm)", ylabel="modal transmission", ylim=(0, 1.05),
           title=f"{rep['kind']}: binary design, MEEP {rep['meep_res_px_per_um']:g} px/µm vs FDFD dx "
                 f"{rep['fdfd_dx_um'] * 1e3:.0f} nm")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def selftest_design() -> dict:
    """Straight 500 nm guide through the window, measured TE0 -> TE0 (expect T ~ 1, R ~ 0)."""
    dev, rp = devices.straight()
    return dict(kind=dev.kind, params=dict(dev.params), dx_um=dev.grid.dx, density=rp.tolist(),
                terms=[asdict(t) for t in dev.terms])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", nargs="?")
    ap.add_argument("--res", type=float, default=50.0, help="MEEP pixels per um (default 50 = 20 nm)")
    ap.add_argument("--tol", type=float, default=0.05, help="max |T_meep - T_fdfd| at design wavelengths")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        design, out = selftest_design(), a.out or HERE / "runs" / "selftest"
    else:
        if not a.design:
            ap.error("DESIGN.json required (or --selftest)")
        design = json.loads(Path(a.design).read_text())
        out = a.out or Path(a.design).with_suffix("")
    sys.exit(0 if check(design, out, a.res, a.tol) else 1)


if __name__ == "__main__":
    main()
