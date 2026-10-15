#!/usr/bin/env python
"""Example 7: shielding by a conducting slab against the classical skin-effect theory.

Exact: the characteristic-matrix transfer matrix (emi_lib.tmm) for a slab of conductivity sigma.
Theory (Schelkunoff): SE = R + A + B exactly, with (e^{-iwt} convention of emi_lib, Im n > 0 absorbs)
    Zw = Z0 / n,   gamma = -i k0 n   (Re gamma = k0 Im n > 0),
    R = 20 log10 |(Zw + Z0)^2 / (4 Zw Z0)|,   A = 20 log10 |exp(gamma d)|,   B = 20 log10 |1 - rho^2 exp(-2 gamma d)|,
and the good-conductor limits A = 8.686 d/delta with delta = sqrt(2/(w mu0 sigma)), R = 20 log10 (Z0 / 4|Zw|),
plus the thin-film limit SE = -20 log10 |2/(2 + Z0 sigma d)| for d << delta.
MEEP 1D with a DC-conductivity medium is run for one thickness and must agree with the transfer matrix.

Material: doped GaAs, sigma = 8.01e4 S/m (1000 nm membrane, van der Pauw), eps_r = 12.9, 75 GHz,
delta = 6.5 um; thicknesses 0.5-30 um span d/delta from 0.08 to 4.6.

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-meep]   (~3 min, MEEP dominated)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "gaas-emi-membrane"))
import exlib  # noqa: E402
import emi_lib as el  # noqa: E402

RES = HERE / "results"
SIGMA, EPS_R, F0 = 8.01e4, 12.9, 75e9
D_UM = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
MEEP_D_UM = 10.0


def schelkunoff(freq, d, sigma, eps_r):
    w = 2 * np.pi * freq
    eps_c = eps_r + 1j * sigma / (el.eps0 * w)
    n = np.sqrt(eps_c); n = np.where(n.imag < 0, -n, n)
    Zw = el.Z0 / n                           # wave impedance of the medium (exact)
    # Classical formulas assume e^{+jwt}: the wave goes as e^{-gamma z} with Re(gamma) = alpha > 0.
    # Our n carries the e^{-iwt} sign (Im n > 0 absorbs), so gamma = -i k0 n gives Re(gamma) = k0 Im(n) > 0.
    gamma = -1j * w / el.c * n
    rho = (Zw - el.Z0) / (Zw + el.Z0)
    R = 20 * np.log10(np.abs((Zw + el.Z0) ** 2 / (4 * Zw * el.Z0)))
    A = 20 * np.log10(np.abs(np.exp(gamma * d)))
    B = 20 * np.log10(np.abs(1 - rho ** 2 * np.exp(-2 * gamma * d)))
    return R, A, B, Zw


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-meep", action="store_true"); args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "sigma_Sm": SIGMA, "eps_r": EPS_R, "f_Hz": F0}
    t0 = time.time()
    delta = el.skin_depth(F0, SIGMA)
    w = 2 * np.pi * F0
    print(f"  sigma = {SIGMA:.3e} S/m, f = {F0/1e9:.0f} GHz: skin depth {delta*1e6:.2f} um, sigma/(w eps0 eps_r) = {SIGMA/(w*el.eps0*EPS_R):.2e} (good conductor)")
    S["delta_m"] = delta

    # ---------------------------------------------------------------- 1. TMM vs Schelkunoff over thickness
    stage("1. Transfer matrix vs Schelkunoff R + A + B and the classical limits")
    rows = []
    for d_um in D_UM:
        d = d_um * 1e-6
        eps_c = EPS_R + 1j * SIGMA / (el.eps0 * w)
        se = float(el.tmm(F0, [(d, eps_c)])["SE_dB"][0])
        R, A, B, Zw = schelkunoff(F0, d, SIGMA, EPS_R)
        A_cl = 8.685889638 * d / delta
        R_cl = 20 * np.log10(el.Z0 / (4 * abs(Zw)))
        thin = el.thin_film_SE(SIGMA, d)
        rows.append({"d_um": d_um, "d_over_delta": d / delta, "SE_tmm": se, "R": float(R), "A": float(A), "B": float(B), "A_cl": A_cl, "R_cl": float(R_cl), "thin": thin,
                     "sum_RAB": float(R + A + B)})
        print(f"  d = {d_um:5.1f} um (d/delta {d/delta:4.2f}): SE_tmm {se:7.2f}  R+A+B {R+A+B:7.2f}  [R {R:6.2f} A {A:6.2f} B {B:6.2f}]  A_cl {A_cl:6.2f}  R_cl {R_cl:6.2f}  thin-film {thin:7.2f}")
    S["rows"] = rows
    checks.add("SE_tmm == R + A + B (Schelkunoff identity)", all(abs(r["SE_tmm"] - r["sum_RAB"]) < 1e-6 for r in rows),
               f"max |diff| = {max(abs(r['SE_tmm'] - r['sum_RAB']) for r in rows):.1e} dB", "< 1e-6 dB at all thicknesses")
    checks.add("absorption term equals 8.686 d/delta", all(abs(r["A"] - r["A_cl"]) < 0.1 for r in rows),
               f"max |A - A_cl| = {max(abs(r['A'] - r['A_cl']) for r in rows):.3f} dB", "< 0.1 dB (good-conductor skin depth)")
    # R - R_cl is not noise: it is exactly 40 log10|1 + Zw/Z0| (0.089 dB here). Gate the identity, report the size.
    _, _, _, Zw = schelkunoff(F0, 1e-6, SIGMA, EPS_R)
    r_ident = 40 * np.log10(abs(1 + Zw / el.Z0))
    checks.add("R - R_cl equals 40 log10|1 + Zw/Z0| exactly", all(abs((r["R"] - r["R_cl"]) - r_ident) < 1e-9 for r in rows),
               f"R - R_cl = {rows[0]['R'] - rows[0]['R_cl']:.4f} dB = {float(r_ident):.4f} dB", "identity to 1e-9 dB; classical R_cl is within 0.2 dB")
    checks.add("reflection term equals the classical 20 log(Z0/4|Zw|)", all(abs(r["R"] - r["R_cl"]) < 0.2 for r in rows),
               f"max |R - R_cl| = {max(abs(r['R'] - r['R_cl']) for r in rows):.3f} dB", "< 0.2 dB (good-conductor approximation)")
    thin_rows = [r for r in rows if r["d_over_delta"] < 0.2]
    checks.add("thin-film limit for d << delta", all(abs(r["SE_tmm"] - r["thin"]) < 0.1 for r in thin_rows),
               f"max |SE - thin| = {max(abs(r['SE_tmm'] - r['thin']) for r in thin_rows):.3f} dB for d/delta < 0.2", "< 0.1 dB")
    thick = [r for r in rows if r["d_over_delta"] > 3]
    checks.add("multiple-reflection term vanishes for d > 3 delta", all(abs(r["B"]) < 0.1 for r in thick),
               f"max |B| = {max(abs(r['B']) for r in thick):.3f} dB", "< 0.1 dB")
    r20, r30 = [r for r in rows if r["d_um"] == 20.0][0], [r for r in rows if r["d_um"] == 30.0][0]
    slope = (r30["SE_tmm"] - r20["SE_tmm"]) / (10e-6 / delta)
    checks.add("SE slope in the thick limit is 8.686 dB per skin depth", abs(slope - 8.685889638) < 0.05, f"{slope:.3f} dB/delta between 20 and 30 um", "8.686 +/- 0.05")
    checks.add("thin-film formula fails for d ~ delta (expected)", abs(rows[-1]["thin"] - rows[-1]["SE_tmm"]) > 5, f"at 30 um: thin {rows[-1]['thin']:.1f} vs exact {rows[-1]['SE_tmm']:.1f} dB", "> 5 dB apart (regime boundary is real)")

    # ---------------------------------------------------------------- 2. MEEP 1D
    if not args.skip_meep:
        stage(f"2. MEEP 1D, {MEEP_D_UM:g} um slab, 65-85 GHz")
        mdir = RES / "meep"; mdir.mkdir()
        res_mm = 4000
        cfg = {"eps_r": EPS_R, "sigma_Sm": SIGMA, "thickness_m": float(f"{MEEP_D_UM:g}e-6"), "f_min_Hz": 65e9, "f_max_Hz": 85e9, "points": 21,
               "resolution_per_mm": res_mm, "pml_mm": 2.5, "pad_mm": 0.5, "decay": 1e-9, "decay_dt": 10,
               "center_offset_px": 0.5}        # slab edges on H sites (d = 40 px even), not on Dx sites
        n_half = (cfg["pad_mm"] + MEEP_D_UM * 1e-3 / 2) * res_mm
        print(f"  slab {MEEP_D_UM:g} um = {MEEP_D_UM*1e-3*res_mm:.0f} px; edge offset from the cell centre {n_half + 0.5:.1f} px (half-integer: edges on H sites)")
        (mdir / "config.json").write_text(json.dumps(cfg))
        try:
            rc, wall, _ = exlib.run([exlib.MEEP_PYTHON, HERE.parent / "gaas-emi-membrane" / "meep_mmwave.py", "config.json", "spectrum.json"], mdir, mdir / "meep.log",
                                       label="meep 1d skin-effect slab", timeout=7200)
        except subprocess.TimeoutExpired:
            rc, wall = 124, 7200.0
        ok = rc == 0 and (mdir / "spectrum.json").exists()
        checks.add("MEEP run completed", ok, f"rc={rc}, {wall:.0f}s", "rc 0 + spectrum.json")
        if ok:
            sp = json.loads((mdir / "spectrum.json").read_text()); fm = np.array(sp["freq_Hz"]); se_m = np.array(sp["SE_dB"])
            eps_c = EPS_R + 1j * SIGMA / (el.eps0 * 2 * np.pi * fm)
            se_t = el.tmm(fm, [(MEEP_D_UM * 1e-6, eps_c)])["SE_dB"]
            dse = float(np.abs(se_m - se_t).max())
            print(f"  MEEP SE(75 GHz) = {np.interp(75e9, fm, se_m):.2f} dB vs TMM {np.interp(75e9, fm, se_t):.2f} dB; max |dSE| over 65-85 GHz = {dse:.3f} dB ({wall:.0f} s)")
            checks.add("MEEP agrees with the transfer matrix", dse < 0.1, f"max |dSE| = {dse:.3f} dB (SE ~ {np.interp(75e9, fm, se_t):.0f} dB, T ~ {10**(-np.interp(75e9, fm, se_t)/10):.1e})", "< 0.1 dB over 65-85 GHz (slab edges on H sites)")
            S["meep"] = {"freq_Hz": fm, "SE_meep": se_m, "SE_tmm": se_t, "max_dSE": dse, "wall_s": wall, "d_um": MEEP_D_UM}

    # ---------------------------------------------------------------- 3. plot + results
    stage("3. Plot and RESULTS.md")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        dd = np.linspace(0.2, 32, 300) * 1e-6
        eps_c = EPS_R + 1j * SIGMA / (el.eps0 * w)
        se_curve = np.array([float(el.tmm(F0, [(d, eps_c)])["SE_dB"][0]) for d in dd])
        R, A, B, Zw = schelkunoff(F0, dd, SIGMA, EPS_R)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(dd / delta, se_curve, "k", label="transfer matrix (exact)")
        ax.plot(dd / delta, R + A, "C0--", label="Schelkunoff R + A")
        ax.plot(dd / delta, R + 8.685889638 * dd / delta, "C1:", label="R + 8.686 d/delta")
        ax.plot(dd / delta, el.thin_film_SE(SIGMA, dd), "C2-.", label="thin-film  -20log|2/(2+Z0 sigma d)|")
        if "meep" in S:
            ax.plot(MEEP_D_UM * 1e-6 / delta, np.interp(75e9, S["meep"]["freq_Hz"], S["meep"]["SE_meep"]), "r*", ms=12, label="MEEP 1D")
        ax.set_xlabel("d / delta  (delta = %.2f um at 75 GHz)" % (delta * 1e6)); ax.set_ylabel("SE (dB)"); ax.legend(fontsize=8)
        ax.set_title("GaAs membrane, sigma = 8.0e4 S/m: skin effect"); fig.tight_layout(); fig.savefig(RES / "skin_effect.png", dpi=140)
        print(f"  wrote {RES / 'skin_effect.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")
    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    L = [f"# Example 7 results: skin effect ({S['provenance']['date']})", "", f"sigma = {SIGMA:.3e} S/m, eps_r = {EPS_R}, 75 GHz, skin depth {delta*1e6:.2f} um", "",
         "## Checks", "", checks.markdown(), "", "| d (um) | d/delta | SE exact (dB) | R | A | B | 8.686 d/delta | thin-film |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['d_um']:.1f} | {r['d_over_delta']:.2f} | {r['SE_tmm']:.2f} | {r['R']:.2f} | {r['A']:.2f} | {r['B']:.2f} | {r['A_cl']:.2f} | {r['thin']:.2f} |")
    if "meep" in S:
        L += ["", f"MEEP 1D ({MEEP_D_UM:g} um, 4000 px/mm): max |SE_meep - SE_tmm| = {S['meep']['max_dSE']:.3f} dB over 65-85 GHz ({S['meep']['wall_s']:.0f} s)."]
    L += ["", "## Files", "", "- results/summary.json, results/skin_effect.png" + (", results/meep/spectrum.json" if "meep" in S else " (MEEP stage not run)")]
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
