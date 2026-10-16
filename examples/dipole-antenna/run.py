#!/usr/bin/env python
"""Example 5: dipole antennas in MEEP 3D against closed-form antenna theory.

Stage 1, Hertzian (point) dipole: the textbook sin^2(theta) pattern, directivity 3/2, and energy
conservation between the power through a closed box and the far-field power from near2far.
Stage 2, centre-fed half-wave wire dipole (radius lambda/50, delta-gap feed): input impedance
Z_in(f) = S/|I|^2 from the complex power S out of a small box around the gap and the Ampere-loop arm
current I, the feed power Re(S) against the radiated power, the radiation pattern and directivity of the sinusoidal-current model (1.641), and the
resonance and input resistance against the induced-EMF method (Balanis eqs. 8-60, 8-61).

    antenna_theory.py -> induced-EMF Z_in(l, a), sinusoidal-current pattern and directivity
    meep_dipole.py    -> MEEP 3D runs (--mode hertzian | halfwave)
    run.py            -> checks, plots, RESULTS.md, summary.json

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-halfwave]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import exlib  # noqa: E402
from antenna_theory import ETA0, directivity, induced_emf_impedance, pattern, x_zero_crossing  # noqa: E402

RES = HERE / "results"
RADIUS = 0.02                 # wire radius in wavelengths at f0 (length / diameter = 12.5: a fat dipole)
LENGTH = 0.5
GAP = 0.02                    # feed gap in wavelengths at f0
# Half-wave runs, coarse then fine (px per wavelength). The first run of this example used 100 px/lambda
# alone with a V/I port: the 0.02-lambda gap was 2 cells, the feed power came out 12-16 % low and the
# resonance at 0.781 f0. With the S/|I|^2 port the feed power matches the radiated power to 0.5 %, but the
# resonance still moves with the grid (0.8405 f0 at 100, 0.8730 f0 at 200 px/lambda), so the gates are
# evaluated on the 200 px/lambda run and the coarse run is reported for the trend.
HALFWAVE_RESOLUTIONS = (100, 200)
# Gates fixed before the first run (2026-10-14).
# Stage 1 is exact physics: only discretisation and near2far quadrature errors remain.
GATE_D_HERTZ = 0.02           # |D/1.5 - 1|
GATE_PATTERN_HERTZ = 0.02     # max |U/U_max - sin^2 theta|
GATE_PHI = 0.01               # azimuthal symmetry: max |U(phi=45) - U(phi=0)| / U_max
GATE_POWER = 0.02             # |P_far / P_box - 1| (both from the same DFT box)
# Stage 2: the references are a model (sinusoidal current) whose error grows with wire radius, so the
# resonance and resistance gates carry the model's own uncertainty, stated in the criterion.
GATE_FEED_POWER = 0.05        # |Re(S_feed) / P_box - 1|: the feed measurement is consistent with radiation
GATE_LOOP = 0.02              # arm current independent of the Ampere-loop size (added 2026-10-15 with the S/|I|^2 port)
GATE_D_HALF = 0.03            # |D/1.641 - 1| at l = lambda/2
GATE_PATTERN_HALF = 0.05      # max |U/U_max - U_sin/U_sin,max| at l = lambda/2
GATE_FRES = 0.05              # resonance (X = 0) frequency vs induced EMF for the same l and a
GATE_RRES = 0.20              # input resistance at resonance vs induced EMF at its own resonance


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def meep(mode: str, extra: list, tag: str = "") -> dict | None:
    out = RES / (mode + tag)
    cmd = [exlib.MEEP_PYTHON, HERE / "meep_dipole.py", out, "--mode", mode, *extra]
    rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label=f"meep dipole {mode}", timeout=6 * 3600)
    ok = rc == 0 and (out / "dipole.json").exists()
    return json.loads((out / "dipole.json").read_text()) if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-halfwave", action="store_true")
    args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    summary: dict = {"provenance": exlib.provenance()}

    stage("theory self-checks")
    th = np.linspace(0, np.pi, 20001)
    z_thin = complex(induced_emf_impedance(0.5, 1e-4))
    checks.add("induced EMF reproduces the thin half-wave dipole", abs(z_thin - (73.13 + 42.55j)) < 0.2,
               f"{z_thin.real:.2f} + j{z_thin.imag:.2f} ohm", "73.13 + j42.55 +/- 0.2 ohm (Balanis, Antenna Theory, sec. 8.5)")
    d_half = directivity(pattern(th, 0.5), th)
    checks.add("half-wave directivity of the sinusoidal current", abs(d_half - 1.641) < 0.002, f"{d_half:.4f}",
               "1.641 = 4/Cin(2 pi) (Balanis eq. 4-91 quotes 1.643 from rounded Cin)")

    stage("Hertzian dipole")
    h = meep("hertzian", [])
    checks.add("MEEP Hertzian run completed", h is not None, "dipole.json" if h else "failed", "rc 0")
    if h:
        theta = np.array(h["theta"])
        u0, u45 = np.array(h["u_phi0"]), np.array(h["u_phi45"])
        checks.add("Hertzian directivity", abs(h["directivity"] / 1.5 - 1) < GATE_D_HERTZ, f"{h['directivity']:.4f}",
                   f"1.5 +/- {GATE_D_HERTZ:.0%} (analytic)")
        dev = float(np.max(abs(u0 / u0.max() - np.sin(theta) ** 2)))
        checks.add("Hertzian pattern is sin^2(theta)", dev < GATE_PATTERN_HERTZ, f"max dev {dev:.4f}", f"< {GATE_PATTERN_HERTZ}")
        dphi = float(np.max(abs(u45 - u0)) / u0.max())
        checks.add("pattern independent of azimuth", dphi < GATE_PHI, f"{dphi:.2e}", f"< {GATE_PHI} between phi = 0 and 45 deg")
        ratio = h["p_far_f0"] / h["p_box_f0"]
        checks.add("far-field power equals box flux (Hertzian)", abs(ratio - 1) < GATE_POWER, f"P_far/P_box = {ratio:.4f}",
                   f"1 +/- {GATE_POWER}")
        summary["hertzian"] = {k: h[k] for k in ("directivity", "p_far_f0", "p_box_f0", "resolution", "voxels", "wall_s")}

    w = None
    if not args.skip_halfwave:
        stage("half-wave wire dipole")
        runs = {}
        for r in HALFWAVE_RESOLUTIONS:
            runs[r] = meep("halfwave", ["--radius", RADIUS, "--length", LENGTH, "--resolution", r,
                                        "--gap", GAP], tag=f"_res{r}")
            checks.add(f"MEEP half-wave run at {r} px/lambda completed", runs[r] is not None,
                       "dipole.json" if runs[r] else "failed", "rc 0")
        w = runs[max(HALFWAVE_RESOLUTIONS)]
        coarse = runs[min(HALFWAVE_RESOLUTIONS)]
    if w:
        f = np.array(w["freq_dft"])
        z = np.array(w["z_re"]) + 1j * np.array(w["z_im"])
        z_ohm = z * ETA0
        feed = np.array(w["p_in"]) / np.array(w["p_box_dft"])
        band = (f > 0.75) & (f < 1.25)          # away from the source spectrum's weak tails
        worst = float(np.max(abs(feed[band] - 1)))
        checks.add("feed power Re(S) equals radiated power", worst < GATE_FEED_POWER, f"worst |ratio - 1| = {worst:.3f}",
                   f"< {GATE_FEED_POWER} for 0.75 < f/f0 < 1.25 (lossless PEC: input = radiated)")
        loop_dev = float(np.max(abs(np.array(w["current_loop_ratio"])[band] - 1)))
        checks.add("arm current independent of loop size", loop_dev < GATE_LOOP, f"max |I(4 cells)/I(2 cells) - 1| = {loop_dev:.4f}",
                   f"< {GATE_LOOP} (conduction current only)")
        x07 = float(np.interp(0.7, f, z_ohm.imag))
        checks.add("short dipole is capacitive", x07 < 0, f"X(0.7 f0) = {x07:.1f} ohm", "X < 0 below resonance (sign convention)")
        r_pow = np.array(w["r_from_power"]) * ETA0
        theta = np.array(w["theta"])
        u0 = np.array(w["u_phi0"])
        checks.add("half-wave directivity at l = lambda/2", abs(w["directivity"] / 1.641 - 1) < GATE_D_HALF,
                   f"{w['directivity']:.4f}", f"1.641 +/- {GATE_D_HALF:.0%} (sinusoidal-current model)")
        ref = pattern(theta, LENGTH)
        dev = float(np.max(abs(u0 / u0.max() - ref / ref.max())))
        checks.add("half-wave pattern", dev < GATE_PATTERN_HALF, f"max dev {dev:.4f}", f"< {GATE_PATTERN_HALF} vs Balanis 4-58a")
        # resonance: X = 0 crossing of the measured Z_in(f), linear interpolation between DFT frequencies
        f_res, r_res = x_zero_crossing(f, z_ohm) or (float("nan"), float("nan"))
        # induced EMF for the same wire: l = LENGTH lambda0 at f, so l/lambda = LENGTH f/f0, a/lambda = RADIUS f/f0
        ff = np.linspace(0.6, 1.4, 8001)
        z_emf = induced_emf_impedance(LENGTH * ff, RADIUS * ff)
        j = int(np.argmin(abs(z_emf.imag[ff < 1.0])))
        f_res_emf, r_res_emf = float(ff[j]), float(z_emf.real[j])
        checks.add("resonance frequency", abs(f_res / f_res_emf - 1) < GATE_FRES,
                   f"MEEP {f_res:.4f} f0 vs induced EMF {f_res_emf:.4f} f0",
                   f"+/- {GATE_FRES:.0%} (model is exact only for thin wires; l/2a = {LENGTH / (2 * RADIUS):.1f})")
        checks.add("input resistance at resonance", abs(r_res / r_res_emf - 1) < GATE_RRES,
                   f"MEEP {r_res:.1f} ohm vs induced EMF {r_res_emf:.1f} ohm",
                   f"+/- {GATE_RRES:.0%} (same model caveat)")
        i0 = int(np.argmin(abs(f - 1.0)))
        if coarse:
            res_c = x_zero_crossing(coarse["freq_dft"], np.array(coarse["z_re"]) + 1j * np.array(coarse["z_im"]))
            fres_c = res_c[0] if res_c else None
            feed_c = np.array(coarse["p_in"]) / np.array(coarse["p_box_dft"])
            summary["halfwave_coarse"] = {"resolution": coarse["resolution"], "f_res": fres_c,
                                          "feed_ratio_range": [float(feed_c[band].min()), float(feed_c[band].max())],
                                          "directivity": coarse["directivity"], "wall_s": coarse["wall_s"]}
        summary["halfwave"] = {"f_res": f_res, "r_res_ohm": r_res, "f_res_emf": f_res_emf, "r_res_emf_ohm": r_res_emf,
                               "z_at_f0_ohm": [float(z_ohm[i0].real), float(z_ohm[i0].imag)],
                               "directivity": w["directivity"], "resolution": w["resolution"], "voxels": w["voxels"],
                               "wall_s": w["wall_s"], "radius": RADIUS, "length": LENGTH}
        plot(h, w, f, z_ohm, r_pow, ff, z_emf)
    return finish(checks, summary)


def plot(h, w, f, z_ohm, r_pow, ff, z_emf) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    for d, lab, ref in ((h, "Hertzian", None), (w, "half-wave", pattern(np.array(w["theta"]), LENGTH))):
        if d is None:
            continue
        th, u = np.array(d["theta"]), np.array(d["u_phi0"])
        ax.plot(th, u / u.max(), "o", ms=2.5, label=f"MEEP {lab}")
        r = np.sin(th) ** 2 if ref is None else ref / ref.max()
        ax.plot(th, r, "-", lw=1, label=f"theory {lab}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamax(180)
    ax.set_title("normalised U(θ)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(f, z_ohm.real, "C0o", ms=3, label="MEEP R = Re(S)/|I|²")
    ax2.plot(f, z_ohm.imag, "C1o", ms=3, label="MEEP X = Im(S)/|I|²")
    ax2.plot(f, r_pow, "C0x", ms=3, label="MEEP R = P_box/|I|²")
    ax2.plot(ff, z_emf.real, "C0-", lw=1, label="induced EMF R")
    ax2.plot(ff, z_emf.imag, "C1-", lw=1, label="induced EMF X")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlim(f.min(), f.max())
    ax2.set_ylim(-250, 250)
    ax2.set_xlabel("f / f0   (wire length = λ0/2)")
    ax2.set_ylabel("Z_in (Ω)")
    ax2.set_title(f"half-wave dipole, a = λ0/{1 / RADIUS:.0f}", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "dipole.png", dpi=150)


def finish(checks: exlib.Checks, summary: dict) -> int:
    summary["checks"] = checks.rows
    summary["all_pass"] = checks.all_pass
    exlib.dump_json(RES / "summary.json", summary)
    lines = [f"# Dipole antenna results ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"All checks pass: **{checks.all_pass}**", "", checks.markdown(), ""]
    for key in ("hertzian", "halfwave"):
        if key in summary:
            s = summary[key]
            lines.append(f"- {key}: resolution {s['resolution']} px/λ0, {s['voxels']:,} voxels before symmetry, "
                         f"wall {s['wall_s']:.0f} s (observation on a shared machine, not a benchmark)")
    if "halfwave" in summary:
        s = summary["halfwave"]
        lines.append(f"- half-wave: resonance {s['f_res']:.4f} f0 at R = {s['r_res_ohm']:.1f} Ω; at f0 "
                     f"Z = {s['z_at_f0_ohm'][0]:.1f} + j{s['z_at_f0_ohm'][1]:.1f} Ω")
    if "halfwave_coarse" in summary:
        c = summary["halfwave_coarse"]
        lines.append(f"- convergence: at {c['resolution']:g} px/λ0 (gap = {round(GAP * c['resolution'])} cells) the resonance is "
                     f"{c['f_res']:.4f} f0 and Re(S)/P_box spans {c['feed_ratio_range'][0]:.3f}-{c['feed_ratio_range'][1]:.3f}; "
                     "the gates use the finer grid")
    lines += ["", "![dipole](dipole.png)", "", "Provenance: see summary.json."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL PASS' if checks.all_pass else 'FAILURES'} -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
