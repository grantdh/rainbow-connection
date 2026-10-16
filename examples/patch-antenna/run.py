#!/usr/bin/env python
"""Example 6: probe-fed microstrip patch (Balanis Example 14.1) in MEEP 3D against the textbook models.

The patch is designed with the transmission-line model for 10 GHz on a 1.588 mm, eps_r = 2.2 substrate
(W = 1.186 cm, L = 0.906 cm). MEEP builds it on a 3 cm square ground plane, drives it with an ideal
probe a quarter of the length in from one radiating edge, and measures the cavity resonances (Harminv),
the power the probe delivers (complex Poynting flux out of a small box around it), and the far field.

    patch_theory.py -> transmission-line design and resonance (Balanis eqs. 14-1..14-7), cavity-model R_in
    meep_patch.py   -> MEEP 3D run
    run.py          -> checks, plots, RESULTS.md, summary.json

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--px-per-h 4]
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
import patch_theory as pt  # noqa: E402

RES = HERE / "results"
EPS_R, H_CM, F0 = 2.2, 0.1588, 10e9
W_CM, L_CM = 1.186, 0.906          # Balanis, Antenna Theory 4th ed., Example 14.1
INSET = 0.25                        # probe distance from the feed edge / L
# Gates fixed before the first run (2026-10-14).
GATE_FRES = 0.03          # TM010 resonance vs the transmission-line model (Balanis: a few percent for h << lambda)
GATE_BROADSIDE = 0.95     # U(theta = 0) / U_max: a patch in its TM010 mode radiates broadside
GATE_POWER = 0.05         # |Re(S_feed) / P_far - 1| at 10 GHz: lossless structure, input power = radiated power


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px-per-h", type=int, default=4)
    args = ap.parse_args()
    exlib.reset_results(RES)
    checks = exlib.Checks()
    summary: dict = {"provenance": exlib.provenance()}

    stage("transmission-line model")
    w, l = pt.design(EPS_R, H_CM / 100, F0)
    checks.add("design reproduces Balanis Example 14.1", abs(w * 100 - W_CM) < 0.002 and abs(l * 100 - L_CM) < 0.002,
               f"W = {w * 100:.4f} cm, L = {l * 100:.4f} cm", "1.186 / 0.906 cm +/- 0.002")
    f_tl = pt.resonant_frequency(EPS_R, H_CM / 100, W_CM / 100, L_CM / 100)
    r_cavity = pt.inset_resistance(f_tl, W_CM / 100, L_CM / 100, INSET * L_CM / 100)
    print(f"  TL-model resonance of the simulated patch: {f_tl / 1e9:.4f} GHz; cavity-model R_in at the probe {r_cavity:.1f} ohm")

    stage("MEEP 3D")
    out = RES / "meep"
    cmd = [exlib.MEEP_PYTHON, HERE / "meep_patch.py", out, "--px-per-h", args.px_per_h, "--eps", EPS_R, "--h", H_CM,
           "--w", W_CM, "--l", L_CM, "--inset-frac", INSET]
    rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label="meep patch", timeout=6 * 3600)
    ok = rc == 0 and (out / "patch.json").exists()
    checks.add("MEEP patch run completed", ok, f"rc={rc}, {wall:.0f} s", "patch.json written")
    if not ok:
        return finish(checks, summary)
    d = json.loads((out / "patch.json").read_text())
    modes = [m for m in d["harminv"] if 7.5 < m["f_ghz"] < 12.5]
    checks.add("Harminv finds a cavity mode in the band", bool(modes), f"{len(modes)} mode(s)", "at least one in 7.5-12.5 GHz")
    if modes:
        tm010 = max(modes, key=lambda m: m["amp"])
        checks.add("TM010 resonance vs transmission-line model", abs(tm010["f_ghz"] * 1e9 / f_tl - 1) < GATE_FRES,
                   f"MEEP {tm010['f_ghz']:.3f} GHz (Q = {tm010['Q']:.1f}) vs {f_tl / 1e9:.3f} GHz",
                   f"+/- {GATE_FRES:.0%} (Balanis eqs. 14-1, 14-2, 14-5)")
        summary["tm010"] = tm010
    checks.add("broadside radiation", d["broadside_over_max"] > GATE_BROADSIDE, f"U(0)/U_max = {d['broadside_over_max']:.3f}",
               f"> {GATE_BROADSIDE} at 10 GHz")
    ratio = d["p_in_f0"] / d["p_rad_f0"]
    checks.add("input power equals radiated power", abs(ratio - 1) < GATE_POWER, f"Re(S_feed)/P_far = {ratio:.3f}",
               f"1 +/- {GATE_POWER} (PEC + lossless substrate)")
    f = np.array(d["freq_ghz"])
    delivered = np.array(d["delivered_norm"])
    i_pk = int(np.argmax(delivered))
    summary.update({"f_tl_ghz": f_tl / 1e9, "f_delivered_peak_ghz": float(f[i_pk]), "directivity_f0": d["directivity_f0"],
                    "directivity_f0_dBi": 10 * np.log10(d["directivity_f0"]), "resolution": d["resolution"],
                    "voxels": d["voxels"], "wall_s": d["wall_s"], "modes": d["harminv"]})
    print(f"  delivered-power peak (input resistance seen by a current source) at {f[i_pk]:.3f} GHz; "
          f"directivity {summary['directivity_f0_dBi']:.2f} dBi")
    plot(d, f, delivered, f_tl)
    return finish(checks, summary)


def plot(d, f, delivered, f_tl) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(f, delivered, "C0-", label="power delivered by the probe (normalised)")
    amp_max = max((m["amp"] for m in d["harminv"]), default=0.0)
    for m in d["harminv"]:
        if m["amp"] >= 0.01 * amp_max:                 # Harminv also returns fits far below the cavity mode
            ax1.axvline(m["f_ghz"], color="C3", lw=1, label=f"Harminv {m['f_ghz']:.2f} GHz, Q = {m['Q']:.0f}")
    ax1.axvline(f_tl / 1e9, color="k", ls=":", lw=1, label="TL-model resonance")
    ax1.set_xlabel("frequency (GHz)")
    ax1.set_ylabel("∝ input resistance")
    ax1.legend(fontsize=8, frameon=False)
    th = np.degrees(np.array(d["theta"]))
    for key, lab in (("u_E", "E-plane (xz)"), ("u_H", "H-plane (yz)")):
        u = np.array(d[key])
        ax2.plot(th, 10 * np.log10(np.maximum(u / u.max(), 1e-4)), label=lab)
    ax2.set_ylim(-30, 1)
    ax2.set_xlabel("θ (deg from broadside)")
    ax2.set_ylabel("normalised U (dB)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title(f"10 GHz, D = {10 * np.log10(d['directivity_f0']):.1f} dBi", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "patch.png", dpi=150)


def finish(checks: exlib.Checks, summary: dict) -> int:
    summary["checks"] = checks.rows
    summary["all_pass"] = checks.all_pass
    exlib.dump_json(RES / "summary.json", summary)
    lines = [f"# Microstrip patch results ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"All checks pass: **{checks.all_pass}**", "", checks.markdown(), ""]
    if "voxels" in summary:
        lines += [f"- power delivered by the probe peaks at {summary['f_delivered_peak_ghz']:.3f} GHz "
                  f"(transmission-line model {summary['f_tl_ghz']:.3f} GHz); absolute Z_in needs a coax-pin port model",
                  f"- directivity at 10 GHz {summary['directivity_f0_dBi']:.2f} dBi",
                  f"- resolution {summary['resolution']:.2f} px/cm, {summary['voxels']:,} voxels before symmetry, wall "
                  f"{summary['wall_s']:.0f} s (observation on a shared machine, not a benchmark)"]
    lines += ["", "![patch](patch.png)", "", "Provenance: see summary.json."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL PASS' if checks.all_pass else 'FAILURES'} -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
