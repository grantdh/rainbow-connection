#!/usr/bin/env python
"""Example 4: plane wave on a dielectric sphere, MEEP 3D FDTD against the exact Mie series.

The simplest 3D scattering problem with a closed-form answer. The analytic side (mie_theory.py) is
checked first against Bohren & Huffman's published sample run, then MEEP computes the scattering
efficiency Q_sca(x) of an n = 2 sphere for size parameters x = 2 pi r / lambda from 2 to 8 at two grid
resolutions. The FDTD answer must agree with Mie theory and must get better as the grid is refined.

    mie_theory.py  -> analytic Q_sca(x), checked against Bohren & Huffman (1983) Appendix A
    meep_mie.py    -> MEEP Q_sca(x) at resolution 10, 20 and 40 (pixels per sphere radius)
    run.py         -> checks, plot, RESULTS.md, summary.json

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--resolutions 10 20 40] [--quick]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import exlib  # noqa: E402
from mie_theory import mie_q, rayleigh_qsca  # noqa: E402

RES = HERE / "results"

# --------------------------------------------------------------------------- comparison targets
# Bohren & Huffman, "Absorption and Scattering of Light by Small Particles" (1983), Appendix A sample
# run of BHMIE: m = 1.55, radius 0.525 um, wavelength 0.6328 um -> Qext = Qsca = 3.10543.
BH_SAMPLE = {"m": 1.55, "x": 2 * np.pi * 0.525 / 0.6328, "qsca": 3.10543, "tol": 1e-5}
INDEX = 2.0                    # lossless sphere, the case in MEEP's own Mie tutorial
# Gates, fixed before the first MEEP run (2026-10-14). That run used 10 and 20 px per radius and expected
# a few percent at 20 (7.9 px per in-sphere wavelength at x = 8, as in MEEP's tutorial); it measured 7.3 %,
# so 40 px per radius was added (2026-10-15). The gates themselves are unchanged and apply to the finest grid.
GATE_RMS_FINE = 0.05           # rms |Q_meep - Q_mie| / mean(Q_mie) on the finest grid
GATE_CONVERGENCE = 1.5         # error(coarse) / error(fine) for a 2x refinement (first order -> 2, second -> 4)
GATE_PEAK_SHIFT = 0.02         # relative shift of the strongest resonance in x


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=float, nargs="+", default=[10, 20, 40])
    ap.add_argument("--quick", action="store_true", help="resolution 6 and 10 only: a plumbing check, gates still apply")
    args = ap.parse_args()
    resolutions = [6, 10] if args.quick else sorted(args.resolutions)
    exlib.reset_results(RES)
    checks = exlib.Checks()
    summary: dict = {"provenance": exlib.provenance(), "index": INDEX, "resolutions": resolutions}

    stage("Mie theory")
    _, q = mie_q(BH_SAMPLE["m"] + 0j, BH_SAMPLE["x"])
    checks.add("Mie series reproduces Bohren & Huffman's sample run", abs(q - BH_SAMPLE["qsca"]) < BH_SAMPLE["tol"],
               f"Qsca = {q:.5f}", f"3.10543 +/- {BH_SAMPLE['tol']:g} (B&H 1983, App. A: m=1.55, x=5.2128)")
    qm, qr = mie_q(1.5 + 0j, 0.05)[1], rayleigh_qsca(1.5, 0.05)
    checks.add("Mie series -> Rayleigh limit for a small sphere", abs(qm / qr - 1) < 1e-3,
               f"ratio {qm / qr:.6f}", "|ratio - 1| < 1e-3 at x = 0.05 (Q = 8/3 x^4 |(m^2-1)/(m^2+2)|^2)")
    # Large spheres: Q_ext -> 2 (extinction paradox) through the diffraction edge term 1.992 x^(-2/3)
    # (van de Hulst 1957; Nussenzveig & Wiscombe, Phys. Rev. Lett. 45, 1490 (1980)). An absorbing sphere
    # removes the interference ripple of the lossless case, leaving an O(1/x) remainder.
    x_big = 1000.0
    qext_large = mie_q(1.33 + 0.05j, x_big)[0]
    q_asym = 2 + 1.992 * x_big ** (-2 / 3)
    checks.add("Mie series -> large-sphere extinction asymptote", abs(qext_large - q_asym) < 1e-3,
               f"Qext = {qext_large:.5f} vs {q_asym:.5f}",
               "2 + 1.992 x^(-2/3) +/- 1e-3 at x = 1000, m = 1.33+0.05i (edge term, Nussenzveig & Wiscombe 1980)")

    stage("MEEP 3D runs")
    spectra = {}
    for res in resolutions:
        out = RES / f"res{res:g}"
        cmd = [exlib.MEEP_PYTHON, HERE / "meep_mie.py", out, "--resolution", res, "--index", INDEX]
        rc, wall, _ = exlib.run(cmd, out, out / "meep.log", label=f"meep mie res {res:g}", timeout=4 * 3600)
        ok = rc == 0 and (out / "spectrum.json").exists()
        checks.add(f"MEEP resolution {res:g} completed", ok, f"rc={rc}, {wall:.0f} s", "spectrum.json written")
        if ok:
            spectra[res] = json.loads((out / "spectrum.json").read_text())
    if len(spectra) < len(resolutions):
        return finish(checks, summary)

    stage("MEEP vs Mie")
    errors = {}
    for res, sp in spectra.items():
        x = np.array(sp["x"])
        q_meep = np.array(sp["qsca"])
        q_mie = np.array([mie_q(INDEX + 0j, xi)[1] for xi in x])
        rel_rms = float(np.sqrt(np.mean((q_meep - q_mie) ** 2)) / np.mean(q_mie))
        errors[res] = rel_rms
        sp["qsca_mie"] = q_mie.tolist()
        summary[f"res{res:g}"] = {"rel_rms": rel_rms, "max_abs_err": float(np.max(abs(q_meep - q_mie))),
                                  "wall_s": sp["wall_s"], "voxels": sp["voxels"],
                                  "pixels_per_wavelength_in_sphere": sp["pixels_per_wavelength_in_sphere"]}
        print(f"  resolution {res:g}: rel rms {rel_rms:.4f}, max |dQ| {np.max(abs(q_meep - q_mie)):.3f}")
    ordered = sorted(spectra)
    fine, coarse = ordered[-1], ordered[-2]
    checks.add(f"MEEP matches Mie on the finest grid (res {fine:g})", errors[fine] < GATE_RMS_FINE,
               f"rel rms {errors[fine]:.4f}", f"< {GATE_RMS_FINE} of mean Q_sca over x = 2-8, n = {INDEX}")
    ratio = errors[coarse] / errors[fine]
    checks.add("FDTD error shrinks with resolution", ratio > GATE_CONVERGENCE,
               f"error ratio {ratio:.2f} ({coarse:g} -> {fine:g})", f"> {GATE_CONVERGENCE} for the last 2x refinement")
    sp = spectra[fine]
    x = np.array(sp["x"])
    xfine = np.linspace(x.min(), x.max(), 4000)
    qfine = np.array([mie_q(INDEX + 0j, xi)[1] for xi in xfine])
    x_peak_mie = xfine[np.argmax(qfine)]
    x_peak_meep = x[np.argmax(sp["qsca"])]
    dx_grid = float(np.max(np.diff(x)))
    shift = abs(x_peak_meep - x_peak_mie) / x_peak_mie
    checks.add("strongest resonance at the right size parameter", shift < GATE_PEAK_SHIFT + dx_grid / x_peak_mie,
               f"x_peak MEEP {x_peak_meep:.3f} vs Mie {x_peak_mie:.3f}",
               f"relative shift < {GATE_PEAK_SHIFT} + one frequency bin ({dx_grid:.3f})")
    summary.update({"x_peak_mie": x_peak_mie, "x_peak_meep": x_peak_meep, "error_ratio": ratio})
    plot(spectra, xfine, qfine)
    return finish(checks, summary)


def plot(spectra: dict, xfine: np.ndarray, qfine: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xfine, qfine, "k-", lw=1.2, label="Mie series")
    for marker, (res, sp) in zip("os^", sorted(spectra.items())):
        ax.plot(sp["x"], sp["qsca"], marker, ms=4, mfc="none", label=f"MEEP 3D, {res:g} px / radius")
    ax.set_xlabel("size parameter  x = 2πr/λ")
    ax.set_ylabel("scattering efficiency  Q_sca")
    ax.set_title(f"Dielectric sphere, n = {INDEX}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "mie.png", dpi=150)


def finish(checks: exlib.Checks, summary: dict) -> int:
    summary["checks"] = checks.rows
    summary["all_pass"] = checks.all_pass
    exlib.dump_json(RES / "summary.json", summary)
    lines = [f"# Mie scattering results ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"All checks pass: **{checks.all_pass}**", "", checks.markdown(), ""]
    for key in sorted((k for k in summary if re.fullmatch(r"res[\d.]+", k)), key=lambda k: float(k[3:])):
        s = summary[key]
        lines.append(f"- {key}: rel rms {s['rel_rms']:.4f}, {s['voxels']:,} voxels before symmetry, "
                     f"{s['pixels_per_wavelength_in_sphere']:.1f} px per in-sphere wavelength at x = 8, "
                     f"wall {s['wall_s']['empty'] + s['wall_s']['sphere']:.0f} s (observation on a shared machine, not a benchmark)")
    lines += ["", "![Q_sca](mie.png)", "", "Provenance: see summary.json."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL PASS' if checks.all_pass else 'FAILURES'} -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
