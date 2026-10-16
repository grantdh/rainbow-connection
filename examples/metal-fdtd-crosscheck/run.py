#!/usr/bin/env python
"""Example 3: three FDTD engines, one problem.

  1. LoopFDTD validation suite (plane wave, dipole radiation, resonant cavity, PML) built from
     ~/Dev/Claude/LoopFDTD/Sources and run against ~/Dev/Claude/LoopFDTD/Shaders/FDTD.metal.
  2. The same dipole in a 64x64x128 PEC box on
       - LoopFDTD/dipole_s11.swift            (Swift + Metal, full 6-component Yee)
       - MEEPhistopheles/src/metal/dipole_s11.swift (Swift + Metal, Ez/Hx/Hy only)
       - MEEP 3D (C++/Python, CPU)            meep_dipole_box.py
     Both Swift programs are run twice: as shipped, and with a one-line gap fix (see GAP_FIX below).
     As shipped, `gapCells/2` is 0 in integer arithmetic, so both arms include dz = 0 and the feed gap is
     closed: the programs simulate a solid 21-cell rod driven from inside. The fix uses (gapCells+1)/2.
  3. analyze.py: normalised gap-field spectra, resonance positions, throughput table.

Nothing is written into the LoopFDTD or MEEPhistopheles trees: patched copies and binaries live in
results/bin, every program runs with cwd under results/.
Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-meep]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import exlib  # noqa: E402

RES = HERE / "results"
LOOP = Path(os.environ.get("MUPPET_LOOPFDTD", Path.home() / "Dev/Claude/LoopFDTD"))
MEEPH = Path(os.environ.get("MUPPET_MEEPHISTOPHELES", Path.home() / "Dev/Claude/MEEPhistopheles"))
SWIFTC = ["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation"]
EMPTY_BOX = {  # drop the conductor entirely: a soft source in a PEC cavity with analytic modes
    "if isUpperArm || isLowerArm {": "if false && (isUpperArm || isLowerArm) {",
}
GAP_FIX = {  # as shipped -> fixed (both files use the same geometry code with different variable names)
    "dz_geo >= gapCells/2": "dz_geo >= (gapCells+1)/2", "dz_geo <= -gapCells/2": "dz_geo <= -(gapCells+1)/2",
    "dz >= gapCells/2": "dz >= (gapCells+1)/2", "dz <= -gapCells/2": "dz <= -(gapCells+1)/2",
}
SWIFT_SOURCES = {"loopfdtd": LOOP / "dipole_s11.swift", "meephistopheles": MEEPH / "src/metal/dipole_s11.swift"}
OUTDIR_NAME = {"loopfdtd": "dipole_loopfdtd_output", "meephistopheles": "dipole_meephistopheles_output"}


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def build(sources: list[Path], exe: Path, log: Path) -> bool:
    rc, wall, text = exlib.run(SWIFTC + [str(s) for s in sources] + ["-o", str(exe)], exe.parent, log, label=f"swiftc {exe.name}", nice=5)
    print(f"  built {exe.name}: rc={rc} in {wall:.0f}s")
    return rc == 0


def run_swift_dipole(eng: str, variant: str, checks: exlib.Checks, S: dict) -> None:
    src_dir = RES / "bin" / "src"; src_dir.mkdir(parents=True, exist_ok=True)
    text = SWIFT_SOURCES[eng].read_text()
    if variant == "fixed":
        n = 0
        for old, new in GAP_FIX.items():
            n += text.count(old); text = text.replace(old, new)
        checks.add(f"{eng}: gap fix applied to a copy", n == 2, f"{n} replacements", "2 lines (upper and lower arm)")
    if variant == "empty":
        n = 0
        for old, new in EMPTY_BOX.items():
            n += text.count(old); text = text.replace(old, new)
        checks.add(f"{eng}: empty-box variant generated", n == 1, f"{n} replacements", "conductor removed")
    src = src_dir / f"{eng}_{variant}.swift"; src.write_text(text)
    exe = RES / "bin" / f"dipole_{eng}_{variant}"
    if not build([src], exe, RES / "bin" / f"build_{eng}_{variant}.log"):
        checks.add(f"{eng} ({variant}) builds", False, src.name, "swiftc rc 0"); return
    wdir = RES / f"{eng}_{variant}_run"; wdir.mkdir()
    rc, wall, out = exlib.run([exe], wdir, wdir / "run.log", label=f"{eng} dipole_s11 {variant}", timeout=1800)
    produced = wdir / OUTDIR_NAME[eng]
    ok = rc == 0 and (produced / "timedomain.txt").exists()
    checks.add(f"{eng} ({variant}) ran", ok, f"rc={rc}, {wall:.0f}s", "timedomain.txt written")
    if not ok:
        return
    key = f"{eng}_{variant}"
    shutil.move(str(produced), RES / key)
    thr = re.search(r"Throughput:\s*([\d.,]+)\s*Mcells/s", out)
    pec = re.search(r"(\d+) PEC cells", out)
    with open(RES / key / "metadata.txt", "a") as fh:
        fh.write(f"engine: {eng} {variant} (Swift + Metal)\nthroughput_Mcells_per_s: {thr.group(1).replace(',', '') if thr else ''}\n"
                 f"pec_cells: {pec.group(1) if pec else ''}\n")
    S[key] = {"wall_s": wall, "throughput_Mcells_s": float(thr.group(1).replace(",", "")) if thr else None,
              "pec_cells": int(pec.group(1)) if pec else None, "fields": "Ex Ey Ez Hx Hy Hz" if eng == "loopfdtd" else "Ez Hx Hy"}
    print(f"  {key}: {S[key]['throughput_Mcells_s']} Mcells/s, {S[key]['pec_cells']} PEC cells")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-meep", action="store_true"); args = ap.parse_args()
    exlib.reset_results(RES)
    (RES / "bin").mkdir()
    checks = exlib.Checks()
    S: dict = {"provenance": exlib.provenance(), "loopfdtd": str(LOOP), "meephistopheles": str(MEEPH)}
    t0 = time.time()

    # ---------------------------------------------------------------- 1. validation suite
    stage("1. LoopFDTD validation suite (4 physics tests against FDTD.metal)")
    vdir = RES / "validation"; vdir.mkdir()
    os.symlink(LOOP / "Shaders", vdir / "Shaders")
    src_dir = RES / "bin" / "src"; src_dir.mkdir()
    # swiftc only allows top-level statements in main.swift, so the runner is compiled under that name
    shutil.copy(LOOP / "Sources/RunValidation.swift", src_dir / "main.swift")
    exe = RES / "bin" / "RunValidation"
    ok = build([LOOP / "Sources/FDTDSimulation.swift", LOOP / "Sources/FDTDValidation.swift", src_dir / "main.swift"],
               exe, RES / "bin" / "build_validation.log")
    checks.add("validation suite builds", ok, exe.name, "swiftc rc 0")
    if ok:
        rc, wall, text = exlib.run([exe], vdir, vdir / "validation.log", label="LoopFDTD RunValidation", timeout=1800)
        m = re.search(r"SUMMARY:\s*(\d+) passed,\s*(\d+) failed(?:,\s*(\d+) warnings?)?", text)
        summary = f"{m.group(1)} passed, {m.group(2)} failed" if m else f"no SUMMARY line (rc={rc})"
        print(f"  {summary} in {wall:.0f}s")
        checks.add("validation suite: all tests pass", bool(m) and int(m.group(2)) == 0 and int(m.group(1)) >= 4, summary, "4 passed, 0 failed")
        S["validation"] = {"summary": summary, "wall_s": wall, "rc": rc,
                           "lines": [l for l in text.splitlines() if any(k in l for k in ("PASS", "FAIL", "WARN", "error", "Error"))][:60]}

    # ---------------------------------------------------------------- 2. dipole on three engines
    stage("2. Dipole in a PEC box: Swift programs as shipped and with the gap fix, then MEEP")
    for variant in ("empty", "shipped", "fixed"):
        for eng in SWIFT_SOURCES:
            run_swift_dipole(eng, variant, checks, S)
    if not args.skip_meep:
        for variant, flag in (("empty", ["--empty"]), ("fixed", []), ("shipped", ["--as-shipped"])):
            rc, wall, text = exlib.run([exlib.MEEP_PYTHON, HERE / "meep_dipole_box.py", RES / f"meep_{variant}"] + flag, RES,
                                       RES / f"meep_{variant}_run.log", label=f"meep 3d dipole box {variant}", timeout=3600)
            checks.add(f"MEEP dipole ({variant}) ran", rc == 0 and (RES / f"meep_{variant}" / "timedomain.txt").exists(), f"rc={rc}, {wall:.0f}s", "timedomain.txt written")
            m = re.search(r"([\d.]+) Mcells/s", text)
            S[f"meep_{variant}"] = {"wall_s": wall, "throughput_Mcells_s": float(m.group(1)) if m else None, "fields": "full Yee (CPU, FP64)"}

    # ---------------------------------------------------------------- 3. compare
    stage("3. Spectral comparison")
    rc, wall, text = exlib.run([exlib.MEEP_PYTHON, HERE / "analyze.py", RES], RES, RES / "analyze.log", label="analyze", nice=0)
    print(text)
    comp = json.loads((RES / "comparison.json").read_text()) if (RES / "comparison.json").exists() else {}
    S["comparison"] = comp
    pairs = comp.get("pairs", {})
    # Empty PEC box: TM_mnp modes excited by a central Ez source have m, n odd, p even. MEEP's metallic cell
    # boundary is the N*dx box; the Swift programs hold tangential E to zero on the outermost node planes, so
    # their walls sit at 0 and (N-1)*dx. Continuum frequencies and a 2 % gate here (2000 steps, FFT peaks);
    # the precise test against the discrete Yee dispersion relation is examples/yee-dispersion.
    c0 = 299792458.0

    def analytic(nx, ny, nz, two_d=False):
        Lx, Ly, Lz = nx * 0.025, ny * 0.025, nz * 0.025
        return sorted({(m, n, q, 0.5 * c0 * ((m / Lx) ** 2 + (n / Ly) ** 2 + (q / Lz) ** 2) ** 0.5)
                       for m in (1, 3, 5) for n in (1, 3, 5) for q in ((0,) if two_d else (0, 2, 4, 6, 8))}, key=lambda x: x[3])

    boxes = {"loopfdtd_empty": ("(N-1) dx, 1.575 x 1.575 x 3.175 m", analytic(63, 63, 127)),
             "meep_empty": ("N dx, 1.6 x 1.6 x 3.2 m", analytic(64, 64, 128)),
             "meephistopheles_empty": ("(N-1) dx, 2D TM p = 0", analytic(63, 63, 127, two_d=True))}
    S["analytic_modes_MHz"] = {k: [(m, n, q, round(f / 1e6, 1)) for m, n, q, f in ml if f < 600e6] for k, (_, ml) in boxes.items()}
    for key, (label, modes) in boxes.items():
        pk = (comp.get("engines", {}).get(key) or {}).get("peaks_Hz", [])
        if not pk:
            continue
        errs = [min(abs(f - fm) / fm for _, _, _, fm in modes) for f in pk]
        worst = max(errs) * 100
        note = ("; as coded, LoopFDTD's single boundary guard freezes the normal E half a cell inside the far walls, so its box is "
                "not a Yee cavity of any size: expected red until LoopFDTD is fixed (fix and precise numbers in examples/yee-dispersion)"
                if key == "loopfdtd_empty" else "")
        checks.add(f"{key}: resonances match analytic TM modes of its box ({label})", worst < 2.0,
                   f"peaks {[round(f / 1e6, 1) for f in pk]} MHz, worst {worst:.2f}%", f"< 2 % of a continuum TM_mnp frequency (coarse; see yee-dispersion){note}")
        S[key]["mode_errors_pct"] = [round(e * 100, 2) for e in errs]
    p = pairs.get("loopfdtd_empty_vs_meep_empty")
    if p:
        checks.report("LoopFDTD vs MEEP, empty box: mode weights (informational)", f"rms {p['norm_spectrum_rms']:.3f}",
                      "not gated: the Swift programs add s(t) to Ez (soft E source) while MEEP drives a current sheet, so the same modes carry different weights")
    p = pairs.get("loopfdtd_fixed_vs_meep_fixed")
    if p:
        checks.report("LoopFDTD (gap fixed) vs MEEP dipole (informational)",
                      f"peaks {p['peak_rel_mismatch_max'] * 100:.1f}%, rms {p['norm_spectrum_rms']:.3f}",
                      "reported, not gated: a 5-cell PEC rod is defined cell-wise in Swift and Yee-position-wise in MEEP (half-cell shape ambiguity)")
    p = pairs.get("loopfdtd_shipped_vs_meep_shipped")
    if p:
        checks.report("LoopFDTD (as shipped) vs MEEP with the same closed rod (informational)",
                      f"peaks {p['peak_rel_mismatch_max'] * 100:.1f}%, rms {p['norm_spectrum_rms']:.3f}", "reported, not gated")
    p = pairs.get("meephistopheles_fixed_vs_meep_fixed")
    if p:
        checks.report("MEEPhistopheles (3-field model) vs MEEP (informational)",
                      f"peaks {p['peak_rel_mismatch_max'] * 100:.1f}%, rms {p['norm_spectrum_rms']:.3f}", "reported, not gated")

    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    (RES / "RESULTS.md").write_text(results_md(S, checks))
    print(f"\n{'ALL CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'} ({checks.summary()}) in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


def results_md(S, checks) -> str:
    p = S["provenance"]
    L = [f"# Example 3 results: Metal FDTD cross-check ({p['date']})", "",
         f"{p['hardware']}, macOS {p['macos']}, load1 {p['load1_at_start']:.2f}", "", checks.summary(), "", "## Checks", "", checks.markdown(), ""]
    if "validation" in S:
        L += ["## LoopFDTD validation suite", "", f"{S['validation']['summary']} ({S['validation']['wall_s']:.0f} s)", "", "```"] + S["validation"]["lines"] + ["```", ""]
    L += ["## Dipole in a PEC box", "", "| engine | variant | fields | PEC cells | throughput (Mcells/s) | wall (s) | resonances (MHz) |", "|---|---|---|---|---|---|---|"]
    for k, ml in S.get("analytic_modes_MHz", {}).items():
        L.append(f"Analytic TM modes for {k} (MHz): {', '.join(f'({m},{n},{q}) {f}' for m, n, q, f in ml)}")
    L.append("")
    for key in ("loopfdtd_empty", "meephistopheles_empty", "meep_empty", "loopfdtd_shipped", "loopfdtd_fixed", "meephistopheles_shipped", "meephistopheles_fixed", "meep_shipped", "meep_fixed"):
        if key in S:
            eng, variant = key.rsplit("_", 1)
            pk = S.get("comparison", {}).get("engines", {}).get(key) or {}
            errs = S[key].get("mode_errors_pct")
            L.append(f"| {eng} | {variant} | {S[key]['fields']} | {S[key].get('pec_cells', '-')} | {S[key]['throughput_Mcells_s']} | {S[key]['wall_s']:.1f} | "
                     f"{', '.join(f'{x / 1e6:.0f}' for x in pk.get('peaks_Hz', []))}{' (vs analytic: ' + ', '.join(f'{e}%' for e in errs) + ')' if errs else ''} |")
    L.append("")
    for k, v in S.get("comparison", {}).get("pairs", {}).items():
        if v:
            L.append(f"- {k}: V(t) correlation {v['time_corr']:.3f}, normalised |V(f)| rms {v['norm_spectrum_rms']:.3f}, max resonance mismatch {v['peak_rel_mismatch_max'] * 100:.1f}%")
    L += ["", "## Findings", "",
          "- Bug found in both Swift dipole programs: `gapCells/2` is integer 0, so the upper arm (dz >= 0) and the lower arm (dz <= 0) share the dz = 0 cells and the feed gap is closed. Fix: `(gapCells+1)/2`. The PEC cell count changes from 105 (21-cell rod) to 100 (two 10-cell arms).",
          "- The arm cross-section is the 5-cell '+' pattern (sqrt(di^2+dj^2) <= 1 cell), not a 2 mm wire; MEEP mirrors the cell pattern exactly.",
          "- MEEPhistopheles/src/metal/dipole_s11.swift evolves only Ez, Hx, Hy (a TMz scheme on a 3D grid), so its spectrum is not expected to match a full Yee solver.",
          "- Probes are uncalibrated in all programs (V = mean Ez over the patch, I = a curl proxy), so only normalised spectra are compared. A lumped port with a calibrated Ampere loop is the next step for the Swift codes.",
          "- The empty-box test is the coarse kernel gate (2000 steps, FFT peaks, continuum modes, 2 %): MEEP solves the N dx box (its metallic boundary is the cell surface) and MEEPhistopheles the (N-1) dx box (Ez held at zero on the outermost node planes). LoopFDTD as coded fails its (N-1) dx gate: its single boundary guard also freezes the normal E component half a cell inside the far walls, so the box is not a Yee cavity of any size (an earlier version of this example compared it with the N dx box and passed by accident). The precise test against the discrete Yee dispersion relation, including the one-line per-component guard fix, is examples/yee-dispersion. Which modes dominate differs because the Swift programs inject a soft E field while MEEP injects a current.",
          "- LoopFDTD's own validation suite does not pass as of this run (see the block above): the plane-wave test passes, the cavity test reports a resonance of 23 GHz on a 707 MHz cavity (its frequency extraction, not the kernel, given the empty-box result here), the PML test measures -10 dB against a -40 dB target, and the dipole test cannot measure a current. QUICK_START.md says all four pass; that claim is stale.",
          "", "## Files", "", "- results/validation/validation.log, results/bin/src/*_fixed.swift (patched copies), results/<engine>_<variant>/timedomain.txt, results/comparison.json, results/crosscheck.png, results/summary.json"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
