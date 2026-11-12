"""Run the 3-D conformance cases through the MEEP backend and gate the results (PASS/FAIL; exit 1 on any FAIL).

Gated, per run (from the backend's own record, which reads the fields back from MEEP):
  - backend exit 0;
  - soft source: E at the source sample after iteration 1 equals A s(dt) (rel <= 1e-12);
  - conductor: max |E| on every PEC sample <= 1e-12 x the probe peak;
  - walls: max |E_tangential| on a 7x7 lattice of each of the six wall planes <= 1e-12 x the probe peak.
Gated, C1 ring-down: every matrix-pencil line of amplitude >= 0.10 lies on the discrete Yee dispersion relation
  of the case box (63 x 63 x 127 cells, c = 299792458, the case dt) to |err| < 0.01 %
  (helpers imported from examples/yee-dispersion/run.py, not copied).
Reported, not gated: C1/C4 against what LoopFDTD and MEEPhistopheles recorded in examples/metal-fdtd-crosscheck
  (same box, conductor cells, source and probe samples; they differ in c, precision and the Swift boundary guard).

Run: python3 lib/fdtd/test_meep_backend.py [--skip-ringdown]      (C1 ring-down: 32,000 steps, about a minute)
Runs go to lib/fdtd/runs/meep/<case>/ (gitignored). Launches go through examples/common/exlib.run (locks, run log).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from testlib import (BACKENDS, CASES, RUNS, XCHECK, check, exlib, finish, gate_ringdown, load_yee,  # noqa: F401
                     readback_checks, ringdown_lines, run_backend)
from case import Case

yee = load_yee()
CROSSCHECK = {"c1-empty-pec-box": "empty", "c4-gap-dipole": "fixed", "c4-gap-dipole-closed-gap": "shipped"}


def peak_mhz(t, v):
    f, X = yee.fft_spectrum(np.asarray(t), np.asarray(v), 0.0, pad=16)
    band = (f > 150e6) & (f < 650e6)
    return float(f[band][np.argmax(X[band])] / 1e6)


def main(argv: list[str]) -> int:
    checks = exlib.Checks()
    names = list(CROSSCHECK) + ([] if "--skip-ringdown" in argv else ["c1-empty-pec-box-ringdown"])
    runs = {}
    for name in names:
        path, out = CASES / f"{name}.json", RUNS / "meep" / name
        rec, pr = run_backend(checks, [exlib.MEEP_PYTHON, BACKENDS / "meep_backend.py", path, out], out,
                              f"meep_backend {name}")
        if rec:
            readback_checks(checks, name, rec, source_rel_tol=1e-12)
            runs[name] = (Case.load(path), rec, pr)

    name = "c1-empty-pec-box-ringdown"
    if name in runs:
        case, rec, pr = runs[name]
        rows = ringdown_lines(yee, rec, pr["gap_ez"].mean(axis=1), case.box_m)
        gate_ringdown(checks, yee, "C1 (meep)", rows)
        for r in sorted(rows, key=lambda r: -r["amp"])[:6]:
            check(checks, "  line", None, f"{r['f_MHz']:.4f} MHz amp {r['amp']:.2f} -> {r['mode']} Yee "
                  f"{r['f_yee_MHz']:.4f} MHz (err {r['err_pct']:+.1e} %, continuum {r['cont_err_pct']:+.3f} %)")
        (RUNS / "meep" / name / "lines.json").write_text(json.dumps(rows, indent=1) + "\n")

    # Reported: the same problems as the Swift engines recorded them (coarse 2000-step FFT peaks).
    for name, variant in CROSSCHECK.items():
        if name not in runs:
            continue
        case, rec, pr = runs[name]
        v = pr["gap_ez"].mean(axis=1)
        parts = [f"MEEP peak {peak_mhz(np.arange(len(v)) * case.dt(), v):.1f} MHz"]
        series = {eng: yee.load_series(XCHECK / f"{eng}_{variant}" / "timedomain.txt")
                  for eng in ("loopfdtd", "meephistopheles") if (XCHECK / f"{eng}_{variant}" / "timedomain.txt").exists()}
        parts += [f"{eng} {peak_mhz(*s):.1f} MHz" for eng, s in series.items()]
        if "loopfdtd" in series:
            w = case.raw["sources"][0]["waveform"]
            s = lambda t: math.sin(2 * math.pi * w["f0_Hz"] * t) * math.exp(-((t - w["t0_s"]) / w["width_s"]) ** 2)
            loop_v1 = series["loopfdtd"][1][1]
            parts.append(f"first nonzero sample: MEEP {v[1]:.5e}, LoopFDTD {loop_v1:.5e} (ratio {v[1] / loop_v1:.6f}; "
                         f"s(dt) at c = 299792458 vs 3e8 predicts {s(case.dt()) / s(case.dt(3e8)):.6f})")
        check(checks, f"{name} vs the Swift engines' recorded run (2000-step FFT peak, coarse)", None, "; ".join(parts))
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
