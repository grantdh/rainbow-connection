"""Run the conformance cases through the LoopFDTD backend and gate the results (PASS/FAIL; exit 1 on any FAIL).

A. The backend is the engine: with --engine-c --swift-source-order (the program's float32 constants and source
   order, kernel as shipped), C1, C4 and C4 closed-gap must reproduce the gap-voltage series LoopFDTD itself
   recorded in examples/metal-fdtd-crosscheck (max |diff| / max |v| < 1e-4; skipped if results/ is absent).
B. Case semantics, kernel guard-fixed and as shipped (case c, PEC wins): every run passes the backend's own
   read-back checks (soft source exact in float32, conductor and every wall sample exactly zero).
C. Against MEEP, same case, guard-fixed kernel: the gap-Ez mean series agree to max |diff| / max |v| < 1e-3 over
   all steps (float32 vs float64 on the same discrete problem). Needs lib/fdtd/runs/meep/ (test_meep_backend.py).
   The as-shipped kernel is reported, not gated, against MEEP.
D. C1 ring-down (32,000 steps), both kernels: every matrix-pencil line of amplitude >= 0.10 on the case box's
   discrete Yee relation to |err| < 0.01 %. The as-shipped kernel is expected to fail (boundary-guard defect,
   examples/yee-dispersion); that gate is kept red, not removed.

Run: python3 lib/fdtd/test_loopfdtd_backend.py [--skip-ringdown]
Runs go to lib/fdtd/runs/loopfdtd/<variant>/<case>/ (gitignored), launched through examples/common/exlib.run.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from testlib import (BACKENDS, CASES, RUNS, XCHECK, check, exlib, finish, gate_ringdown, load_yee,
                     readback_checks, rel_diff, ringdown_lines, run_backend)
from case import Case

yee = load_yee()
VARIANT = {"empty": "c1-empty-pec-box", "fixed": "c4-gap-dipole", "shipped": "c4-gap-dipole-closed-gap"}


def run(checks, case_name: str, variant: str, flags: list[str]):
    out = RUNS / "loopfdtd" / variant / case_name
    return run_backend(checks, ["python3", BACKENDS / "loopfdtd_backend.py", CASES / f"{case_name}.json", out] + flags,
                       out, f"loopfdtd {variant}/{case_name}")


def main(argv: list[str]) -> int:
    checks = exlib.Checks()

    # A. reproduce the program's own recorded runs
    for xvariant, name in VARIANT.items():
        rec, pr = run(checks, name, "engine-c-shipped", ["--engine-c", "--swift-source-order"])
        p = XCHECK / f"loopfdtd_{xvariant}" / "timedomain.txt"
        if not rec:
            continue
        if not p.exists():
            check(checks, f"A {name}: reproduces LoopFDTD's recorded run", None, f"{p} absent (gitignored)")
            continue
        recorded = yee.load_series(p)[1]
        ours = pr["gap_ez"].mean(axis=1)
        d = rel_diff(ours, recorded)
        check(checks, f"A {name}: reproduces LoopFDTD's recorded run ({xvariant})",
              d < 1e-4 and len(ours) == len(recorded), f"max |diff| / max |v| = {d:.1e} over {len(recorded)} steps",
              "max |diff| / max |v| < 1e-4")

    # B + C. case semantics, both kernels; MEEP comparison
    names = list(VARIANT.values()) + ([] if "--skip-ringdown" in argv else ["c1-empty-pec-box-ringdown"])
    for variant, flags in (("guard-fixed", ["--guard-fix"]), ("shipped", [])):
        for name in names:
            rec, pr = run(checks, name, variant, flags)
            if not rec:
                continue
            readback_checks(checks, f"B {variant}/{name}", rec)
            v = pr["gap_ez"].mean(axis=1)
            mp_file = RUNS / "meep" / name / "probes.npz"
            if mp_file.exists():
                d = rel_diff(v, np.load(mp_file)["gap_ez"].mean(axis=1))
                gated = variant == "guard-fixed" and not name.endswith("ringdown")
                check(checks, f"C {variant}/{name}: gap-Ez series vs MEEP", (d < 1e-3) if gated else None,
                      f"max |diff| / max |v| = {d:.2e} over {len(v)} steps", "max |diff| / max |v| < 1e-3" if gated else "")
            else:
                check(checks, f"C {variant}/{name} vs MEEP", None, "no MEEP run; run test_meep_backend.py first")

            # D. ring-down on the Yee relation
            if name.endswith("ringdown"):
                case = Case.load(CASES / f"{name}.json")
                rows = ringdown_lines(yee, rec, v, case.box_m)
                gate_ringdown(checks, yee, f"D loopfdtd {variant}", rows,
                              "" if variant == "guard-fixed" else " (as shipped: expected red, boundary-guard defect)")
                (RUNS / "loopfdtd" / variant / name / "lines.json").write_text(json.dumps(rows, indent=1) + "\n")
                check(checks, f"  {variant} throughput", None, f"{rec['throughput_Mcells_per_s']:.0f} Mcells/s, "
                      f"{rec['wall_s']:.1f} s ({rec['throughput_note']}; observation)")
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
