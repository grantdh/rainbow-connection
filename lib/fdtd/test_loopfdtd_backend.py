"""Run the conformance cases through the LoopFDTD backend and gate the results (PASS/FAIL; exit 1 on any FAIL).

A. The backend is the engine: with --engine-c --swift-source-order (the program's float32 constants and source
   order, kernel as shipped), C1, C4 and C4 closed-gap must reproduce the gap-voltage series LoopFDTD itself
   recorded in examples/metal-fdtd-crosscheck (max |diff| / max |v| < 1e-4; skipped if results/ is absent).
B. Case semantics, kernel guard-fixed and as shipped (case c, PEC wins): every run passes the backend's own
   read-back checks (soft source exact in float32, conductor and every wall sample exactly zero).
C. Against MEEP, same case, guard-fixed kernel: the gap-Ez mean series agree to max |diff| / max |v| < 1e-3 over
   all steps (float32 vs float64 on the same discrete problem). The suite launches the MEEP C1/C4 runs it compares
   with (test_all.py shares them with the MEEP suite); the ring-down comparison is reported when test_all.py has run
   the MEEP ring-down too. The as-shipped kernel is reported, not gated, against MEEP.
D. C1 ring-down (32,000 steps), both kernels: every matrix-pencil line of amplitude >= 0.10 on the case box's
   discrete Yee relation to |err| < 0.01 %. The as-shipped kernel is expected to fail (boundary-guard defect,
   examples/yee-dispersion); that gate is kept red, not removed.

Run: python3 lib/fdtd/test_loopfdtd_backend.py [--skip-ringdown] [--jobs N]
Runs go to lib/fdtd/runs/loopfdtd/<variant>/<case>/ (gitignored), launched through examples/common/exlib.run.
"""
from __future__ import annotations

import json
import sys

from testlib import (CASES, XCHECK, backend_job, check, exlib, gate_ringdown, load_yee, readback_checks, rel_diff,
                     ringdown_lines, run_suite, timing_note)
from case import Case

yee = load_yee()
VARIANT = {"empty": "c1-empty-pec-box", "fixed": "c4-gap-dipole", "shipped": "c4-gap-dipole-closed-gap"}
RINGDOWN = "c1-empty-pec-box-ringdown"
FLAGS = {"engine-c-shipped": ("--engine-c", "--swift-source-order"), "guard-fixed": ("--guard-fix",), "shipped": ()}


def job(variant: str, name: str):
    return backend_job("loopfdtd", name, variant, FLAGS[variant])


def case_names(argv: list[str]) -> list[str]:
    return list(VARIANT.values()) + ([] if "--skip-ringdown" in argv else [RINGDOWN])


def jobs(argv: list[str]):
    return ([job("engine-c-shipped", n) for n in VARIANT.values()]
            + [job(v, n) for v in ("guard-fixed", "shipped") for n in case_names(argv)]
            + [backend_job("meep", n) for n in VARIANT.values()])          # the runs gate C compares with


def gate(checks: exlib.Checks, results: dict, argv: list[str]) -> None:
    # A. reproduce the program's own recorded runs
    for xvariant, name in VARIANT.items():
        rec, pr = results[job("engine-c-shipped", name).key]
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

    # B + C + D. case semantics, both kernels; MEEP comparison; ring-down
    for variant in ("guard-fixed", "shipped"):
        for name in case_names(argv):
            rec, pr = results[job(variant, name).key]
            if not rec:
                continue
            readback_checks(checks, f"B {variant}/{name}", rec)
            v = pr["gap_ez"].mean(axis=1)
            mp_pr = results.get(backend_job("meep", name).key, (None, None))[1]
            if mp_pr is not None:
                d = rel_diff(v, mp_pr["gap_ez"].mean(axis=1))
                gated = variant == "guard-fixed" and name != RINGDOWN
                check(checks, f"C {variant}/{name}: gap-Ez series vs MEEP", (d < 1e-3) if gated else None,
                      f"max |diff| / max |v| = {d:.2e} over {len(v)} steps", "max |diff| / max |v| < 1e-3" if gated else "")
            else:
                check(checks, f"C {variant}/{name} vs MEEP", None, "no MEEP run of this case in this session")
            if name == RINGDOWN:
                rows = ringdown_lines(yee, rec, v, Case.load(CASES / f"{name}.json"))
                gate_ringdown(checks, yee, f"D loopfdtd {variant}", rows,
                              "" if variant == "guard-fixed" else " (as shipped: expected red, boundary-guard defect)")
                (job(variant, name).out / "lines.json").write_text(json.dumps(rows, indent=1) + "\n")
                check(checks, f"  {variant} throughput", None, f"{rec['throughput_Mcells_per_s']:.0f} Mcells/s, "
                      f"{rec['wall_s']:.1f} s ({rec['throughput_note']}; {timing_note(rec)})")


if __name__ == "__main__":
    sys.exit(run_suite(jobs, gate, sys.argv[1:]))
