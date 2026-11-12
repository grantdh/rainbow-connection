"""Run the 2-D TM conformance cases through the MEEPhistopheles and MEEP backends and gate them (PASS/FAIL).

A. The backend is the engine: with --engine-c --swift-source-order (the program's float32 constants and source
   order), the 2-D cases reproduce the gap-voltage series MEEPhistopheles recorded in examples/metal-fdtd-crosscheck
   (voltage = mean gap Ez x dx; max |diff| / max |v| < 1e-4). Its 'empty' and 'fixed' runs are both c2d-empty-pec-box
   (the open gap leaves no conductor in the driven plane); 'shipped' is c2d-pec-cross.
B. Read-back checks on every run (both backends): soft source exact, conductor and walls zero; for
   MEEPhistopheles also every plane other than the data plane exactly zero.
C. MEEPhistopheles (case c) vs MEEP 2-D on the same case: gap-Ez series max |diff| / max |v| < 1e-3.
D. 2-D ring-down (32,000 steps), both backends: every matrix-pencil line of amplitude >= 0.10 on the 2-D discrete
   Yee relation of the 63 x 63 box (TM_mn0 modes) to |err| < 0.01 %.

Run: python3 lib/fdtd/test_meephistopheles_backend.py [--skip-ringdown]
Runs go to lib/fdtd/runs/{meephistopheles,meep}/... (gitignored), launched through examples/common/exlib.run.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from testlib import (BACKENDS, CASES, RUNS, XCHECK, check, exlib, finish, gate_ringdown, load_yee,
                     readback_checks, rel_diff, ringdown_lines, run_backend)
from case import Case

yee = load_yee()
RECORDED = {"empty": "c2d-empty-pec-box", "fixed": "c2d-empty-pec-box", "shipped": "c2d-pec-cross"}
CMD = {"meephistopheles": ["python3", BACKENDS / "meephistopheles_backend.py"],
       "meep": [exlib.MEEP_PYTHON, BACKENDS / "meep_backend.py"]}


def run(checks, engine: str, case_name: str, variant: str, flags: list[str]):
    out = RUNS / engine / variant / case_name
    return run_backend(checks, CMD[engine] + [CASES / f"{case_name}.json", out] + flags, out,
                       f"{engine} {variant}/{case_name}")


def main(argv: list[str]) -> int:
    checks = exlib.Checks()

    # A. reproduce the program's recorded runs
    for xvariant, name in RECORDED.items():
        rec, pr = run(checks, "meephistopheles", name, "engine-c", ["--engine-c", "--swift-source-order"])
        if not rec:
            continue
        p = XCHECK / f"meephistopheles_{xvariant}" / "timedomain.txt"
        if not p.exists():
            check(checks, f"A {name} ({xvariant})", None, f"{p} absent (gitignored)")
            continue
        recorded = yee.load_series(p)[1]
        ours = pr["gap_ez"].mean(axis=1) * np.float32(0.025)
        d = rel_diff(ours, recorded)
        check(checks, f"A {name}: reproduces MEEPhistopheles' recorded '{xvariant}' run",
              d < 1e-4 and len(ours) == len(recorded),
              f"max |diff| / max |v| = {d:.1e} over {len(recorded)} steps (the program ran 64 x 64 x 128; "
              "the host runs the one driven plane)", "max |diff| / max |v| < 1e-4")

    # B + C. case semantics; MEEPhistopheles vs MEEP 2-D
    names = ["c2d-empty-pec-box", "c2d-pec-cross"] + ([] if "--skip-ringdown" in argv else ["c2d-empty-pec-box-ringdown"])
    series = {}
    for name in names:
        for engine in ("meephistopheles", "meep"):
            rec, pr = run(checks, engine, name, "case", [])
            if rec:
                readback_checks(checks, f"B {engine}/{name}", rec, source_rel_tol=1e-12 if engine == "meep" else 0.0)
                series[engine] = (rec, pr["gap_ez"].mean(axis=1))
        if len(series) == 2:
            d = rel_diff(series["meephistopheles"][1], series["meep"][1])
            check(checks, f"C {name}: MEEPhistopheles vs MEEP 2-D gap-Ez series", d < 1e-3,
                  f"max |diff| / max |v| = {d:.2e} over {len(series['meep'][1])} steps", "max |diff| / max |v| < 1e-3")

        # D. 2-D ring-down on the Yee relation
        if name.endswith("ringdown"):
            case = Case.load(CASES / f"{name}.json")
            for engine, (rec, v) in series.items():
                rows = ringdown_lines(yee, rec, v, case.box_m, two_d=True)
                gate_ringdown(checks, yee, f"D {engine} (2-D)", rows)
                (RUNS / engine / "case" / name / "lines.json").write_text(json.dumps(rows, indent=1) + "\n")
                check(checks, f"  {engine} wall time", None, f"{rec['wall_s']:.1f} s for {len(v)} steps (observation)")
        series = {}
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
