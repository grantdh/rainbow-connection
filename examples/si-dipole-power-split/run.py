#!/usr/bin/env python3
"""The whole chain: Si meander dipole on SOI, power split and heating. Device data stays in the private repository (NM_REPRO); see README.md.

  run.py                 every stage, skipping what is already computed; registered predictions that reproduce stay untouched
  run.py --stage NAME    one of raw, epw, materials, thermal, fullwave, reciprocity, fem, fingerprints, validate
Each prediction must be committed in the private repository before validate runs; validate refuses otherwise. Failures stay red and do not change the exit code.
"""
import argparse
import subprocess
import sys

import common as c

NMPHYS = c.HOME / ".venvs/nmphys/bin/python"
OPENEMS = c.HOME / ".venvs/openems/bin/python"


def py(*args, python=NMPHYS):
    print("==", *args, flush=True)
    subprocess.run([str(python), *map(str, args)], cwd=c.HERE, check=True)


def reciprocity():
    import reciprocity_openems as ro
    for which in (1, 2, 0):
        if not ro.done(which):
            c.run_solver([OPENEMS, c.HERE / "reciprocity_openems.py", "run", str(which)], c.HERE, c.OUT / f"recip{which}.log", lock="yield")
    py("reciprocity_openems.py", "analyse", python=OPENEMS)
    py("register_stages.py", "fullwave")


STAGES = {
    "raw": lambda: py("recompute_y.py"),
    "epw": lambda: py("epw_series.py", "run"),
    "materials": lambda: py("materials_si.py"),
    "thermal": lambda: (py("thermal_closed.py"), py("thermal_elmer.py")),
    "fullwave": lambda: (py("fullwave_queue.py"), py("register_stages.py", "fullwave")),
    "reciprocity": reciprocity,
    "fem": lambda: (py("register_stages.py", "fem-run"), py("register_stages.py", "fem")),
    "fingerprints": lambda: py("heating.py"),
    "validate": lambda: py("validate.py"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=list(STAGES))
    a = ap.parse_args()
    c.require_private_repo()
    for name, fn in STAGES.items():
        if a.stage in (None, name):
            print(f"--- stage {name}", flush=True)
            fn()


if __name__ == "__main__":
    sys.exit(main())
