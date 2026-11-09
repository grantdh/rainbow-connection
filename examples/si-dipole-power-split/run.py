#!/usr/bin/env python3
"""The whole chain: Si meander dipole on SOI, power split and heating. Device data stays in the private repository (NM_REPRO); see README.md.

  run.py                 every stage, skipping what is already computed; registered predictions that reproduce stay untouched
  run.py --stage NAME    one of raw, epw, materials, thermal, fullwave, reciprocity, fem, fingerprints, validate, deep
  deep: the map of what heating would need (flip_map, contact, deep, validate_deep; rules prereg/00d-00g)
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


DEEP = [("flip_map.py", "map"), ("flip_map.py", "threshold"), ("contact.py",), ("flip_map.py", "linearity"), ("deep.py", "fingerprints"), ("deep.py", "gaas"),
        ("deep.py", "evidence"), ("deep.py", "redrows"), ("deep.py", "protocol"), ("deep.py", "contact"), ("validate_deep.py",)]      # in dependency order (07c needs 07b and 08)

STAGES = {
    "raw": lambda: py("recompute_y.py"),
    "epw": lambda: py("epw_series.py", "run"),
    "materials": lambda: py("materials_si.py"),
    "thermal": lambda: (py("thermal_closed.py"), py("thermal_elmer.py")),
    "fullwave": lambda: (py("fullwave_queue.py"), py("register_stages.py", "fullwave")),
    "reciprocity": lambda: (py("reciprocity_openems.py", "queue", python=OPENEMS), py("register_stages.py", "fullwave")),
    "fem": lambda: (py("register_stages.py", "fem-run"), py("register_stages.py", "fem")),
    "fingerprints": lambda: py("heating.py"),
    "validate": lambda: py("validate.py"),
    "deep": lambda: [py(*a) for a in DEEP],
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
