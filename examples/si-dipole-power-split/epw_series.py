"""EPW transport for the power split: mobility vs temperature at the device's carrier density, from stage-1 matrix elements
(symmetric NSCF, lifc = .true.) in STAGE1. Runs the frozen, validated fast build for the 48^3 series and an unmodified-EPW
control on a 24^3 grid at the lowest and highest temperature, which must agree to the printed digits.

  epw_series.py run      the transport runs (nice'd, timing lock honoured)
  epw_series.py stage1   the matrix elements again with the unmodified build, compared bit by bit with the production file
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as c

BASE = c.HOME / "qe-test/Si_EPW"
STAGE1 = Path(os.environ.get("EPW_STAGE1", BASE / "work"))
MOB = BASE / "mob"
UPSTREAM = c.HOME / "qe-test/q-e-qe-7.4.1/EPW_upstream/src/epw.x"
TR = c.HOME / "qe-test/EPW_metal/frozen/epw_tr-0cdcb82e/epw.x"
TEMPS = "300 350 400 450 500"
ENV = {"AB_MODE": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
TR_ENV = {"EPW_TR": "1", "EPW_TR_SCREEN": "1", "EPW_TR_QWIN": "1", "EPW_TR_PAIRS": "1", "EPW_TR_BLAS": "1", "EPW_TR_BATCH": "64",
          "EPW_TR_MEM_MB": "1000"}
RUNS = {  # name: (carrier, fine grid, temps, binary)
    "psC_h24_up": ("h", 24, "300 500", UPSTREAM), "psC_h24_tr": ("h", 24, "300 500", TR),
    "psC_e24_up": ("e", 24, "300 500", UPSTREAM), "psC_e24_tr": ("e", 24, "300 500", TR),
    "psT_h48": ("h", 48, TEMPS, TR), "psT_e48": ("e", 48, TEMPS, TR),
}


def finished(out):
    return out.exists() and "Total program execution" in out.read_text(errors="ignore")


def make(name, carrier, nk, temps):
    d = MOB / name
    if (d / "epw.in").exists():
        return d
    subprocess.run([BASE / "make_si_run.sh", name, str(STAGE1), carrier, str(nk)], check=True)
    t = (d / "epw.in").read_text()
    t = t.replace("temps       = 300", f"temps       = {temps}")
    t = re.sub(r"fsthick\s*=\s*\S+", "fsthick     = " + ("0.6" if carrier == "h" else "0.8"), t)     # 12-15 kT at 500 K
    t = re.sub(r"ncarrier\s*=\s*\S+", "ncarrier         = " + ("-1E15" if carrier == "h" else "+1E15"), t)
    (d / "epw.in").write_text(t)
    return d


def run():
    c.require_private_repo()
    for name, (carrier, nk, temps, exe) in RUNS.items():
        d = make(name, carrier, nk, temps)
        if finished(d / "epw.out"):
            print("done", name, flush=True)
            continue
        env = {**ENV, **(TR_ENV if exe == TR else {})}
        x = sum((["-x", k] for k in env), [])
        rc = c.run_solver(["mpirun", *x, "-np", "8", exe, "-nk", "8", "-in", "epw.in"], d, d / "epw.out", lock="build", env=env)
        print(name, "rc", rc, flush=True)


def stage1():
    """Wannierisation and the matrix elements again with the unmodified build, from the same wavefunctions and phonons; written next to the production files for a bit-level comparison."""
    c.require_private_repo()
    d = BASE / "work_upstream"
    d.mkdir(exist_ok=True)
    for name in ("si.save", "save"):
        if not (d / name).exists():
            (d / name).symlink_to(BASE / "work" / name)
    shutil.copy(BASE / "work/epw1.in", d / "epw1.in")
    if not finished(d / "epw1.out"):
        x = sum((["-x", k] for k in ENV), [])
        rc = c.run_solver(["mpirun", *x, "-np", "8", UPSTREAM, "-nk", "8", "-in", "epw1.in"], d, d / "epw1.out", lock="build", env=ENV)
        print("stage 1 rc", rc, flush=True)
    compare_stage1(d)


def compare_stage1(d):
    """Largest relative difference of the Wannier-space e-ph matrix, between the unmodified rerun and the file the transport runs used."""
    a = np.memmap(d / "si.epmatwp", dtype=np.complex128, mode="r")
    b = np.memmap(BASE / "work/si.epmatwp", dtype=np.complex128, mode="r")
    if a.size != b.size:
        print("sizes differ", a.size, b.size)
        return
    step = 1 << 24
    diff = max(float(np.abs(a[i:i + step] - b[i:i + step]).max()) for i in range(0, a.size, step))
    rel = diff / max(float(np.abs(b[i:i + step]).max()) for i in range(0, b.size, step))
    hr = (d / "si_hr.dat").read_text().splitlines()[1:], (BASE / "work/si_hr.dat").read_text().splitlines()[1:]
    print(f"si.epmatwp max relative difference {rel:.2e}; si_hr.dat identical apart from its date line: {hr[0] == hr[1]}")
    (c.OUT / "stage1_unmodified_vs_production.txt").write_text(f"max relative difference of si.epmatwp: {rel:.3e}\nsi_hr.dat identical apart from its date line: {hr[0] == hr[1]}\n")


if __name__ == "__main__":
    {"run": run, "stage1": stage1}[sys.argv[1]]()
