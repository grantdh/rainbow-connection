"""EPW transport for the power split: mobility vs temperature at the device's carrier density, from stage-1 matrix elements
(symmetric NSCF, lifc = .true.) in STAGE1. Runs the frozen, validated fast build for the 48^3 series and an unmodified-EPW
control on a 24^3 grid at the lowest and highest temperature, which must agree to the printed digits.

  epw_series.py run      the transport runs (nice'd, timing lock honoured)
  epw_series.py stage1   the matrix elements again with the unmodified build, compared with the production file, and the 24^3
                         unmodified control repeated from them, so the mobility shift is measured rather than bounded
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as c
from materials_si import epw_mob, read_hr

BASE = c.HOME / "qe-test/Si_EPW"
PROD = BASE / "work"                   # the stage 1 the transport runs read
STAGE1 = Path(os.environ.get("EPW_STAGE1", PROD))
STAGE1_UP = BASE / "work_upstream"     # the same stage 1 from the unmodified build
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
CONTROLS = ("psC_h24_up", "psC_e24_up")    # repeated from the unmodified stage 1 as psS_*


def finished(out):
    return out.exists() and "Total program execution" in out.read_text(errors="ignore")


def make(name, carrier, nk, temps, stage1=STAGE1):
    d = MOB / name
    if (d / "epw.in").exists():
        return d
    subprocess.run([BASE / "make_si_run.sh", name, str(stage1), carrier, str(nk)], check=True)
    t = (d / "epw.in").read_text()
    t = t.replace("temps       = 300", f"temps       = {temps}")
    t = re.sub(r"fsthick\s*=\s*\S+", "fsthick     = " + ("0.6" if carrier == "h" else "0.8"), t)     # 12-15 kT at 500 K
    t = re.sub(r"ncarrier\s*=\s*\S+", "ncarrier         = " + ("-1E15" if carrier == "h" else "+1E15"), t)
    (d / "epw.in").write_text(t)
    return d


def solve(name, carrier, nk, temps, exe, stage1=STAGE1):
    d = make(name, carrier, nk, temps, stage1)
    if finished(d / "epw.out"):
        print("done", name, flush=True)
        return
    print(name, "rc", epw(exe, d, "epw", {**ENV, **(TR_ENV if exe == TR else {})}), flush=True)


def epw(exe, d, stem, env):
    """One epw.x run on 8 ranks and 8 pools, under the build lock."""
    x = sum((["-x", k] for k in env), [])
    return c.run_solver(["mpirun", *x, "-np", "8", exe, "-nk", "8", "-in", f"{stem}.in"], d, d / f"{stem}.out", lock="build", env=env)


def run():
    c.require_private_repo()
    for name, spec in RUNS.items():
        solve(name, *spec)


def stage1():
    """Wannierisation and the matrix elements again with the unmodified build, from the same wavefunctions and phonons, then the 24^3
    unmodified controls repeated from those files: the measured effect of the build used for stage 1 on the mobility."""
    c.require_private_repo()
    d = STAGE1_UP
    d.mkdir(exist_ok=True)
    for name in ("si.save", "save"):
        if not (d / name).exists():
            (d / name).symlink_to(PROD / name)
    shutil.copy(PROD / "epw1.in", d / "epw1.in")
    if not finished(d / "epw1.out"):
        print("stage 1 rc", epw(UPSTREAM, d, "epw1", ENV), flush=True)
    for name in CONTROLS:
        solve(name.replace("psC", "psS"), *RUNS[name], d)
    compare_stage1()


def wannier_stop(wout):
    """Iteration and total spread (Angstrom^2) at which Wannier90 declared convergence."""
    line = [s for s in wout.read_text().splitlines() if "<-- CONV" in s][-1].split()
    return int(line[0]), float(line[3])


def compare_stage1():
    """The unmodified stage 1 against the file the transport runs used: Wannier-space e-ph matrix, H(R), where Wannier90 stopped, and the
    24^3 unmodified mobilities computed from each."""
    d = STAGE1_UP
    a = np.memmap(d / "si.epmatwp", dtype=np.complex128, mode="r")
    b = np.memmap(PROD / "si.epmatwp", dtype=np.complex128, mode="r")
    if a.size != b.size:
        print("sizes differ", a.size, b.size)
        return
    diff2 = big2 = 0.0
    step = 1 << 24
    for i in range(0, a.size, step):                      # one pass over both 4.6 GB files, squared moduli
        x, y = a[i:i + step], b[i:i + step]
        e = x - y
        diff2 = max(diff2, float((e.real**2 + e.imag**2).max()))
        big2 = max(big2, float((y.real**2 + y.imag**2).max()))
    (ra, _, ha), (rb, _, hb) = read_hr(d / "si_hr.dat"), read_hr(PROD / "si_hr.dat")
    assert np.array_equal(ra, rb)
    res = {"epmatwp_max_abs_diff_over_max_abs": (diff2 / big2) ** 0.5, "hr_max_abs_diff_meV": 1e3 * float(np.abs(ha - hb).max()),
           "hr_max_abs_meV": 1e3 * float(np.abs(hb).max()),
           "wannier90_stop_iteration_and_spread_A2": {"unmodified": wannier_stop(d / "si.wout"), "production": wannier_stop(PROD / "si.wout")}}
    for name in CONTROLS:
        new, old = MOB / name.replace("psC", "psS") / "epw.out", MOB / name / "epw.out"
        if finished(new) and finished(old):
            mn, mo = epw_mob(new), epw_mob(old)
            res[f"mu_{RUNS[name][0]}_24cube"] = {f"{T:g}": {"from_unmodified_stage1": mn[T], "from_production_stage1": mo[T],
                                                            "rel_diff_ibte": mn[T][1] / mo[T][1] - 1} for T in mo}
    s = json.dumps(res, indent=1)
    (c.OUT / "stage1_unmodified_vs_production.json").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    {"run": run, "stage1": stage1}[sys.argv[1]]()
