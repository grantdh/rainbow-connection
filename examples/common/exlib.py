"""Shared helpers for the worked examples.

Every example driver imports this file by path (no package install needed):

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common")); import exlib

Paths come from the environment when set (same variable names as env.sh) and
otherwise from the layout described in ../../env.sh and ~/Dev/PROTOCOL.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]            # rainbow-connection checkout
EXAMPLES = REPO_ROOT / "examples"
HC_EV_UM = 1.23984198                                        # h c in eV·µm
HC_EV_NM = 1239.84198


def _p(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


QE_BIN = _p("SF_QE_BIN", HOME / "qe-test/q-e-qe-7.4.1/bin")
YAMBO_BIN = _p("SF_YAMBO_BIN", HOME / "yambo-build/yambo/bin")
MEEP_PYTHON = _p("MUPPET_MEEP_PYTHON", HOME / "miniconda3_arm64/envs/meep_arm/bin/python")
# apple-bottom-linked pw.x, frozen with its own dylib (never the live build/ of the library repo).
AB_FROZEN = _p("MUPPET_AB_FROZEN", HOME / "qe-test/ab-frozen/e56d1b46c753")
STUDIO = _p("MUPPET_STUDIO", HOME / "Dev/Claude/muppet-studio")
PSEUDO_SI = REPO_ROOT / "workflows/si-photovoltaic/inputs/pseudo/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
RUN_LOG = EXAMPLES / "common" / "solver-runs.log"

SOLVERS = {"pw.x", "ph.x", "epw.x", "yambo", "p2y", "ypp", "mpirun", "prterun", "orterun"}


# --------------------------------------------------------------------------- machine etiquette

def external_solver_pids(exclude_substring: str = "examples/") -> list[tuple[int, str]]:
    """Solver processes started by someone else (the si64 timing campaign, for instance)."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        exe = os.path.basename(cmd.split()[0]) if cmd.split() else ""
        if exe in SOLVERS and exclude_substring not in cmd:
            found.append((int(pid), cmd[:120]))
    return found


def wait_for_idle(max_wait_s: int = 1800, poll_s: int = 20, log=print) -> bool:
    """Block while another session's solver is running. Returns True if it is safe to start."""
    t0 = time.time()
    while True:
        busy = external_solver_pids()
        if not busy:
            return True
        if time.time() - t0 > max_wait_s:
            log(f"  [guard] still busy after {max_wait_s}s: {busy[:2]} -> proceeding anyway (noted in log)")
            return False
        log(f"  [guard] external solver running ({busy[0][1][:60]}...), waiting {poll_s}s")
        time.sleep(poll_s)


def load1() -> float:
    return os.getloadavg()[0]


def run(cmd: list[str], cwd: Path, log_file: Path, env: dict | None = None, label: str = "",
        nice: int = 10, timeout: int = 3600, stdin_file: Path | None = None) -> tuple[int, float, str]:
    """Run a solver with low priority, record it in the shared run log, return (rc, wall_s, text)."""
    cwd.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    full_env.update(env or {})
    argv = ["nice", "-n", str(nice)] + [str(c) for c in cmd]
    was_idle = wait_for_idle()
    t0 = time.time()
    with open(log_file, "w") as fh, (open(stdin_file) if stdin_file else open(os.devnull)) as fin:
        proc = subprocess.run(argv, cwd=cwd, env=full_env, stdout=fh, stderr=subprocess.STDOUT,
                              stdin=fin, timeout=timeout)
    wall = time.time() - t0
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rc={proc.returncode} wall={wall:.1f}s load1={load1():.2f} "
                 f"idle_at_start={was_idle} label={label or cmd[0]} cwd={cwd}\n")
    return proc.returncode, wall, log_file.read_text(errors="replace")


# --------------------------------------------------------------------------- parsers

def parse_pw(text: str) -> dict:
    out: dict = {}
    m = re.findall(r"!\s+total energy\s+=\s+(-?[\d.]+) Ry", text)
    if m:
        out["total_energy_Ry"] = float(m[-1])
    m = re.search(r"highest occupied, lowest unoccupied level \(ev\):\s+(-?[\d.]+)\s+(-?[\d.]+)", text)
    if m:
        out["homo_eV"], out["lumo_eV"] = float(m.group(1)), float(m.group(2))
        out["gap_eV"] = out["lumo_eV"] - out["homo_eV"]
    m = re.search(r"convergence has been achieved in\s+(\d+) iterations", text)
    if m:
        out["scf_iterations"] = int(m.group(1))
    m = re.search(r"PWSCF\s+:\s+(.*?) CPU\s+(.*?) WALL", text)
    if m:
        out["cpu_time"], out["wall_time"] = m.group(1).strip(), m.group(2).strip()
    m = re.search(r"Program PWSCF v\.([\w.]+)", text)
    if m:
        out["qe_version"] = m.group(1)
    out["job_done"] = "JOB DONE" in text
    return out


def read_yambo_eps(path: Path) -> dict:
    """o-*.eps_q1_* (IP or BSE): columns E(eV), Im eps, Re eps, [...]. Header notes the Green's function."""
    rows, resonant_only = [], False
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "Green`s Function" in s and s.rstrip().endswith("Resonant"):
                resonant_only = True
            continue
        v = s.split()
        if len(v) >= 3:
            rows.append([float(v[0]), float(v[1]), float(v[2])])
    a = np.array(rows)
    return {"energy_eV": a[:, 0], "eps2": a[:, 1], "eps1": a[:, 2], "resonant_only": resonant_only,
            "source": str(path)}


def lorentz_eps(energy_eV: np.ndarray, eps_inf: float, poles: list[dict]) -> np.ndarray:
    eps = np.full(np.shape(energy_eV), eps_inf, dtype=complex)
    for p in poles:
        w, g, s = p["omega_eV"], p["gamma_eV"], p["sigma"]
        eps += s * w ** 2 / (w ** 2 - energy_eV ** 2 - 1j * g * energy_eV)
    return eps


def converter():
    """rainbow-connection's Lorentz fitter (lib/converters/yambo_eps_to_meep.py), loaded by path."""
    import importlib.util
    path = REPO_ROOT / "lib/converters/yambo_eps_to_meep.py"
    spec = importlib.util.spec_from_file_location("yambo_eps_to_meep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- optics

def airy_tra(eps: np.ndarray, thickness_nm: float, energy_eV: np.ndarray) -> dict:
    """Free-standing slab in vacuum at normal incidence (e^{-iwt}: Im n >= 0 absorbs)."""
    n = np.sqrt(eps.astype(complex))
    n = np.where(n.imag < 0, -n, n)
    lam_nm = HC_EV_NM / energy_eV
    delta = 2 * np.pi * n * thickness_nm / lam_nm
    r12, r23 = (1 - n) / (1 + n), (n - 1) / (n + 1)
    t12, t23 = 2 / (1 + n), 2 * n / (n + 1)
    ph = np.exp(2j * delta)
    r = (r12 + r23 * ph) / (1 + r12 * r23 * ph)
    t = t12 * t23 * np.exp(1j * delta) / (1 + r12 * r23 * ph)
    R, T = np.abs(r) ** 2, np.abs(t) ** 2
    return {"energy_eV": energy_eV, "T": T, "R": R, "A": 1 - R - T}


# --------------------------------------------------------------------------- reporting

class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, value, criterion: str):
        self.rows.append({"check": name, "pass": bool(passed), "value": value, "criterion": criterion})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {value}  ({criterion})")

    @property
    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.rows)

    def markdown(self) -> str:
        lines = ["| check | result | value | criterion |", "|---|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['check']} | {'PASS' if r['pass'] else 'FAIL'} | {r['value']} | {r['criterion']} |")
        return "\n".join(lines)


def reset_results(res: Path, keep=("run.log",)) -> None:
    """Empty results/ but keep the directory (and the driver's own log, whose fd the shell holds open)."""
    res.mkdir(parents=True, exist_ok=True)
    for child in res.iterdir():
        if child.name in keep:
            continue
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()


def dump_json(path: Path, data) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=default))


def provenance() -> dict:
    def sha(path: Path) -> str | None:
        try:
            return subprocess.run(["shasum", "-a", "256", str(path)], capture_output=True, text=True).stdout.split()[0]
        except Exception:
            return None
    git = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rainbow_connection_commit": git,
        "hardware": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
        "pw_x": str(QE_BIN / "pw.x"), "pw_x_sha256": sha(QE_BIN / "pw.x"),
        "ab_frozen": str(AB_FROZEN), "ab_dylib_sha256": sha(AB_FROZEN / "libapplebottom.dylib"),
        "yambo": str(YAMBO_BIN / "yambo"), "meep_python": str(MEEP_PYTHON),
        "load1_at_start": load1(),
    }
