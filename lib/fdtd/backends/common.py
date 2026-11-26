"""Shared pieces of every FDTD backend: refusals, the capability check, hashing and the run record.

Plain numpy, so it imports under any interpreter (the MEEP backend runs under meep_arm, the Metal ones under the
system python). The record layout is muppet-fdtd-record/0 of docs/FDTD_BACKEND_SPEC.md section 5.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case import Case  # noqa: E402

MU0 = 1.25663706212e-6                  # CODATA 2018, H/m
REFUSED = 3                             # backend exit code: the case needs a feature this engine lacks


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(REFUSED)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_capabilities(case: Case, engine: str, *, dims: set[int], precisions: set[str]) -> None:
    """Refuse (exit 3) unless the case uses only what every backend here implements so far: PEC walls, vacuum,
    soft_e sources, plus the engine's dimensions and precisions."""
    if case.dim not in dims:
        refuse(f"{engine} runs dimensions {sorted(dims)}; the case is {case.dim}-D")
    bad = {f: k for f, k in case.boundary.items() if k != "pec"}
    if bad:
        refuse(f"boundaries {bad}: {engine} has PEC walls only so far")
    if not case.is_vacuum():
        refuse(f"{engine} runs vacuum only so far (no material boxes, eps_r = mu_r = 1, no loss or dispersion)")
    kinds = {s["kind"] for s in case.raw.get("sources", [])} - {"soft_e"}
    if kinds:
        refuse(f"source kinds {sorted(kinds)}: {engine} implements soft_e only")
    if case.raw.get("precision", "any") not in precisions | {"any"}:
        refuse(f"case asks for {case.raw['precision']}; {engine} is {'/'.join(sorted(precisions))}")


def git_state(repo: Path, program: Path) -> dict:
    """commit (None when the repo is not under git) and whether the engine's program file is modified/untracked."""
    def g(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = g("rev-parse", "--short", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "vcs": "none (not a git repository)"}
    status = g("status", "--short", "--", str(program.relative_to(repo)))
    return {"commit": commit, "dirty": bool(status), "program_status": status or "tracked, clean"}


def probe_entries(case: Case, layout: dict[str, int], shapes: dict[str, list], dt_s: float) -> dict:
    """record.probes: per probe its component, stride, array shape, units and spec sample indices."""
    return {name: {"component": case.probe_component[name], "every": every, "shape": shapes[name],
                   "units": "V/m" if case.probe_component[name] in case.e_comps else "A/m",
                   "t0_s": 0.0, "dt_s": dt_s * every, "samples": case.probe_samples[name].tolist()}
            for name, every in layout.items()}


def e_probe_peak(case: Case, arrays: dict[str, np.ndarray]) -> float:
    return max((float(np.abs(v).max()) for k, v in arrays.items()
                if case.probe_component[k] in case.e_comps and v.size), default=0.0)


def write_record(run_dir: Path, case: Case, case_path: Path, *, engine: dict, realised: dict, checks: dict,
                 wall_s: float, throughput_note: str, probes: dict, time_labels: str, **extra) -> dict:
    record = {
        "schema": "muppet-fdtd-record/0",
        "case": str(case_path), "case_sha256": case.sha256,
        "engine": engine,
        "realised": {"c_rel_mismatch": realised["c_m_per_s"] / case.c - 1, "box_m": case.box_m,
                     "steps": case.steps, **realised},
        "checks": checks,
        "wall_s": wall_s, "timing_grade": "observation",
        # Runs the lib/fdtd test runner let overlap with this one (MUPPET_FDTD_JOBS; null when launched another way).
        "concurrent_jobs": int(os.environ["MUPPET_FDTD_JOBS"]) if "MUPPET_FDTD_JOBS" in os.environ else None,
        "throughput_Mcells_per_s": float(np.prod(case.n)) * case.steps / wall_s / 1e6,
        "throughput_note": throughput_note,
        "probes": probes, "time_labels": time_labels, **extra,
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    return record
