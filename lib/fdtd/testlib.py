"""Helpers shared by the lib/fdtd test scripts (each stays a standalone PASS/FAIL script).

Gating and the summary are examples/common/exlib.Checks; `check` maps the scripts' (name, ok, detail) calls onto it,
with ok None meaning a reported (never counted) row. Each backend suite has two phases: `jobs(argv)` lists its
backend launches and `gate(checks, results, argv)` checks what they produced. `run_suite` runs one suite on its own;
lib/fdtd/test_all.py runs every suite's launches in one pool (concurrency: docs/FDTD_BACKEND_SPEC.md sections 5, 7).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "examples" / "common"))
import exlib  # noqa: E402

CASES = HERE / "cases"
RUNS = HERE / "runs"
XCHECK = REPO / "examples" / "metal-fdtd-crosscheck" / "results"
BACKENDS = HERE / "backends"
INTERPRETER = {"meep": exlib.MEEP_PYTHON}                 # backends run under the system python3 unless listed


@functools.cache
def load_yee():
    """examples/yee-dispersion/run.py as a module (matrix pencil, Yee mode tables, load_series), loaded once."""
    spec = importlib.util.spec_from_file_location("yee_dispersion", REPO / "examples" / "yee-dispersion" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(checks: exlib.Checks, name: str, ok: bool | None, detail: str, criterion: str = "") -> None:
    if ok is None:
        checks.report(name, detail, criterion or "reported, not gated")
    else:
        checks.add(name, ok, detail, criterion or "see test docstring")


def print_failures(checks: exlib.Checks, indent: str = "  ") -> None:
    for r in checks.gated:
        if not r["pass"]:
            print(f"{indent}FAIL: {r['check']}")


def finish(checks: exlib.Checks) -> int:
    print(f"\n{checks.summary()}")
    print_failures(checks)
    return 0 if checks.all_pass else 1


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max |b| over the common length."""
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max() / np.abs(b[:n]).max())


@dataclass
class Job:
    """One backend launch, named by its run directory under RUNS (engine/variant/case)."""
    key: str
    cmd: tuple

    @property
    def out(self) -> Path:
        return RUNS / self.key

    @property
    def long(self) -> bool:                               # the 32,000-step ring-downs start first
        return self.key.endswith("-ringdown")


def backend_job(engine: str, case: str, variant: str = "case", flags: tuple = ()) -> Job:
    """lib/fdtd/backends/<engine>_backend.py on cases/<case>.json, into RUNS/<engine>/<variant>/<case>."""
    key = f"{engine}/{variant}/{case}"
    return Job(key, (INTERPRETER.get(engine, "python3"), BACKENDS / f"{engine}_backend.py", CASES / f"{case}.json",
                     RUNS / key) + tuple(flags))


def unique_jobs(jobs: list[Job]) -> list[Job]:
    """One job per run directory, in listed order; two jobs naming the same directory must be the same launch."""
    seen: dict[str, Job] = {}
    for j in jobs:
        if j.key in seen and seen[j.key].cmd != j.cmd:
            raise ValueError(f"two different launches write {j.key}:\n  {seen[j.key].cmd}\n  {j.cmd}")
        seen.setdefault(j.key, j)
    return list(seen.values())


def jobs_arg(argv: list[str]) -> int:
    """`--jobs N` from the command line; 1 when absent."""
    return max(1, int(argv[argv.index("--jobs") + 1])) if "--jobs" in argv else 1


def effective_jobs(n_jobs: int) -> int:
    """How many launches can overlap: exlib.run serialises every launch under the shared machine lock."""
    return 1 if exlib.lock_mode() == "machlock" else n_jobs


def share_note(n_jobs: int) -> str | None:
    if effective_jobs(n_jobs) < n_jobs:
        return (f"--jobs {n_jobs}: machlock serialises every launch; set MUPPET_SHARE=1 for untimed correctness runs "
                "(PROTOCOL section 9) to run them concurrently")
    return None


def timing_note(rec: dict) -> str:
    """Qualifier for any wall time or throughput taken from a record."""
    n = rec.get("concurrent_jobs")
    if n is None:
        return "observation; launched outside the lib/fdtd test runner, concurrency unknown"
    return f"observation; ran alongside up to {n - 1} other backend runs, not comparable" if n > 1 else "observation"


def run_jobs(checks: exlib.Checks, jobs: list[Job], n_jobs: int) -> dict[str, tuple[dict | None, dict | None]]:
    """Launch the jobs through exlib.run (machine locks, shared run log), up to n_jobs at a time, ring-downs first.
    Each launch learns the effective concurrency (MUPPET_FDTD_JOBS) and records it. The "backend ran" checks are
    added afterwards in the listed order, so the output does not depend on which run finished first.
    Returns {key: (record, probes)}, with (None, None) for a failed launch."""
    unique = unique_jobs(jobs)
    n = effective_jobs(n_jobs)
    env = {"MUPPET_FDTD_JOBS": str(n)}

    def launch(j: Job):
        try:
            return exlib.run(j.cmd, j.out, j.out / "backend.log", label=j.key, env=env)
        except Exception as e:                            # a timeout or launch error is a failed run, not a crash
            return -1, 0.0, repr(e)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {j.key: pool.submit(launch, j) for j in sorted(unique, key=lambda j: not j.long)}
    note = timing_note({"concurrent_jobs": n})
    results = {}
    for j in unique:
        rc, wall, text = futures[j.key].result()
        check(checks, f"{j.key}: backend ran", rc == 0,
              f"rc={rc}, {wall:.1f} s ({note})" + ("" if rc == 0 else f": {text[-400:]}"))
        results[j.key] = ((json.loads((j.out / "record.json").read_text()), dict(np.load(j.out / "probes.npz")))
                          if rc == 0 else (None, None))
    return results


def run_suite(jobs, gate, argv: list[str]) -> int:
    """One suite on its own: launch its jobs, gate them, print the summary. Exit code 0 only if every gate passes."""
    checks = exlib.Checks()
    n = jobs_arg(argv)
    note = share_note(n)
    if note:
        print(note)
    gate(checks, run_jobs(checks, jobs(argv), n), argv)
    return finish(checks)


def readback_checks(checks: exlib.Checks, label: str, rec: dict, source_rel_tol: float = 0.0) -> None:
    """The backend's own read-back: soft source exact (bit-equal, or within source_rel_tol for float64 MEEP),
    conductor and walls zero (<= 1e-12 x the probe peak), and any extra engine check that must be exactly zero."""
    ck = rec["checks"]
    peak = ck["probe_e_max_abs_at_end"]
    sc = ck["source_equivalence"]
    exact = sc is not None and (sc["E_got"] == sc["E_want"]
                                or (sc["rel_err"] is not None and sc["rel_err"] <= source_rel_tol))
    note = " (drive underflows to 0 in float32 at iteration 1)" if sc and sc["E_want"] == 0 else ""
    check(checks, f"{label}: soft source exact", exact,
          f"{sc['E_got']:.9e} vs {sc['E_want']:.9e}{note}" if sc else "no probed source sample")
    check(checks, f"{label}: conductor holds E = 0", ck["pec_max_abs_E"] <= 1e-12 * peak,
          f"max |E| {ck['pec_max_abs_E']:.1e} (probe peak {peak:.2e})")
    check(checks, f"{label}: tangential E = 0 on the walls", ck["wall_max_abs_E"] <= 1e-12 * peak,
          f"max {ck['wall_max_abs_E']:.1e}; grid {rec['realised']['grid']}, box "
          f"{[round(b, 4) for b in rec['realised']['box_m']]} m")
    if "max_abs_E_outside_data_plane" in ck:
        check(checks, f"{label}: planes other than the data plane stay zero",
              ck["max_abs_E_outside_data_plane"] == 0.0, f"{ck['max_abs_E_outside_data_plane']:.1e}")


def ringdown_lines(yee, rec: dict, v: np.ndarray, case, two_d: bool = False) -> list[dict]:
    """Matrix-pencil lines of a ring-down series, matched to the discrete Yee modes of the case box at the engine's
    realised dt and c. The mode tables are yee-dispersion's, which use its own DX: any other dx is refused."""
    if case.dx != yee.DX:
        raise ValueError(f"{case.raw['name']}: dx = {case.dx} m, but yee-dispersion's Yee mode tables use {yee.DX} m")
    dt = rec["realised"]["dt_s"]
    L = list(case.box_m) + [1.0] * (3 - len(case.box_m))  # TM_mn0 in 2-D: Lz never enters
    lines = yee.pencil_lines(np.arange(len(v)) * dt, v, yee.T_RINGDOWN)
    return yee.match_lines(lines, yee.modes_all_parity(L, dt, rec["realised"]["c_m_per_s"], two_d=two_d))


def gate_ringdown(checks: exlib.Checks, yee, label: str, rows: list[dict], suffix: str = "") -> None:
    gated = [r for r in rows if r["amp"] >= yee.AMP_GATE]
    worst = max((abs(r["err_pct"]) for r in gated), default=float("nan"))
    top = max(rows, key=lambda r: r["amp"]) if rows else None
    check(checks, f"{label}: ring-down lines on the case box's Yee relation", bool(gated) and worst < 0.01,
          f"{len(gated)} lines amp >= {yee.AMP_GATE}, worst |err| {worst:.2e} %"
          + (f"; strongest {top['f_MHz']:.4f} MHz -> {top['mode']} ({top['err_pct']:+.1e} %)" if top else "") + suffix,
          "every line of amplitude >= 0.10 within 0.01 % of a discrete Yee mode")
