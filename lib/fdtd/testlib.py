"""Helpers shared by the lib/fdtd test scripts (each stays a standalone PASS/FAIL script).

Gating and the summary are examples/common/exlib.Checks; `check` only maps the scripts' (name, ok, detail) calls
onto it, with ok None meaning a reported (never counted) row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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


def load_yee():
    """examples/yee-dispersion/run.py as a module (matrix pencil, Yee mode tables, load_series)."""
    spec = importlib.util.spec_from_file_location("yee_dispersion", REPO / "examples" / "yee-dispersion" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(checks: exlib.Checks, name: str, ok: bool | None, detail: str, criterion: str = "") -> None:
    if ok is None:
        checks.report(name, detail, criterion or "reported, not gated")
    else:
        checks.add(name, ok, detail, criterion or "see test docstring")


def finish(checks: exlib.Checks) -> int:
    print(f"\n{checks.summary()}")
    for r in checks.gated:
        if not r["pass"]:
            print(f"  FAIL: {r['check']}")
    return 0 if checks.all_pass else 1


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max |b| over the common length."""
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max() / np.abs(b[:n]).max())


def run_backend(checks: exlib.Checks, cmd: list, out: Path, label: str):
    """Launch a backend through exlib.run (machine locks, shared run log); return (record, probes) or (None, None)."""
    rc, wall, text = exlib.run(cmd, out, out / "backend.log", label=label, nice=10, timeout=3600)
    check(checks, f"{label}: backend ran", rc == 0, f"rc={rc}, {wall:.1f} s" + ("" if rc == 0 else f": {text[-400:]}"))
    if rc != 0:
        return None, None
    return json.loads((out / "record.json").read_text()), dict(np.load(out / "probes.npz"))


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


def ringdown_lines(yee, rec: dict, v: np.ndarray, box_m: list[float], two_d: bool = False) -> list[dict]:
    """Matrix-pencil lines of a ring-down series, matched to the discrete Yee modes of the box the engine solved
    (its realised dt and c)."""
    dt = rec["realised"]["dt_s"]
    L = list(box_m) + [1.0] * (3 - len(box_m))            # TM_mn0 in 2-D: Lz never enters
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
