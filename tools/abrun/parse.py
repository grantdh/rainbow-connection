"""Parsers for out.txt (QE), err.txt (apple-bottom AB_REPORT / AB_PROFILE / dyld), AB_PROFILE_JSON (v1 and
v2 schemas) and AB_TRACE headers, plus the GPU verdict.  Every regex is pinned by tests/test_parse.py.

GPU use is only ever derived from the library's own counters, never from the requested mode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import exlib

FALLBACK_KEYS = ("no_device", "too_large", "gpu_failed", "other")
STAGE_KEYS = ("init", "alloc", "pack", "upload", "kernel", "download", "epilogue", "other")

RE_DURATION = re.compile(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:([\d.]+)s)?")
RE_WALL = re.compile(r"PWSCF\s*:\s*(.*?) CPU\s*(.*?) WALL")
RE_SCF_E = re.compile(r"^\s+total energy\s+=\s+(-?[\d.]+) Ry", re.M)
RE_SCF_ACC = re.compile(r"estimated scf accuracy\s+<\s+([\d.Ee+-]+) Ry")
RE_ITER = re.compile(r"iteration #\s*(\d+)")
RE_NPROC = re.compile(r"running on\s+(\d+) processor")
RE_NPOOL = re.compile(r"npool\s*=\s*(\d+)")
RE_NKS = re.compile(r"number of k points=\s*(\d+)")
RE_NPW = re.compile(r"\(\s*(\d+) PWs\)")
RE_NGM = re.compile(r"Dense\s+grid:\s+(\d+) G-vectors")
RE_NKB = re.compile(r"number of Kohn-Sham states=\s*(\d+)")
RE_FORCE = re.compile(r"atom\s+(\d+) type\s+\d+\s+force =\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")
RE_SCF_CORR = re.compile(r"Total SCF correction\s*=\s*([\d.Ee+-]+)")
RE_STRESS = re.compile(r"total\s+stress.*P=\s*(-?[\d.]+)")
RE_MAXSTEP = re.compile(r"^\s*electron_maxstep\s*=\s*(\d+)", re.M | re.I)

RE_AB_HEADER = re.compile(r"^AB_REPORT\[(\d+)\] (apple-bottom \S+ \([^)]*\)) AB_MODE=(\S+)", re.M)
RE_AB_ROUTINE = re.compile(r"^AB_REPORT\[(\d+)\] (\w+) calls=(\d+) gpu_calls=(\d+) cpu_calls=(\d+) gpu_calls%=\S+ "
                           r"total_ms=([\d.]+) gpu_ms=([\d.]+) cpu_ms=([\d.]+) gpu_time%=\S+ fallbacks=(\d+)", re.M)
RE_AB_STAGES = re.compile(r"^AB_REPORT\[(\d+)\] (\w+) gpu_stages_ms init=([\d.]+) alloc=([\d.]+) pack=([\d.]+) "
                          r"upload=([\d.]+) kernel=([\d.]+) download=([\d.]+) epilogue=([\d.]+) other=([\d.]+)", re.M)
RE_AB_FALLBACKS = re.compile(r"^AB_REPORT\[(\d+)\] (\w+) fallbacks no_device=(\d+) too_large=(\d+) gpu_failed=(\d+) other=(\d+)", re.M)
RE_PER_CALL = re.compile(r"^AB_PROFILE \w+ M=", re.M)
RE_AB_WARN = re.compile(r"apple-bottom: AB_(DISPATCH_TABLE|TRACE|TEST_FAIL_STAGE)[^\n]*")
RE_LIB_LOADED = re.compile(r"(/\S*libapplebottom\S*\.dylib)")
RE_JSON_WRITTEN = re.compile(r"ab_prof: JSON written to (\S+)")


# --------------------------------------------------------------------------- out.txt

def parse_duration(s: str | None) -> float | None:
    """'2m27.89s' -> 147.89; '2m 1.49s' -> 121.49; '1h 2m 3.50s' -> 3723.5; '12.34s' -> 12.34."""
    if s is None:
        return None
    m = RE_DURATION.match(s.strip())
    if not m or not any(m.groups()):
        return None
    h, mi, se = m.groups()
    return (int(h or 0) * 3600) + (int(mi or 0) * 60) + float(se or 0)


def parse_out(text: str, electron_maxstep: int | None = None, input_text: str | None = None) -> dict:
    """exlib.parse_pw plus per-iteration series, WALL seconds, run sizes, convergence, forces and stress."""
    r = exlib.parse_pw(text)
    out = {
        "total_energy_Ry": r.get("total_energy_Ry"), "scf_iterations": r.get("scf_iterations"),
        "homo_eV": r.get("homo_eV"), "lumo_eV": r.get("lumo_eV"), "gap_eV": r.get("gap_eV"),
        "qe_version": r.get("qe_version"), "job_done": bool(r.get("job_done")),
        "cpu_time": r.get("cpu_time"), "wall_time": r.get("wall_time"),
    }
    out["scf_energies"] = [float(x) for x in RE_SCF_E.findall(text)]
    out["scf_accuracy"] = [float(x) for x in RE_SCF_ACC.findall(text)]
    iters = RE_ITER.findall(text)
    if electron_maxstep is None and input_text:
        m = RE_MAXSTEP.search(input_text)
        electron_maxstep = int(m.group(1)) if m else 100
    out["progress"] = {"iteration": int(iters[-1]) if iters else 0, "max": electron_maxstep or 100,
                       "scf_accuracy_Ry": out["scf_accuracy"][-1] if out["scf_accuracy"] else None}
    m = RE_NPROC.search(text)
    out["nproc"] = int(m.group(1)) if m else None
    m = RE_NPOOL.search(text)
    out["npool"] = int(m.group(1)) if m else 0
    m = RE_NKS.search(text)
    out["nks"] = int(m.group(1)) if m else None
    npws = [int(x) for x in RE_NPW.findall(text)]
    out["npw_max"] = max(npws) if npws else None
    m = RE_NGM.search(text)
    out["ngm"] = int(m.group(1)) if m else None
    m = RE_NKB.search(text)
    out["nbnd"] = int(m.group(1)) if m else None
    out["converged"] = not re.search(r"convergence NOT achieved", text)
    out["ieee_note"] = bool(re.search(r"IEEE_\w+_FLAG", text))
    m = RE_WALL.search(text)
    out["wall_qe_s"] = parse_duration(m.group(2)) if m else None
    out["cpu_qe_s"] = parse_duration(m.group(1)) if m else None
    out["gamma_only"] = bool(re.search(r"gamma-point specific algorithms are used", text))
    m = re.search(r"(\d+) Sym\. Ops\.", text)
    out["nsym"] = int(m.group(1)) if m else None
    # forces: the block after 'Forces acting on atoms'
    out["forces"], out["total_scf_correction"] = None, None
    i = text.find("Forces acting on atoms")
    if i >= 0:
        block = text[i:]
        j = block.find("Total force")
        totals = block[: j if j > 0 else None]
        # verbosity='high' prints the per-term decompositions ("The non-local contrib.  to forces", ionic, local,
        # core correction, Hubbard, SCF correction; PW/src/forces.f90:360-392) between the total forces and the
        # 'Total force =' line: keep only the first block (the totals).
        m = re.search(r"^\s+The \S", totals, re.M)
        rows = RE_FORCE.findall(totals[: m.start() if m else None])
        if rows:
            out["forces"] = [[float(a), float(b), float(c)] for _, a, b, c in rows]
        m = RE_SCF_CORR.search(block)
        out["total_scf_correction"] = float(m.group(1)) if m else None
    # stress: the three rows after 'total   stress ... P= ...' (Ry/bohr^3 then kbar; keep the kbar triplet)
    out["stress_kbar"], out["pressure_kbar"] = None, None
    m = RE_STRESS.search(text)
    if m:
        out["pressure_kbar"] = float(m.group(1))
        rows = []
        for line in text[m.end():].splitlines()[1:4]:
            v = line.split()
            if len(v) >= 6:
                try:
                    rows.append([float(v[3]), float(v[4]), float(v[5])])
                except ValueError:
                    break
        if len(rows) == 3:
            out["stress_kbar"] = rows
    return out


# --------------------------------------------------------------------------- err.txt

def _empty_routine() -> dict:
    return {"calls": 0, "gpu_calls": 0, "cpu_calls": 0, "fallbacks": None, "fallback_total": 0,
            "total_ms": 0.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "gpu_stages_ms": None, "pids": []}


def parse_err(text: str) -> dict:
    """AB_REPORT lines summed per routine over pids, per-call line count, warnings, dyld library path."""
    headers = [{"pid": int(p), "build": b, "ab_mode": m} for p, b, m in RE_AB_HEADER.findall(text)]
    routines: dict[str, dict] = {}
    for pid, name, calls, gpu, cpu, tot, gms, cms, fb in RE_AB_ROUTINE.findall(text):
        r = routines.setdefault(name, _empty_routine())
        r["calls"] += int(calls); r["gpu_calls"] += int(gpu); r["cpu_calls"] += int(cpu)
        r["total_ms"] += float(tot); r["gpu_ms"] += float(gms); r["cpu_ms"] += float(cms)
        r["fallback_total"] += int(fb)
        r["pids"].append(int(pid))
    for m in RE_AB_STAGES.finditer(text):
        r = routines.setdefault(m.group(2), _empty_routine())
        vals = [float(x) for x in m.groups()[2:]]
        st = r["gpu_stages_ms"] or dict.fromkeys(STAGE_KEYS, 0.0)
        for k, v in zip(STAGE_KEYS, vals):
            st[k] += v
        r["gpu_stages_ms"] = st
    for m in RE_AB_FALLBACKS.finditer(text):
        r = routines.setdefault(m.group(2), _empty_routine())
        fb = r["fallbacks"] or dict.fromkeys(FALLBACK_KEYS, 0)
        for k, v in zip(FALLBACK_KEYS, m.groups()[2:]):
            fb[k] += int(v)
        r["fallbacks"] = fb
    warnings = [m.group(0).strip() for m in RE_AB_WARN.finditer(text)]
    npids = len({h["pid"] for h in headers})
    for name, r in routines.items():
        if headers and len(set(r["pids"])) < npids:
            warnings.append(f"routine {name} reported by {len(set(r['pids']))} of {npids} AB_REPORT pids")
    dyld = [line for line in text.splitlines() if line.startswith("dyld[")]
    lib_lines = [line for line in dyld if "libapplebottom" in line]
    m = RE_LIB_LOADED.search("\n".join(lib_lines)) if lib_lines else None
    mj = RE_JSON_WRITTEN.search(text)
    return {
        "headers": headers, "ranks_reporting": npids,
        "build": headers[0]["build"] if headers else None,
        "ab_mode_echo": headers[0]["ab_mode"] if headers else None,
        "routines": routines,
        "ab_report_empty": "no BLAS-ABI calls recorded" in text.lower() or "No BLAS-ABI calls recorded" in text,
        "per_call_lines": len(RE_PER_CALL.findall(text)),
        "json_written": mj.group(1) if mj else None,
        "warnings": warnings,
        "lib_loaded": m.group(1) if m else None,
        "dyld_lines": len(dyld), "libapplebottom_dyld_lines": len(lib_lines),
    }


# --------------------------------------------------------------------------- AB_PROFILE_JSON

def parse_profile_json(path: Path | str) -> dict:
    """Normalise v1 (gpu_dispatch_count/amx_fallback_count, no build) and v2 (gpu_calls/cpu_calls/fallbacks/
    gpu_stage_ns/build).  Anything else -> {"schema": None, "error": ...} (never a silent 0 GPU calls)."""
    p = Path(path)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return {"schema": None, "error": f"cannot read {p}: {exc}", "routines": {}}
    rs = d.get("routines")
    if not isinstance(rs, list):
        return {"schema": None, "error": "unknown profile schema: no routines list", "routines": {}}
    schema = None
    if any("gpu_calls" in r for r in rs):
        schema = "v2"
    elif any("gpu_dispatch_count" in r for r in rs):
        schema = "v1"
    elif not rs:
        schema = "v2" if "build" in d else None
    if schema is None:
        return {"schema": None, "error": "unknown profile schema", "routines": {}}
    routines: dict[str, dict] = {}
    for r in rs:
        name = r.get("name", "?")
        fb = r.get("fallbacks") if isinstance(r.get("fallbacks"), dict) else None
        gpu_ns = r.get("gpu_ns")
        cpu_ns = r.get("cpu_ns")
        routines[name] = {
            "calls": int(r.get("calls", 0)),
            "gpu_calls": int(r.get("gpu_calls", r.get("gpu_dispatch_count", 0))),
            "cpu_calls": int(r.get("cpu_calls", r.get("amx_fallback_count", 0))),
            "fallbacks": {k: int(fb.get(k, 0)) for k in FALLBACK_KEYS} if fb else None,
            "fallback_total": sum(int(v) for v in fb.values()) if fb else 0,
            "total_ms": r.get("total_ns", 0) / 1e6,
            "gpu_ms": gpu_ns / 1e6 if gpu_ns is not None else None,
            "cpu_ms": cpu_ns / 1e6 if cpu_ns is not None else None,
            "gpu_stage_ns": r.get("gpu_stage_ns"),
            "max_m": r.get("max_m"), "max_n": r.get("max_n"), "max_k": r.get("max_k"),
            "avg_m": r.get("avg_m"), "avg_n": r.get("avg_n"), "avg_k": r.get("avg_k"),
            "dim_histogram": r.get("dim_histogram"), "ns_by_bin": r.get("ns_by_bin"),
        }
    return {"schema": schema, "build": d.get("build"), "timestamp": d.get("timestamp"), "pid": d.get("pid"),
            "threshold_ns": d.get("threshold_ns"), "routines": routines, "error": None}


# --------------------------------------------------------------------------- AB_TRACE

def parse_trace_headers(run_dir: Path | str) -> list[dict]:
    """One record per ab_trace.<pid>: build, AB_MODE, effective AB_MIN_GPU_DIM, mpi rank/size, path counts."""
    out = []
    for f in sorted(Path(run_dir).glob("ab_trace.*")):
        rec = {"file": f.name, "build": None, "ab_mode": None, "min_gpu_dim": None, "mpi_rank": None,
               "mpi_size": None, "path_cpu": 0, "path_gpu": 0, "path_fallback": 0, "min_dim_below_floor": 0, "lines": 0}
        try:
            with open(f, errors="replace") as fh:
                for line in fh:
                    if line.startswith("#"):
                        m = re.match(r"# (apple-bottom \S+ \([^)]*\)) AB_TRACE pid=(\d+)", line)
                        if m:
                            rec["build"] = m.group(1)
                        m = re.search(r"AB_MODE=(\S+)", line)
                        if m and "AB_TRACE_HASH" in line:
                            rec["ab_mode"] = m.group(1)
                        m = re.search(r"AB_MIN_GPU_DIM=(\d+)", line)
                        if m:
                            rec["min_gpu_dim"] = int(m.group(1))
                        m = re.search(r"mpi_rank=(\d+) mpi_size=(\d+)", line)
                        if m:
                            rec["mpi_rank"], rec["mpi_size"] = int(m.group(1)), int(m.group(2))
                        continue
                    rec["lines"] += 1
                    if "path=gpu" in line:
                        rec["path_gpu"] += 1
                    elif "path=fallback" in line:
                        rec["path_fallback"] += 1
                    elif "path=cpu" in line:
                        rec["path_cpu"] += 1
                    m = re.search(r"\bM=(\d+)\s+N=(\d+)\s+K=(\d+)", line)
                    if m and rec["min_gpu_dim"] is not None and "path=gpu" in line:
                        if min(int(x) for x in m.groups()) <= rec["min_gpu_dim"]:
                            rec["min_dim_below_floor"] += 1
        except OSError:
            continue
        out.append(rec)
    return out


# --------------------------------------------------------------------------- totals + verdict

def totals_of(routines: dict) -> dict:
    t = {"calls": 0, "gpu_calls": 0, "cpu_calls": 0, "fallbacks": 0, "fallback_breakdown": None}
    bd = None
    for r in routines.values():
        t["calls"] += r.get("calls", 0); t["gpu_calls"] += r.get("gpu_calls", 0); t["cpu_calls"] += r.get("cpu_calls", 0)
        t["fallbacks"] += r.get("fallback_total", 0)
        if r.get("fallbacks"):
            bd = bd or dict.fromkeys(FALLBACK_KEYS, 0)
            for k in FALLBACK_KEYS:
                bd[k] += r["fallbacks"].get(k, 0)
    t["fallback_breakdown"] = bd
    return t


def verify_gpu(mode: str, evidence_source: str, totals: dict | None, has_build_info: bool = True) -> tuple[bool | None, str, str]:
    """(gpu_verified, reason, badge) from the counters alone.  badge: green | amber | red | grey | n/a."""
    if mode == "reference":
        return None, "n/a: apple-bottom-free binary", "n/a"
    if evidence_source == "none" or not totals:
        return None, "unverifiable: no counters (AB_REPORT absent on this build, or ranks>1 without AB_REPORT, or the profile JSON was not written)", "amber"
    calls, gpu, fb = totals["calls"], totals["gpu_calls"], totals["fallbacks"]
    bd = totals.get("fallback_breakdown") or {}
    hard = bd.get("gpu_failed", 0) + bd.get("other", 0)
    if bd:
        hard_desc = f"gpu_failed={bd.get('gpu_failed', 0)} other={bd.get('other', 0)} no_device={bd.get('no_device', 0)} too_large={bd.get('too_large', 0)}"
    else:
        hard_desc = f"fallbacks={fb}"
    if calls == 0:
        # Counters exist but no dgemm/zgemm reached libapplebottom (the library's own "No BLAS-ABI calls recorded."
        # case, or a profile JSON with an empty routines list): nothing was measured, so no mode can pass on it.
        if mode == "ab_gpu_all":
            return False, f"no GEMM calls reached apple-bottom ({evidence_source})", "red"
        if mode in ("ab_cpu", "ab_auto", "ab_gpu"):
            return None, f"no GEMM call reached apple-bottom (calls == 0); nothing measured ({evidence_source})", "amber"
    if mode == "ab_cpu":
        if gpu == 0 and fb == 0:
            return True, f"no GPU dispatch, as requested ({calls} GEMM calls on Accelerate; {evidence_source})", "grey"
        return False, f"unexpected: {gpu}/{calls} GPU calls, {fb} fallbacks in AB_MODE=cpu ({evidence_source})", "red"
    if mode == "ab_auto":
        if fb:
            return False, f"auto: {fb} fallbacks ({hard_desc}); {gpu}/{calls} on the GPU ({evidence_source})", "red"
        if gpu == 0:
            why = ("auto routed 0/{n} GEMMs to the GPU (calibrated table has one row zgemm T T 4096^3; auto == cpu on this machine)"
                   if has_build_info else "auto routed 0/{n} GEMMs to the GPU (pre-1.4.0 fixed FLOP thresholds; informational)")
            return True, why.format(n=calls) + f" ({evidence_source})", "grey"
        if has_build_info:
            return False, f"auto routed {gpu}/{calls} GEMMs to the GPU; on a calibrated-table dylib (ab_build_info present) every QE shape should stay on Accelerate ({evidence_source})", "amber"
        return None, f"auto routed {gpu}/{calls} GEMMs to the GPU (pre-1.4.0 fixed FLOP thresholds; informational, not gated) ({evidence_source})", "amber"
    if mode == "ab_gpu":
        if hard or (not bd and fb):
            return False, f"GPU fallbacks: {hard_desc} ({evidence_source})", "red"
        if gpu == 0:
            return True, f"no GEMM above the floor (M,N,K <= AB_MIN_GPU_DIM): 0/{calls} on the GPU, expected on this cell ({evidence_source})", "grey"
        note = f"; benign too_large={bd.get('too_large', 0)}" if bd.get("too_large") else ""
        return True, f"{gpu}/{calls} GEMMs above the floor ran on the GPU, {hard_desc}{note} ({evidence_source})", ("amber" if note else "green")
    if mode == "ab_gpu_all":
        if bd.get("gpu_failed", 0) or bd.get("other", 0) or bd.get("no_device", 0):
            return False, f"GPU fallbacks: {hard_desc}; {gpu}/{calls} on the GPU ({evidence_source}); gpu_failed>0 is a bug to report", "red"
        if gpu == calls and (fb == 0 or (bd and fb == bd.get("too_large", 0))):
            if bd.get("too_large"):
                return True, f"gpu_calls == calls ({calls}) with benign too_large fallbacks ({bd['too_large']}: operand > 2^31-1 elements or maxBufferLength) ({evidence_source})", "amber"
            return True, f"gpu_calls == calls ({calls}) and fallbacks == 0 ({evidence_source})", "green"
        if gpu < calls and bd and fb == bd.get("too_large", 0) and gpu + fb == calls:
            return True, f"{gpu}/{calls} on the GPU; the rest are benign too_large fallbacks ({fb}) ({evidence_source})", "amber"
        return False, f"only {gpu}/{calls} GEMMs ran on the GPU with AB_MIN_GPU_DIM=0 ({hard_desc}) ({evidence_source})", "red"
    return None, f"unknown mode {mode}", "amber"


def ab_record(mode: str, binary: dict, run_dir: Path | str, err_text: str, profile_path: Path | str | None = None,
              min_gpu_dim_effective=None) -> dict:
    """The 'ab' section of run.json: evidence, normalised routines/totals, verdict, library identity."""
    run_dir = Path(run_dir)
    err = parse_err(err_text)
    prof = parse_profile_json(profile_path) if profile_path and Path(profile_path).exists() else None
    warnings = list(err["warnings"])
    if prof and prof.get("schema"):
        source, routines, schema, build = "profile_json", prof["routines"], prof["schema"], prof.get("build") or err["build"]
    else:
        if prof and prof.get("error"):
            warnings.append(prof["error"])
        if err["routines"]:
            source, routines, schema, build = "ab_report", err["routines"], None, err["build"]
        else:
            source, routines, schema, build = "none", {}, None, err["build"]
    totals = totals_of(routines) if source != "none" else None
    dylib_build = binary.get("ab_build_info") if binary else None
    has_build = bool(dylib_build)
    verified, reason, badge = verify_gpu(mode, source, totals, has_build_info=has_build)
    if mode != "reference" and source != "none" and totals and binary.get("dylib_sha256"):
        reason += f" (dylib sha {binary['dylib_sha256'][:12]})"
    lib_loaded = err["lib_loaded"]
    dylib_path = binary.get("dylib_path") if binary else None
    if mode == "reference":
        lib_ok = (err["libapplebottom_dyld_lines"] == 0) if err["dyld_lines"] else None
    elif lib_loaded is None or dylib_path is None:
        lib_ok = None
    else:
        try:
            lib_ok = Path(lib_loaded).resolve() == Path(dylib_path).resolve()
        except OSError:
            lib_ok = lib_loaded == dylib_path
    build_matches = (build == dylib_build) if (build and dylib_build) else None
    trace = parse_trace_headers(run_dir)
    return {
        "evidence_source": source, "schema": schema, "build": build, "build_matches_dylib": build_matches,
        "ab_mode_echo": err["ab_mode_echo"], "min_gpu_dim_effective": min_gpu_dim_effective,
        "routines": routines, "totals": totals, "gpu_verified": verified, "gpu_verified_reason": reason, "gpu_badge": badge,
        "lib_loaded": lib_loaded, "lib_ok": lib_ok, "per_call_lines": err["per_call_lines"],
        "ranks_reporting": err["ranks_reporting"], "ab_report_empty": err["ab_report_empty"],
        "json_written": err["json_written"], "trace": trace, "warnings": warnings,
    }
