"""Pure launch planning: Request -> Plan (argv + env) and the violation list.

No process is started here.  The full command is wrapper_argv + shim_argv + solver_argv:
  [machlock wrapper, bench/build only] /opt/homebrew/bin/python3 ~/qe-test/xp/tools/machlock.py <kind> --
  [always]                             <sys.executable> tools/abrun/shim.py --run-dir <cwd> --nice 10 --
  [solver]                             /opt/homebrew/bin/mpirun -np N -x NAME=value ... <abs pw.x> -in input.in [-nk K]
/usr/bin/nice is never used (SIP strips DYLD_*); the shim calls os.nice(10) itself inside the lock.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import AB_FROZEN_ROOT, DEFAULT_FROZEN, INTERPRETER, MACHLOCK_PYTHON, MPIRUN, SHIM, exlib
from .binaries import Binary
from .presets import PRESETS

NICE = 10
MACHLOCK = exlib.MACHLOCK                     # module attribute so tests can point it elsewhere in-process
LIBRARY_DEFAULT_MIN_GPU_DIM = 32
ENV_REMOVE_RE = re.compile(r"^(AB_|DYLD_)")

MODES: dict[str, dict] = {
    "reference": {
        "id": "reference",
        "label": "CPU only (A): reference pw.x, QE 7.4.1, OpenBLAS for all BLAS/LAPACK, no apple-bottom",
        "short": "reference: QE 7.4.1, OpenBLAS (all BLAS/LAPACK), no apple-bottom",
        "requires": {"ab_linkage": "none"},
        "ab_mode": None,
    },
    "ab_cpu": {
        "id": "ab_cpu",
        "label": "CPU only (B): apple-bottom AB_MODE=cpu, dgemm/zgemm on Accelerate/AMX via libapplebottom, rest OpenBLAS (any apple-bottom-linked pw.x; the static relink exports no counters)",
        "short": "apple-bottom AB_MODE=cpu: dgemm/zgemm on Accelerate/AMX via libapplebottom; rest OpenBLAS",
        "requires": {"ab_linkage": ["dynamic", "static"]},
        "ab_mode": "cpu",
    },
    "ab_auto": {
        "id": "ab_auto",
        "label": "Auto: AB_MODE=auto, calibrated table (1.4.0: expect 0 GPU calls on QE shapes; pre-1.4.0 builds send Davidson ZGEMMs to the GPU)",
        "short": "apple-bottom AB_MODE=auto (calibrated dispatch table)",
        "requires": {"ab_linkage": ["dynamic", "static"]},
        "ab_mode": "auto",
    },
    "ab_gpu": {
        "id": "ab_gpu",
        "label": ("GPU (library floor): AB_MODE=gpu, GEMMs with M,N,K > AB_MIN_GPU_DIM (default 32) on Metal; 0 GPU calls is a "
                  "legitimate result on small cells. Measured on si64 (campaign 2026-10-15, apple-bottom 1.4.0): this mode took "
                  "1.14-1.40x the cpu wall at 1 rank, 1.06-1.21x at 8 ranks"),
        "short": "apple-bottom AB_MODE=gpu at the library floor (AB_MIN_GPU_DIM default 32)",
        "requires": {"ab_linkage": ["dynamic", "static"]},
        "ab_mode": "gpu",
    },
    "ab_gpu_all": {
        "id": "ab_gpu_all",
        "label": "GPU, every GEMM: AB_MODE=gpu AB_MIN_GPU_DIM=0 (parity + dispatch evidence; no bench-grade si64 wall exists for this mode)",
        "short": "apple-bottom AB_MODE=gpu AB_MIN_GPU_DIM=0 (every GEMM on Metal)",
        "requires": {"ab_linkage": ["dynamic", "static"]},
        "ab_mode": "gpu",
    },
}
LOCKS = ("auto", "share", "build", "bench")


@dataclass
class Request:
    preset: str = "si2"
    input_path: str | None = None
    binary_id: str = DEFAULT_FROZEN
    mode: str = "ab_gpu_all"
    ranks: int = 1
    nk: int = 1
    omp: int = 1
    min_gpu_dim: int | None = None
    profile_json: bool = True
    report: bool = True
    trace: bool = False
    verify_dylib: bool = True
    lock: str = "auto"
    timed: bool = False
    allow_omp_gt1: bool = False
    allow_dev: bool = False
    force_profile_json: bool = False
    purge_tmp: bool = True
    label: str = ""
    derived_from: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        r = cls(**known)
        for k in ("ranks", "nk", "omp"):
            setattr(r, k, int(getattr(r, k)))
        if r.min_gpu_dim is not None and r.min_gpu_dim != "":
            r.min_gpu_dim = int(r.min_gpu_dim)
        else:
            r.min_gpu_dim = None
        return r

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    cwd: str
    wrapper_argv: list = field(default_factory=list)
    shim_argv: list = field(default_factory=list)
    solver_argv: list = field(default_factory=list)
    env_set: dict = field(default_factory=dict)
    env_removed: list = field(default_factory=list)
    lock_requested: str = "auto"
    lock_would_resolve_to: str = "share"
    launcher: str = "mpirun"
    timing_grade_possible: bool = False
    omp_effective: int = 1
    min_gpu_dim_effective: str | int = "library default (32)"
    notes: list = field(default_factory=list)
    shell: str = ""

    @property
    def argv(self) -> list[str]:
        return list(self.wrapper_argv) + list(self.shim_argv) + list(self.solver_argv)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["argv"] = self.argv
        return d


# --------------------------------------------------------------------------- lock resolution

def resolve_lock(lock: str, timed: bool, bench_held: bool = False) -> str:
    """auto -> bench if timed; build if someone holds machlock bench right now; else share."""
    if lock == "auto":
        if timed:
            return "bench"
        return "build" if bench_held else "share"
    return lock


def wrapper_for(lock_effective: str) -> list[str]:
    if lock_effective in ("bench", "build") and Path(MACHLOCK).exists() and os.environ.get("MUPPET_NO_MACHLOCK") != "1":
        return [str(MACHLOCK_PYTHON), str(MACHLOCK), lock_effective, "--"]
    return []


def lock_effective_label(lock_effective: str, wrapper_present: bool) -> str:
    """share | machlock-build | machlock-bench | guard (recorded in run.json)."""
    if lock_effective == "share":
        return "share"
    return f"machlock-{lock_effective}" if wrapper_present else "guard"


# --------------------------------------------------------------------------- validation

def _is_under(path: Path, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


def validate(req: Request, binary: Binary | None, cwd: Path | str | None = None) -> list[str]:
    """Every reason this request must not launch (empty list = valid)."""
    v: list[str] = []
    mode = MODES.get(req.mode)
    if mode is None:
        return [f"unknown mode {req.mode!r}; one of {list(MODES)}"]
    if binary is None:
        return [f"unknown binary id {req.binary_id!r}"]
    if req.preset not in PRESETS:
        v.append(f"unknown preset {req.preset!r}; one of {list(PRESETS)}")
    if req.preset == "custom":
        if not req.input_path:
            v.append("custom preset needs input_path")
        elif not Path(req.input_path).is_absolute():
            v.append(f"input_path must be absolute: {req.input_path}")
        elif not Path(req.input_path).is_file():
            v.append(f"input file not found: {req.input_path}")
    if not Path(binary.path).is_absolute():
        v.append(f"binary path must be absolute: {binary.path}")
    if cwd is not None and _is_under(Path(cwd), AB_FROZEN_ROOT):
        v.append(f"run directory {cwd} is under {AB_FROZEN_ROOT}: never run pw.x inside a frozen directory")
    if not binary.launchable:
        v.append(f"binary {binary.id} is not launchable: {binary.launch_reason}")
    if binary.sha_verified is False:
        v.append(f"binary {binary.id}: SHA256SUMS mismatch")
    req_link = mode["requires"]["ab_linkage"]
    if req.mode == "reference":
        if binary.ab_linkage != "none":
            how = "links @rpath/libapplebottom.dylib" if binary.ab_linkage == "dynamic" else f"has {binary.ab_symbols.get('T', 0)} static _ab_ symbols"
            v.append(f"mode reference requires an apple-bottom-free binary; {binary.id} {how}")
    elif binary.ab_linkage not in req_link:
        v.append(f"mode {req.mode} requires an apple-bottom-linked binary; {binary.id} has ab_linkage {binary.ab_linkage}")
    if binary.group == "dev" and not req.allow_dev:
        live = "; loads the LIVE apple-bottom/build dylib" if binary.live_library else ""
        v.append(f"{binary.id} is a dev build (never citable; rebuilt by other sessions{live}); set allow_dev=true to launch it")
    ncpu = os.cpu_count() or 8
    if req.ranks < 1:
        v.append("ranks must be >= 1")
    elif req.ranks > ncpu:
        v.append(f"ranks {req.ranks} > {ncpu} CPUs on this machine")
    if req.omp > ncpu:
        v.append(f"omp {req.omp} > {ncpu} CPUs on this machine")
    if req.min_gpu_dim is not None and req.min_gpu_dim > (1 << 16):
        v.append(f"min_gpu_dim {req.min_gpu_dim} > 65536 (no QE GEMM dimension reaches it; a typo?)")
    if binary is not None and binary.links.get("libmpi") and not MPIRUN.is_file():
        v.append(f"mpirun missing: {MPIRUN} (binary {binary.id} links libmpi)")
    if req.nk < 1:
        v.append("nk must be >= 1")
    if req.nk > max(req.ranks, 1):
        v.append(f"nk {req.nk} > ranks {req.ranks}")
    if req.omp < 1:
        v.append("omp must be >= 1")
    if req.omp > 1 and binary.is_apple_bottom and binary.omp1_only:
        v.append(f"{binary.id} is OMP_NUM_THREADS=1 only (PROVENANCE / -O3 pw.x 2f0f029a): omp {req.omp} refused")
    if req.trace and "AB_TRACE" not in binary.features:
        v.append(f"trace requested but {binary.id}'s dylib has no AB_TRACE feature")
    if req.trace and req.mode == "reference":
        v.append("trace is an apple-bottom feature; not available in mode reference")
    if req.min_gpu_dim is not None:
        if req.mode == "reference":
            v.append("min_gpu_dim is an apple-bottom setting; not available in mode reference")
        elif req.mode == "ab_gpu_all" and req.min_gpu_dim != 0:
            v.append("ab_gpu_all fixes AB_MIN_GPU_DIM=0; use mode ab_gpu for a floor")
        elif req.mode == "ab_gpu" and req.min_gpu_dim == 0:
            v.append("ab_gpu with min_gpu_dim 0 is mode ab_gpu_all; use it so the gate is the right one")
        elif req.min_gpu_dim < 0:
            v.append("min_gpu_dim must be >= 0")
    if req.lock not in LOCKS:
        v.append(f"lock must be one of {LOCKS}")
    if req.timed:
        if req.lock not in ("auto", "bench"):
            v.append(f"timed runs need lock auto or bench (got {req.lock}): PROTOCOL: timing windows are exclusive")
        if not binary.citable:
            v.append(f"timed refused: {binary.id} is class {binary.cls} (not citable: {binary.class_reason})")
        if binary.live_library:
            v.append(f"timed refused: {binary.id} loads the live apple-bottom build")
    return v


# --------------------------------------------------------------------------- planning

def env_for(req: Request, binary: Binary, cwd: Path) -> tuple[dict, list[str], int, str | int]:
    """(env_set in -x order, notes, omp_effective, min_gpu_dim_effective)."""
    notes: list[str] = []
    omp = req.omp
    if binary.is_apple_bottom and omp > 1 and not req.allow_omp_gt1:
        notes.append(f"OMP_NUM_THREADS forced to 1: apple-bottom binaries run at OMP=1 unless allow_omp_gt1 (requested {omp})")
        omp = 1
    if binary.is_apple_bottom and binary.omp1_only and omp != 1:
        omp = 1
    env: dict[str, str] = {
        "OMP_NUM_THREADS": str(omp),
        "OPENBLAS_NUM_THREADS": str(omp),
        "VECLIB_MAXIMUM_THREADS": str(omp),
    }
    min_eff: str | int = "n/a"
    mode = MODES[req.mode]
    if mode["ab_mode"] is None:
        if req.verify_dylib:
            env["DYLD_PRINT_LIBRARIES"] = "1"
        return env, notes, omp, min_eff
    env["AB_MODE"] = mode["ab_mode"]                      # literal cpu | auto | gpu, never anything else
    if req.mode == "ab_gpu_all":
        env["AB_MIN_GPU_DIM"] = "0"
        min_eff = 0
    elif req.min_gpu_dim is not None:
        env["AB_MIN_GPU_DIM"] = str(req.min_gpu_dim)
        min_eff = req.min_gpu_dim
    else:
        min_eff = f"library default ({LIBRARY_DEFAULT_MIN_GPU_DIM})"
    if req.report:
        if "AB_REPORT" in binary.features:
            env["AB_REPORT"] = "1"
        else:
            notes.append(f"AB_REPORT not exported: {binary.id}'s dylib has no AB_REPORT (pre-1.4.0 build)")
    if req.profile_json:
        if req.ranks == 1:
            env["AB_PROFILE"] = "1"
            env["AB_PROFILE_JSON"] = str(Path(cwd) / "ab_profile.json")
        elif req.force_profile_json:
            env["AB_PROFILE"] = "1"
            env["AB_PROFILE_JSON"] = str(Path(cwd) / "ab_profile.json")
            notes.append("WARNING: AB_PROFILE_JSON at ranks > 1 has no pid suffix; the last rank to exit wins (force_profile_json)")
        else:
            notes.append("AB_PROFILE_JSON auto-off at ranks > 1 (no pid suffix, last rank wins); AB_REPORT per rank is summed instead")
    if req.trace and "AB_TRACE" in binary.features:
        env["AB_TRACE"] = str(Path(cwd) / "ab_trace")
    if req.verify_dylib:
        env["DYLD_PRINT_LIBRARIES"] = "1"
    return env, notes, omp, min_eff


def plan(req: Request, binary: Binary, cwd: Path | str, lock_effective: str | None = None,
         environ: dict | None = None, bench_held: bool = False) -> Plan:
    """Pure: the argv/env this request would launch with in `cwd`.  lock_effective overrides the resolution
    (the runner passes the dequeue-time value); environ defaults to os.environ for env_removed."""
    cwd = Path(cwd)
    environ = os.environ if environ is None else environ
    env_set, notes, omp_eff, min_eff = env_for(req, binary, cwd)
    env_removed = sorted(k for k in environ if ENV_REMOVE_RE.match(k))
    lock_eff = lock_effective or resolve_lock(req.lock, req.timed, bench_held)
    wrapper = wrapper_for(lock_eff)
    if lock_eff in ("bench", "build") and not wrapper:
        notes.append(f"lock {lock_eff} requested but machlock is unavailable (MUPPET_NO_MACHLOCK or missing): guard mode")
    if req.lock == "auto" and lock_eff == "build" and not req.timed:
        notes.append("bench held by another session -> this run takes machlock build and waits")
    launcher = "mpirun" if binary.links.get("libmpi") else "direct"
    ranks = req.ranks if launcher == "mpirun" else 1
    if launcher == "direct" and req.ranks > 1:
        notes.append(f"{binary.id} has no Homebrew libmpi: launched directly, ranks forced to 1")
    shim_argv = [INTERPRETER, str(SHIM), "--run-dir", str(cwd), "--nice", str(NICE), "--"]
    if launcher == "mpirun":
        solver = [str(MPIRUN), "-np", str(ranks)]
        for k, val in env_set.items():
            solver += ["-x", f"{k}={val}"]
        solver += [str(binary.path), "-in", "input.in"]
        if req.nk > 1:
            solver += ["-nk", str(req.nk)]
    else:
        solver = [str(binary.path), "-in", "input.in"]
    if binary.omp1_only and req.omp > 1:
        notes.append("OMP_NUM_THREADS forced to 1: -O3 frozen pw.x is OMP=1 only")
    timing_possible = bool(req.timed and lock_eff == "bench" and binary.citable and not binary.live_library)
    if req.timed and not timing_possible:
        notes.append("timed requested but timing grade cannot be bench: see violations/lock")
    p = Plan(cwd=str(cwd), wrapper_argv=wrapper, shim_argv=shim_argv, solver_argv=solver, env_set=env_set,
             env_removed=env_removed, lock_requested=req.lock, lock_would_resolve_to=lock_eff, launcher=launcher,
             timing_grade_possible=timing_possible, omp_effective=omp_eff, min_gpu_dim_effective=min_eff, notes=notes)
    env_part = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_set.items())
    p.shell = f"cd {shlex.quote(str(cwd))} && env {env_part} " + " ".join(shlex.quote(a) for a in p.argv)
    return p


def child_env(env_set: dict, environ: dict | None = None) -> dict:
    """os.environ minus every AB_*/DYLD_* key, plus the mode's NAME=value pairs."""
    base = dict(os.environ if environ is None else environ)
    for k in list(base):
        if ENV_REMOVE_RE.match(k):
            del base[k]
    base.update(env_set)
    return base


def modes_list() -> list[dict]:
    out = []
    for m in MODES.values():
        d = {"id": m["id"], "label": m["label"], "short": m["short"], "requires": m["requires"], "removes": ["AB_*", "DYLD_*"]}
        if m["ab_mode"] is None:
            d["env"] = {"OMP_NUM_THREADS": "<omp>", "OPENBLAS_NUM_THREADS": "<omp>", "VECLIB_MAXIMUM_THREADS": "<omp>",
                        "DYLD_PRINT_LIBRARIES": "1 (verify_dylib)"}
        else:
            d["env"] = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                        "AB_MODE": m["ab_mode"], "AB_REPORT": "1 (if the dylib has it)",
                        "AB_PROFILE": "1 (ranks == 1)", "AB_PROFILE_JSON": "<cwd>/ab_profile.json (ranks == 1)",
                        "DYLD_PRINT_LIBRARIES": "1 (verify_dylib)"}
            if m["id"] == "ab_gpu_all":
                d["env"]["AB_MIN_GPU_DIM"] = "0"
            elif m["id"] == "ab_gpu":
                d["env"]["AB_MIN_GPU_DIM"] = "<n> only when set (library default 32)"
        out.append(d)
    return out
