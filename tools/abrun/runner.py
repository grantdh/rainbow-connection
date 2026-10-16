"""Run state machine, the single worker with its persisted queue, execute(), cancel() and run_sync().

States: queued -> waiting_timing_lock -> waiting_guard -> running -> done | failed | cancelled
(cancelling is the transient state while the solver's process tree is being stopped).

Etiquette, in code: ~/Dev/.timing-lock is awaited before every launch, re-checked by the shim inside the
machine lock (exit 75 -> the runner waits again and relaunches) and never discharged by a guard max-wait;
the lock field resolves at dequeue time (auto -> bench if timed, build if someone holds machlock bench,
else share; an explicit share while a bench is held also waits as build unless allow_share_during_bench);
the shim writes machine.json inside the lock and os.nice(10)s itself; load1 and foreign solvers are
sampled every 20 s while running with the run's own process tree (a ppid walk from the launched pid,
refreshed each sample) excluded; every run that started a process appends one exlib-format line to
solver-runs.log whose wall is counted from lock acquisition, never from queue or lock waits.

Cancel is descendant-aware: the solver tree (launcher and ranks, which PRRTE puts in their own process
groups) is SIGINTed directly, never the machlock wrapper first (its subprocess.call would SIGKILL the
launcher and orphan the ranks); SIGTERM and SIGKILL escalate over the whole descendant snapshot.

run_sync(request, run_dir=None) is the blocking entry point the example drivers and the CLI use; the
server's Worker drives the same execute() on a thread.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from . import RUNS_DIR, VERSION, exlib
from . import binaries as binaries_mod
from . import compare as compare_mod
from . import launch, parse, presets
from .events import EventBus, RingBuffer

TIMING_LOCK = exlib.TIMING_LOCK            # module attributes: tests monkeypatch these in-process only
RUN_LOG = exlib.RUN_LOG

POLL_TIMING_LOCK_S = 5.0
POLL_PROC_S = 0.5
CONTENTION_POLL_S = 20.0
GUARD_ANNOUNCE_S = 30.0
CANCEL_SIGTERM_AFTER_S = 10.0
CANCEL_SIGKILL_AFTER_S = 20.0
HOLD_POLL_S = 5.0                          # how often the worker re-checks an orphaned solver after a restart
SHIM_TIMING_LOCK_RC = 75                   # shim.py: timing lock present inside the machine lock -> released, relaunch
MAX_TIMING_LOCK_RELAUNCHES = 50
PROFILE_STREAM_FIRST = 50
PROFILE_SUPPRESS_EVERY = 10000
TAIL_LINES = 25
LOAD1_TIMED_MAX = 1.5

STATES = ("queued", "waiting_timing_lock", "waiting_guard", "running", "cancelling", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
SOLVERS = exlib.SOLVERS
CAMPAIGN_MARKERS = exlib.CAMPAIGN_MARKERS

RE_ITER = re.compile(r"iteration #\s*(\d+)")
RE_ACC = re.compile(r"estimated scf accuracy\s+<\s+([\d.Ee+-]+) Ry")

global_bus = EventBus()


class PlanError(ValueError):
    """The request cannot be launched; .violations lists why."""

    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = list(violations)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json(path: Path, data) -> None:
    """Atomic write through a unique temp name (concurrent writers never share or clobber a temp file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=1, default=str))
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- machine probes

def ps_table() -> dict[int, tuple[int, int, str]]:
    """{pid: (ppid, pgid, command)} from one `ps -axo pid=,ppid=,pgid=,stat=,command=`; {} when ps fails.
    Zombies (stat Z: exited, not yet reaped) are left out: they run nothing and hold nothing."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,stat=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    table: dict[int, tuple[int, int, str]] = {}
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if parts[3].startswith("Z"):
            continue
        table[pid] = (ppid, pgid, parts[4] if len(parts) > 4 else "")
    return table


def descendants(root: int, table: dict | None = None) -> set[int]:
    """Every transitive child of `root` (root excluded) by a ppid walk over one ps snapshot.
    This is the only way to see mpirun's ranks: PRRTE puts each rank in its own process group."""
    table = ps_table() if table is None else table
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgid, _cmd) in table.items():
        children.setdefault(ppid, []).append(pid)
    out: set[int] = set()
    stack = [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _exe_of(cmd: str) -> str:
    parts = cmd.split()
    return os.path.basename(parts[0]) if parts else ""


def foreign_solvers(own_pids: set[int] | None = None, campaigns: bool = True, table: dict | None = None) -> list[list]:
    """[[pid, cmd]...]: solver/campaign processes that are not ours (exlib's list minus `own_pids`, which the
    caller computes as the launched pid plus its descendants; a pgid would miss the ranks)."""
    own = own_pids or set()
    table = ps_table() if table is None else table
    found = []
    for pid in sorted(table):
        ppid, pgid, cmd = table[pid]
        if pid in own or pid == os.getpid():
            continue
        exe = _exe_of(cmd)
        if exe in SOLVERS or (campaigns and any(m in cmd for m in CAMPAIGN_MARKERS)):
            found.append([pid, cmd[:160]])
    return found


def pgid_members(pgid: int) -> list[int]:
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[1].isdigit() and int(p[1]) == pgid:
            pids.append(int(p[0]))
    return pids


def live_pids(process: dict | None, run_dir: Path | str | None = None, table: dict | None = None) -> list[int]:
    """Which of a run's recorded pids (launched pid + descendant snapshot) are still alive and still look like
    ours: in our recorded tree/pgid, a solver by name, or naming the run directory.  Guards pid reuse after a
    server restart; an empty list means the solver tree is gone."""
    if not process or process.get("exited"):
        return []
    pids = [p for p in [process.get("pid")] + list(process.get("descendants") or []) if p]
    if not pids:
        return []
    table = ps_table() if table is None else table
    ours = set(pids)
    pgid = process.get("pgid")
    rd = str(run_dir) if run_dir else None
    alive = []
    for pid in pids:
        ent = table.get(pid)
        if ent is None:
            continue
        ppid, pg, cmd = ent
        if ppid in ours or (pgid and pg == pgid) or _exe_of(cmd) in SOLVERS or (rd and rd in cmd):
            alive.append(pid)
    return alive


def machlock_bench_held(machlock: Path | None = None) -> bool:
    """Non-blocking LOCK_SH probe of machlock's machine.lock (exactly muppet-studio's benchmark_running())."""
    machlock = Path(machlock or launch.MACHLOCK)
    if not machlock.exists():
        return False
    lock = machlock.resolve().parent / "machine.lock"
    if not lock.exists():
        return False
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- Run

class Run:
    """One solver run: request, plan, directory, live buffers, and the record that becomes run.json."""

    def __init__(self, run_id: str, request: launch.Request, run_dir: Path, binary: binaries_mod.Binary,
                 prepared: dict, plan: launch.Plan):
        self.id = run_id
        self.request = request
        self.run_dir = Path(run_dir)
        self.binary = binary
        self.prepared = prepared
        self.plan = plan
        self.state = "queued"
        self.state_why = ""
        self.state_since = _now()
        self.created = _now()
        self.started: str | None = None
        self.finished: str | None = None
        self.rc: int | None = None
        self.log = RingBuffer()
        self.bus = EventBus()
        self.cancel_event = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.pgid: int | None = None
        self.caffeinate: subprocess.Popen | None = None
        # The launched process tree, persisted in run.json so a restarted server can tell whether the solver
        # of a job that was running at the previous exit is still alive (pid, pgid of the session leader,
        # descendant snapshot refreshed every sample, wrapper = machlock in front of the shim).
        self.process: dict = {"pid": None, "pgid": None, "descendants": [], "sampled": None, "wrapper": False, "exited": False}
        self.allow_share_during_bench = False
        self.etiquette: dict = {
            "timing_lock_waited_s": 0.0, "guard_waited_s": 0.0, "lock_requested": request.lock,
            "lock_effective": None, "machine_at_acquisition": None, "contention": [], "load1_samples": [],
            "foreign_at_launch": [], "caffeinate": False, "gpu_idle": "unverified", "load1_at_end": None,
            "timing_grade": "untimed", "timing_grade_reason": "not finished",
            "timing_lock_relaunches": 0, "lock_share_promoted_to_build": False,
        }
        self.result: dict | None = None
        self.ab: dict | None = None
        self.files: dict = {"input": "input.in", "out": "out.txt", "err": "err.txt", "profile": None,
                            "machine": "machine.json", "trace": [], "summary": "summary.json", "run": "run.json"}
        self.progress = {"iteration": 0, "max": prepared.get("meta", {}).get("electron_maxstep") or 100, "scf_accuracy_Ry": None}
        self.suppressed_profile_lines = 0
        self.suppressed_dyld_lines = 0
        self.pinned_reference = False
        self.position: int | None = None
        self.wall_runner_s: float | None = None      # from lock acquisition (machine.json) or launch to exit: no waits
        self.wall_total_s: float | None = None       # from execute() start to the terminal state: every wait included
        self._profile_streamed = 0
        self._lock_eff = "share"
        self._t0: float | None = None
        self._t_launch: float | None = None
        self._terminating = False
        self._lock = threading.RLock()

    # ---- state / events
    def set_state(self, state: str, why: str = "") -> None:
        with self._lock:
            self.state, self.state_why, self.state_since = state, why, _now()
        self.publish("abrun.state", {"run_id": self.id, "state": state, "why": why, "since": self.state_since})
        self.save()

    def publish(self, etype: str, data: dict) -> None:
        self.bus.publish(etype, data)
        global_bus.publish(etype, data)

    def guard_line(self, text: str) -> None:
        seq = self.log.append("guard", text)
        self.publish("muppet.log", {"run_id": self.id, "seq": seq, "stream": "guard", "text": text})

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    # ---- records
    def summary(self) -> dict:
        r, ab, e = self.result or {}, self.ab or {}, self.etiquette
        totals = ab.get("totals") or {}
        return {
            "id": self.id, "state": self.state, "state_why": self.state_why, "created": self.created,
            "started": self.started, "finished": self.finished, "preset": self.request.preset,
            "input_sha256": self.prepared.get("input_sha256"), "binary_id": self.binary.id,
            "binary_short": _binary_short(self.binary), "mode": self.request.mode, "ranks": self.request.ranks,
            "nk": self.request.nk, "omp": self.plan.omp_effective, "lock_effective": e.get("lock_effective"),
            "timing_grade": e.get("timing_grade"), "timing_grade_reason": e.get("timing_grade_reason"),
            "total_energy_Ry": r.get("total_energy_Ry"), "scf_iterations": r.get("scf_iterations"),
            "converged": r.get("converged"), "rc": self.rc, "wall_qe_s": r.get("wall_qe_s"), "wall_runner_s": self.wall_runner_s,
            "wall_total_s": self.wall_total_s,
            "gemm_calls": totals.get("calls"), "gpu_calls": totals.get("gpu_calls"), "fallbacks": totals.get("fallbacks"),
            "gpu_verified": ab.get("gpu_verified"), "gpu_badge": ab.get("gpu_badge"), "evidence_source": ab.get("evidence_source"),
            "lib_ok": ab.get("lib_ok"), "pinned_reference": self.pinned_reference,
            "derived_from": self.request.derived_from, "label": self.request.label,
            "expected_E_Ry": r.get("expected_E_Ry"), "dE_vs_expected_Ry": r.get("dE_vs_expected_Ry"),
            "progress": dict(self.progress),
        }

    def record(self) -> dict:
        return {
            "abrun_version": VERSION, "id": self.id, "state": self.state, "state_why": self.state_why,
            "created": self.created, "started": self.started, "finished": self.finished, "rc": self.rc,
            "run_dir": str(self.run_dir), "preset": self.request.preset, "label": self.request.label,
            "input_sha256": self.prepared.get("input_sha256"), "source_sha256": self.prepared.get("source_sha256"),
            "input_source": self.prepared.get("source"), "pseudos": self.prepared.get("pseudos"),
            "input_meta": self.prepared.get("meta"), "request": self.request.to_dict(), "plan": self.plan.to_dict(),
            "binary": self.binary.to_dict(), "etiquette": self.etiquette, "result": self.result, "ab": self.ab,
            "files": self.files, "progress": self.progress, "wall_runner_s": self.wall_runner_s, "wall_total_s": self.wall_total_s,
            "process": dict(self.process), "allow_share_during_bench": self.allow_share_during_bench,
            "pinned_reference": self.pinned_reference, "suppressed_profile_lines": self.suppressed_profile_lines,
            "derived_from": self.request.derived_from,
        }

    def save(self) -> None:
        with self._lock:
            try:
                _write_json(self.run_dir / "run.json", self.record())
            except OSError:
                pass


def _binary_short(b: binaries_mod.Binary) -> str:
    return compare_mod.binary_short({"binary": b.to_dict()})


# --------------------------------------------------------------------------- creating runs

def make_run_id(request: launch.Request, binary: binaries_mod.Binary, parent: Path) -> str:
    bid = "ref" if binary.group == "reference" else binary.id.replace(":", "-")[:6]
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{request.preset}-{request.mode}-{bid}"
    rid, n = base, 1
    while (parent / rid).exists():
        n += 1
        rid = f"{base}-{n}"
    return rid


def _run_dir_violations(run_dir: Path) -> list[str]:
    """A run directory must never be the runs root itself or one of its ancestors (a run.json planted there
    would make DELETE /api/runs/{id} or a later purge remove every run, or the package)."""
    try:
        rd, root = Path(run_dir).resolve(), Path(RUNS_DIR).resolve()
    except OSError as exc:
        return [f"run directory {run_dir}: {exc}"]
    if rd == root:
        return [f"run directory {rd} is the runs root itself; use a subdirectory"]
    if root.is_relative_to(rd):
        return [f"run directory {rd} contains the runs root {root}; use a directory outside it"]
    return []


def new_run(request: launch.Request | dict, run_dir: Path | str | None = None,
            binary: binaries_mod.Binary | None = None, allow_share_during_bench: bool | None = None) -> Run:
    """Validate, prepare the directory (input.in, pseudo/), plan, and write run.json (state queued).
    Raises PlanError with the violation list.  `allow_share_during_bench` (also read from a dict request)
    lets an explicit lock=share launch while another session holds machlock bench; by default it waits."""
    if isinstance(request, dict):
        if allow_share_during_bench is None:
            allow_share_during_bench = bool(request.get("allow_share_during_bench", False))
        req = launch.Request.from_dict(request)
    else:
        req = request
    binary = binary or binaries_mod.get(req.binary_id)
    if binary is None:
        raise PlanError([f"unknown binary id {req.binary_id!r}"])
    if run_dir is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        rid = make_run_id(req, binary, RUNS_DIR)
        run_dir = RUNS_DIR / rid
    else:
        run_dir = Path(run_dir).resolve()
        bad = _run_dir_violations(run_dir)
        if bad:
            raise PlanError(bad)
        rid = make_run_id(req, binary, RUNS_DIR) if not (run_dir / "run.json").exists() else run_dir.name
    violations = launch.validate(req, binary, run_dir)
    if violations:
        raise PlanError(violations)
    try:
        prepared = presets.prepare(req.preset, run_dir, req.input_path, write=True)
    except (presets.PresetError, ValueError) as exc:
        raise PlanError([str(exc)]) from exc
    plan = launch.plan(req, binary, run_dir)
    run = Run(rid, req, run_dir, binary, prepared, plan)
    run.allow_share_during_bench = bool(allow_share_during_bench)
    run.save()
    return run


def plan_only(request: launch.Request | dict, binary: binaries_mod.Binary | None = None) -> dict:
    """POST /api/plan: {plan, violations, run_id_would_be} without creating anything on disk."""
    req = launch.Request.from_dict(request) if isinstance(request, dict) else request
    binary = binary or binaries_mod.get(req.binary_id)
    rid = make_run_id(req, binary, RUNS_DIR) if binary else "invalid"
    cwd = RUNS_DIR / rid
    violations = launch.validate(req, binary, cwd)
    try:
        presets.prepare(req.preset, cwd, req.input_path, write=False)
    except (presets.PresetError, ValueError) as exc:
        violations.append(str(exc))
    if binary is None:
        return {"plan": None, "violations": violations, "run_id_would_be": rid}
    p = launch.plan(req, binary, cwd, bench_held=machlock_bench_held())
    return {"plan": p.to_dict(), "violations": violations, "run_id_would_be": rid}


# --------------------------------------------------------------------------- execution

def wait_for_timing_lock(run: Run, log=None) -> bool:
    """Block while ~/Dev/.timing-lock exists; 5 s cancellable polls; never bypassed. False when cancelled."""
    t0 = time.time()
    announced = False
    while Path(TIMING_LOCK).exists():
        if not announced:
            text = ""
            try:
                text = Path(TIMING_LOCK).read_text(errors="ignore").strip()[:80]
            except OSError:
                pass
            run.set_state("waiting_timing_lock", f"{TIMING_LOCK} present ({text})")
            run.guard_line(f"[guard] {TIMING_LOCK} present ({text}) -> waiting")
            if log:
                log(f"  [guard] {TIMING_LOCK} present -> waiting")
            announced = True
        if run.cancel_event.wait(POLL_TIMING_LOCK_S):
            run.etiquette["timing_lock_waited_s"] = round(time.time() - t0, 1)
            return False
        waited = time.time() - t0
        if waited and int(waited) % int(GUARD_ANNOUNCE_S) < POLL_TIMING_LOCK_S:
            run.guard_line(f"[guard] waiting for {TIMING_LOCK} since {int(waited)} s")
    if announced:
        run.guard_line("[guard] timing lock released")
        if log:
            log("  [guard] timing lock released")
    run.etiquette["timing_lock_waited_s"] = round(time.time() - t0, 1)
    return True


def wait_for_idle_guard(run: Run, log=None) -> bool:
    """exlib.wait_for_idle semantics in 5 s cancellable steps (MUPPET_GUARD_OVERRIDE / MUPPET_GUARD_MAX_WAIT_S).
    MUPPET_GUARD_MAX_WAIT_S only discharges foreign solvers; the timing lock is waited for however long it
    exists (MUPPET_GUARD_OVERRIDE skips the foreign-solver guard, never the lock)."""
    override = os.environ.get("MUPPET_GUARD_OVERRIDE") == "1"
    max_wait = int(os.environ.get("MUPPET_GUARD_MAX_WAIT_S", "0")) or None
    t0 = time.time()
    announced = announced_lock = False
    while True:
        lock_present = Path(TIMING_LOCK).exists()
        busy = [] if override else foreign_solvers()
        if lock_present:
            busy.append([0, f"timing lock present: {TIMING_LOCK}"])
        if not busy:
            if announced or announced_lock:
                run.guard_line(f"[guard] machine free after {time.time() - t0:.0f}s")
            return True
        if max_wait is not None and time.time() - t0 > max_wait and not lock_present:
            run.guard_line(f"[guard] still busy after {max_wait}s: {busy[:2]} -> proceeding (MUPPET_GUARD_MAX_WAIT_S set)")
            return True
        if lock_present and not announced_lock:
            run.guard_line(f"[guard] {TIMING_LOCK} appeared while waiting in guard mode -> waiting (max-wait never discharges it)")
            announced_lock = True
        if not announced and not (lock_present and len(busy) == 1):
            run.guard_line(f"[guard] external solver/campaign running: {busy[0][1][:90]} -> waiting")
            announced = True
        if run.cancel_event.wait(POLL_TIMING_LOCK_S):
            return False


def _reader(run: Run, stream, name: str, path: Path, errors_q: list) -> None:
    """Pump one pipe into its file and the ring buffer; throttle AB_PROFILE per-call lines and dyld noise."""
    try:
        with open(path, "w", errors="replace") as fh:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip("\n")
                fh.write(text + "\n")
                if name == "err":
                    if parse.RE_PER_CALL.match(text):
                        if run._profile_streamed < PROFILE_STREAM_FIRST:
                            run._profile_streamed += 1
                        else:
                            run.suppressed_profile_lines += 1
                            if run.suppressed_profile_lines % PROFILE_SUPPRESS_EVERY == 0:
                                _marker(run)
                            continue
                    elif text.startswith("dyld[") and "libapplebottom" not in text:
                        run.suppressed_dyld_lines += 1
                        continue
                seq = run.log.append(name, text)
                run.publish("muppet.log", {"run_id": run.id, "seq": seq, "stream": name, "text": text})
                if name == "out":
                    m = RE_ITER.search(text)
                    if m:
                        run.progress["iteration"] = int(m.group(1))
                        run.publish("progress", {"run_id": run.id, "value": run.progress["iteration"],
                                                 "max": run.progress["max"], "scf_accuracy_Ry": run.progress["scf_accuracy_Ry"]})
                    m = RE_ACC.search(text)
                    if m:
                        try:
                            run.progress["scf_accuracy_Ry"] = float(m.group(1))
                        except ValueError:
                            pass
    except (OSError, ValueError) as exc:
        errors_q.append(f"{name} reader: {exc}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _marker(run: Run) -> None:
    marker = f"+{run.suppressed_profile_lines} AB_PROFILE per-call lines suppressed, see err.txt"
    seq = run.log.append("err", marker)
    run.publish("muppet.log", {"run_id": run.id, "seq": seq, "stream": "err", "text": marker})


def _kill(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


def _tree_alive(run: Run, table: dict | None = None) -> bool:
    """True while the launched process (or, for an adopted orphan without a Popen, any recorded pid) lives."""
    if run.proc is not None:
        return run.proc.poll() is None
    return bool(live_pids(run.process, run.run_dir, table))


def _sweep(run: Run, snapshot: set[int], root: int | None) -> list[int]:
    """SIGKILL every pid of `snapshot` that is still alive and still ours (a descendant of the snapshot, an
    orphan re-parented to launchd, or a solver by name); returns the pids killed.  A pid that now belongs to
    an unrelated process (ppid outside the snapshot, not a solver) is left alone."""
    table = ps_table()
    ours = set(snapshot) | ({root} if root else set())
    killed = []
    for pid in sorted(snapshot):
        ent = table.get(pid)
        if ent is None:
            continue
        ppid, _pgid, cmd = ent
        if ppid in ours or ppid == 1 or _exe_of(cmd) in SOLVERS:
            _kill(pid, signal.SIGKILL)
            killed.append(pid)
    return killed


def _terminate(run: Run) -> None:
    """Stop the solver tree without orphaning a rank.

    1. snapshot the descendants of the launched pid (ppid walk; mpirun's ranks sit in their own process
       groups, invisible to killpg);
    2. SIGINT the solver processes directly: every descendant when a machlock wrapper is the root (the
       wrapper itself is NOT signalled: its subprocess.call turns KeyboardInterrupt into p.kill() on the
       launcher before mpirun can abort the ranks), the root's group plus descendants otherwise; a root with
       no descendants yet (machlock still waiting, shim before execv) gets the group SIGINT as before;
    3. SIGTERM after CANCEL_SIGTERM_AFTER_S and SIGKILL after CANCEL_SIGKILL_AFTER_S over every descendant
       seen at any point (plus the root and its group on the direct path; a machlock wrapper is spared so it
       can log its release, and gets the group SIGKILL only as the last resort);
    4. a final sweep of the snapshot (only pids that are still ours).
    """
    if run._terminating:
        return
    run._terminating = True
    proc = run.proc
    root = proc.pid if proc is not None else run.process.get("pid")
    if root is None:
        return
    pgid = run.pgid or run.process.get("pgid") or root
    wrapper = bool(run.process.get("wrapper"))
    table = ps_table()
    desc = descendants(root, table)
    snapshot = set(desc) | set(run.process.get("descendants") or [])
    run.process["descendants_at_cancel"] = sorted(snapshot)
    try:
        run.save()
    except Exception:  # noqa: BLE001  (saving the snapshot must never block the signals)
        pass
    t0 = time.time()
    # phase 1: SIGINT the solver tree
    if desc and wrapper:
        for pid in sorted(desc):
            _kill(pid, signal.SIGINT)
    else:
        _killpg(pgid, signal.SIGINT)
        for pid in sorted(desc):
            _kill(pid, signal.SIGINT)
    # phases 2-3: escalate over the solver tree (everything seen at any point); the wrapper is spared so it can
    # log its release and exit by itself once its child is gone, and gets the group kill only as a last resort
    for sig, deadline in ((signal.SIGTERM, CANCEL_SIGTERM_AFTER_S), (signal.SIGKILL, CANCEL_SIGKILL_AFTER_S)):
        while _tree_alive(run) and time.time() - t0 < deadline:
            time.sleep(0.2)
        if not _tree_alive(run):
            break
        snapshot |= descendants(root)
        if not wrapper:
            _killpg(pgid, sig)
            _kill(root, sig)
        for pid in sorted(snapshot):
            _kill(pid, sig)
    deadline = CANCEL_SIGKILL_AFTER_S + 5
    while _tree_alive(run) and time.time() - t0 < deadline:
        time.sleep(0.2)
    if _tree_alive(run) and wrapper:
        _killpg(pgid, signal.SIGKILL)
        _kill(root, signal.SIGKILL)
    snapshot |= descendants(root)
    killed = _sweep(run, snapshot, root)
    if killed:
        run.guard_line(f"[cancel] swept {len(killed)} surviving process(es): {killed}")


def cancel(run: Run) -> str:
    """Non-blocking: flips the cancel event (ends any wait loop; the poll loop stops a process launched after
    the last check) and stops the solver tree on a thread when one is running."""
    if run.terminal:
        return run.state
    run.cancel_event.set()
    if run.state != "cancelling":
        run.set_state("cancelling", "cancel requested")
    if (run.proc is not None and run.proc.poll() is None) or (run.proc is None and live_pids(run.process, run.run_dir)):
        threading.Thread(target=_terminate, args=(run,), name=f"abrun-cancel-{run.id}", daemon=True).start()
    return "cancelling"


def _timing_grade(run: Run, lock_eff: str, wrapper_present: bool) -> tuple[str, str]:
    req, e = run.request, run.etiquette
    if not req.timed:
        return "untimed", f"not timed (lock={e.get('lock_effective')})"
    if lock_eff != "bench" or not wrapper_present:
        return "untimed", f"timed requires machlock bench (lock_effective={e.get('lock_effective')})"
    if not run.binary.citable or run.binary.live_library:
        return "untimed", f"binary {run.binary.id} is class {run.binary.cls}, not citable"
    m = e.get("machine_at_acquisition")
    if not m:
        return "untimed", "no machine.json at acquisition"
    reasons = []
    if m.get("load1", 99) >= LOAD1_TIMED_MAX:
        reasons.append(f"load1 {m.get('load1'):.2f} >= {LOAD1_TIMED_MAX} at acquisition")
    if m.get("power") != "AC":
        reasons.append(f"power {m.get('power')} (not AC)")
    if m.get("foreign_solvers"):
        reasons.append(f"foreign solvers at acquisition: {[f[0] for f in m['foreign_solvers']][:3]}")
    if m.get("timing_lock_present"):
        reasons.append(f"timing lock present at acquisition ({TIMING_LOCK})")
    if e.get("contention"):
        reasons.append(f"contended: {len(e['contention'])} contention events during the run")
    if reasons:
        return "untimed", "; ".join(reasons)
    return "bench", "machlock bench; load1 < 1.5, AC power, no timing lock, no foreign solvers, no contention (gpu_idle unverified)"


def _finish(run: Run, state: str, why: str, log=None) -> None:
    e = run.etiquette
    run.finished = _now()
    t_end = time.time()
    if run._t0:
        run.wall_total_s = round(t_end - run._t0, 2)
    m_acq = e.get("machine_at_acquisition") or {}
    if run._t_launch:
        # the solver's wall as the runner saw it: from lock acquisition (the shim's timestamp) or, without a
        # wrapper, from the launch; queue, timing-lock, guard and machlock waits are in wall_total_s/etiquette
        t_start = m_acq.get("t_acquired_epoch") if isinstance(m_acq.get("t_acquired_epoch"), (int, float)) else None
        if not t_start or t_start < run._t_launch:
            t_start = run._t_launch
        run.wall_runner_s = round(t_end - t_start, 2)
    else:
        run.wall_runner_s = None
    e["load1_at_end"] = round(exlib.load1(), 2)
    # parse whatever exists
    out_txt = run.run_dir / "out.txt"
    err_txt = run.run_dir / "err.txt"
    out_text = out_txt.read_text(errors="replace") if out_txt.exists() else ""
    err_text = err_txt.read_text(errors="replace") if err_txt.exists() else ""
    res = parse.parse_out(out_text, input_text=run.prepared.get("input_text"))
    res["rc"] = run.rc
    res["success"] = (run.rc == 0 and res["job_done"])
    res["wall_runner_s"] = run.wall_runner_s
    exp = run.prepared.get("expected_E_Ry")
    res["expected_E_Ry"] = exp
    res["expected_source"] = run.prepared.get("expected_source")
    res["dE_vs_expected_Ry"] = (res["total_energy_Ry"] - exp) if (exp is not None and res["total_energy_Ry"] is not None) else None
    run.result = res
    prof = run.run_dir / "ab_profile.json"
    run.files["profile"] = "ab_profile.json" if prof.exists() else None
    run.files["trace"] = sorted(p.name for p in run.run_dir.glob("ab_trace.*"))
    run.ab = parse.ab_record(run.request.mode, run.binary.to_dict(), run.run_dir, err_text,
                             prof if prof.exists() else None, run.plan.min_gpu_dim_effective)
    run.ab["suppressed_dyld_lines"] = run.suppressed_dyld_lines
    if run.request.mode != "reference" and run.plan.env_set.get("AB_PROFILE_JSON") and not prof.exists() and state == "done":
        run.ab["warnings"].append("AB_PROFILE_JSON requested but no ab_profile.json was written")
    grade, reason = _timing_grade(run, run._lock_eff, bool(run.plan.wrapper_argv))
    e["timing_grade"], e["timing_grade_reason"] = grade, reason
    if state == "done" and not res["success"]:
        state, why = "failed", f"rc={run.rc}, JOB DONE={res['job_done']}"
    if state == "done" and not res["converged"]:
        why = (why + "; " if why else "") + "convergence NOT achieved"
    run.set_state(state, why)
    _write_json(run.run_dir / "summary.json", {**run.summary(), "ab_totals": (run.ab or {}).get("totals"),
                                                 "gpu_verified_reason": (run.ab or {}).get("gpu_verified_reason"),
                                                 "lib_loaded": (run.ab or {}).get("lib_loaded"), "lib_build": (run.ab or {}).get("build"),
                                                 "build_matches_dylib": (run.ab or {}).get("build_matches_dylib"),
                                                 "machine_at_acquisition": e.get("machine_at_acquisition"),
                                                 "binary_path": run.binary.path, "binary_sha256": run.binary.sha256,
                                                 "dylib_path": run.binary.dylib_path, "dylib_sha256": run.binary.dylib_sha256,
                                                 "run_dir": str(run.run_dir)})
    run.save()
    # exlib-format line in the shared solver log: only for a run that started a process (exlib always writes an
    # integer rc and a subprocess wall), never for a fake binary into the real shared log
    m = e.get("machine_at_acquisition") or {}
    was_idle = not (m.get("foreign_solvers") if m else e.get("foreign_at_launch"))
    lock3 = {"share": "share", "guard": "guard"}.get(e.get("lock_effective") or "share", "machlock")
    started = run._t_launch is not None and run.rc is not None
    fake_into_shared = run.binary.id in binaries_mod._FAKES and Path(RUN_LOG).resolve() == Path(exlib.RUN_LOG).resolve()
    if started and not fake_into_shared:
        try:
            Path(RUN_LOG).parent.mkdir(parents=True, exist_ok=True)
            with open(RUN_LOG, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rc={run.rc} wall={(run.wall_runner_s or 0):.1f}s "
                         f"load1={exlib.load1():.2f} idle_at_start={was_idle} lock={lock3} "
                         f"label=abrun {run.request.preset} {run.request.mode} {run.binary.id} cwd={run.run_dir}\n")
        except OSError:
            pass
    if run.request.purge_tmp:
        shutil.rmtree(run.run_dir / "tmp", ignore_errors=True)
    # events
    if state == "done":
        run.publish("executed", {"run_id": run.id, "summary": run.summary()})
        status = "success"
    elif state == "cancelled":
        run.publish("execution_interrupted", {"run_id": run.id})
        status = "interrupted"
    else:
        run.publish("execution_error", {"run_id": run.id, "message": why, "tail": run.log.tail(TAIL_LINES)})
        status = "error"
    run.publish("muppet.done", {"run_id": run.id, "status": status, "seconds": run.wall_runner_s})
    if log:
        r, ab = run.result, run.ab or {}
        log(f"  [{state}] {run.id}: E_tot={r.get('total_energy_Ry')} Ry iters={r.get('scf_iterations')} "
            f"QE wall={r.get('wall_qe_s')} s (observation, grade={grade}) rc={run.rc}"
            + (f" gpu={ab.get('gpu_verified')} ({(ab.get('totals') or {}).get('gpu_calls')}/{(ab.get('totals') or {}).get('calls')}) "
               f"lib_ok={ab.get('lib_ok')}" if run.request.mode != "reference" else ""))


def _sample_tree(run: Run, table: dict | None = None) -> set[int]:
    """Refresh the run's descendant snapshot (persisted for restore()) and return launched pid + descendants."""
    root = run.proc.pid if run.proc is not None else run.process.get("pid")
    if root is None:
        return set()
    desc = descendants(root, table)
    run.process["descendants"] = sorted(set(desc) | set(run.process.get("descendants") or []))
    run.process["sampled"] = _now()
    return {root} | desc


def _launch_and_poll(run: Run, errors: list[str], log=None) -> str:
    """Steps 4-6 of execute(): fresh scratch, Popen in its own session, readers, the poll loop (machine.json,
    cancel, 20 s samples), reap.  Returns 'done' | 'failed' | 'cancelled' | 'relaunch' (shim exit 75: the
    timing lock appeared inside the machine lock, which the shim released without starting the solver)."""
    # 4. fresh scratch
    shutil.rmtree(run.run_dir / "tmp", ignore_errors=True)
    for f in ("out.txt", "err.txt", "machine.json", "ab_profile.json"):
        try:
            (run.run_dir / f).unlink()
        except OSError:
            pass
    # 5. launch in its own session
    env = launch.child_env(run.plan.env_set)
    env["ABRUN_TIMING_LOCK"] = str(TIMING_LOCK)          # the shim re-checks the same lock inside the machine lock
    try:
        run.proc = subprocess.Popen(run.plan.argv, cwd=str(run.run_dir), env=env, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        run.rc = None
        errors.append(f"launch failed: {exc}")
        return "failed"
    run._t_launch = time.time()
    run.pgid = run.proc.pid
    run._terminating = False
    run.process.update({"pid": run.proc.pid, "pgid": run.pgid, "descendants": [], "sampled": _now(),
                        "wrapper": bool(run.plan.wrapper_argv), "exited": False})
    run.publish("executing", {"run_id": run.id, "node": "pw"})
    run.set_state("waiting_guard", f"launched pgid {run.pgid}; waiting for machine.json (lock {run.etiquette['lock_effective']})")
    if log:
        log(f"  [launch] {run.id} pgid={run.pgid} lock={run.etiquette['lock_effective']} argv={' '.join(run.plan.argv)}")
    if run.request.timed:
        try:
            run.caffeinate = subprocess.Popen(["caffeinate", "-ims", "-w", str(run.proc.pid)],
                                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run.etiquette["caffeinate"] = True
        except OSError:
            run.etiquette["caffeinate"] = False
    t_out = threading.Thread(target=_reader, args=(run, run.proc.stdout, "out", run.run_dir / "out.txt", errors), daemon=True)
    t_err = threading.Thread(target=_reader, args=(run, run.proc.stderr, "err", run.run_dir / "err.txt", errors), daemon=True)
    t_out.start(); t_err.start()
    machine_seen = False
    last_sample = time.time()
    # 6. poll: machine.json -> running; a cancel that landed after the last check; samples every 20 s
    while run.proc.poll() is None:
        time.sleep(POLL_PROC_S)
        if run.cancel_event.is_set() and not run._terminating:
            if run.state != "cancelling":
                run.set_state("cancelling", "cancel requested")
            threading.Thread(target=_terminate, args=(run,), name=f"abrun-cancel-{run.id}", daemon=True).start()
        if not machine_seen and (run.run_dir / "machine.json").exists():
            try:
                run.etiquette["machine_at_acquisition"] = json.loads((run.run_dir / "machine.json").read_text())
                machine_seen = True
                run.etiquette["guard_waited_s"] = round(time.time() - run._t_launch, 1)
                _sample_tree(run)
                if not run.cancel_event.is_set():
                    run.set_state("running", f"lock acquired after {run.etiquette['guard_waited_s']} s")
                if log:
                    m = run.etiquette["machine_at_acquisition"]
                    log(f"  [running] load1={m.get('load1'):.2f} power={m.get('power')} foreign={len(m.get('foreign_solvers') or [])} "
                        f"nice={m.get('nice')}")
                last_sample = time.time()
            except (OSError, ValueError):
                pass
        elif machine_seen and time.time() - last_sample >= CONTENTION_POLL_S:
            last_sample = time.time()
            table = ps_table()
            own = _sample_tree(run, table)                      # our launcher and ranks are never "foreign"
            foreign = foreign_solvers(own, table=table)
            l1 = round(exlib.load1(), 2)
            run.etiquette["load1_samples"].append([_now(), l1])
            if foreign:
                run.etiquette["contention"].append({"t": _now(), "load1": l1, "foreign_solvers": foreign})
                run.guard_line(f"[guard] contention: load1={l1} foreign={foreign[0][1][:80]}")
            run.save()
    run.rc = run.proc.returncode
    # never orphan a rank: sweep the descendant snapshot (refreshed each sample) plus whatever is still attached,
    # BEFORE draining the pipes: a surviving rank holds their write ends and would stall the readers for 60 s
    snapshot = set(run.process.get("descendants") or []) | descendants(run.proc.pid) | set(run.process.get("descendants_at_cancel") or [])
    killed = _sweep(run, snapshot, run.proc.pid)
    if killed:
        run.guard_line(f"[guard] swept {len(killed)} process(es) that outlived the launcher: {killed}")
    run.process["exited"] = True
    t_out.join(timeout=30); t_err.join(timeout=30)
    if run.suppressed_profile_lines % PROFILE_SUPPRESS_EVERY:
        _marker(run)
    if not machine_seen and (run.run_dir / "machine.json").exists():
        try:
            run.etiquette["machine_at_acquisition"] = json.loads((run.run_dir / "machine.json").read_text())
            machine_seen = True
        except (OSError, ValueError):
            pass
    if run.caffeinate is not None:
        try:
            run.caffeinate.terminate()
            run.caffeinate.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        run.caffeinate = None
    if run.cancel_event.is_set():
        return "cancelled"
    if run.rc == SHIM_TIMING_LOCK_RC and not machine_seen:
        return "relaunch"
    return "done" if run.rc == 0 else "failed"


def execute(run: Run, log=None) -> dict:
    """Run one job to a terminal state (blocking) and return the run record dict."""
    run._profile_streamed = 0
    run._lock_eff = "share"
    run.started = _now()
    run._t0 = time.time()
    run._t_launch = None
    run.publish("execution_start", {"run_id": run.id})
    if run.cancel_event.is_set():
        _finish(run, "cancelled", "cancelled before start", log)
        return run.record()
    # 1. the timing lock, in every lock mode
    if not wait_for_timing_lock(run, log):
        _finish(run, "cancelled", "cancelled while waiting for the timing lock", log)
        return run.record()
    # 2. SHA256SUMS re-verified at launch
    bad = binaries_mod.verify_at_launch(run.binary)
    if bad:
        run.rc = None
        _finish(run, "failed", "launch refused: " + "; ".join(bad), log)
        return run.record()
    # 3. lock resolution at dequeue time
    bench_held = machlock_bench_held()
    lock_eff = launch.resolve_lock(run.request.lock, run.request.timed, bench_held)
    if lock_eff == "share" and bench_held:
        # PROTOCOL 9: timing windows are exclusive.  machlock only checks for foreign pw.x before a bench starts,
        # so a share run launched now would be invisible to the bench holder: wait as build unless overridden.
        if run.allow_share_during_bench:
            run.guard_line("[guard] lock=share while another session holds machlock bench: launching anyway "
                           "(allow_share_during_bench=true); recorded, not waited")
        else:
            lock_eff = "build"
            run.etiquette["lock_share_promoted_to_build"] = True
            run.guard_line("[guard] lock=share requested but another session holds machlock bench -> waiting as "
                           "machlock build (set allow_share_during_bench=true to launch into the bench window)")
    run.plan = launch.plan(run.request, run.binary, run.run_dir, lock_effective=lock_eff, bench_held=bench_held)
    wrapper_present = bool(run.plan.wrapper_argv)
    run._lock_eff = lock_eff
    run.etiquette["lock_effective"] = launch.lock_effective_label(lock_eff, wrapper_present)
    run.etiquette["bench_held_at_dequeue"] = bench_held
    if lock_eff in ("bench", "build") and not wrapper_present:
        run.set_state("waiting_guard", "guard mode (no machlock): waiting for foreign solvers")
        if not wait_for_idle_guard(run, log):
            _finish(run, "cancelled", "cancelled while waiting in guard mode", log)
            return run.record()
    run.etiquette["foreign_at_launch"] = foreign_solvers()
    if run.cancel_event.is_set():                 # a cancel that arrived between the first check and the launch
        _finish(run, "cancelled", "cancelled before launch", log)
        return run.record()
    # 4-6. launch and poll; relaunch when the shim found the timing lock inside the machine lock (it exits 75
    #      without starting the solver, releasing the lock so the lock holder is never deadlocked on us)
    errors: list[str] = []
    outcome = "failed"
    for attempt in range(MAX_TIMING_LOCK_RELAUNCHES + 1):
        outcome = _launch_and_poll(run, errors, log)
        if outcome != "relaunch":
            break
        run.etiquette["timing_lock_relaunches"] += 1
        run.guard_line(f"[guard] shim found {TIMING_LOCK} inside the machine lock -> released it without launching; "
                       f"waiting for the lock again (relaunch {run.etiquette['timing_lock_relaunches']})")
        if log:
            log(f"  [guard] timing lock present inside the machine lock -> relaunch {run.etiquette['timing_lock_relaunches']}")
        run.rc = None
        run.process = {"pid": None, "pgid": None, "descendants": [], "sampled": None, "wrapper": False, "exited": False}
        time.sleep(POLL_PROC_S)
        if not wait_for_timing_lock(run, log):
            _finish(run, "cancelled", "cancelled while waiting for the timing lock", log)
            return run.record()
        if run.cancel_event.is_set():
            _finish(run, "cancelled", "cancelled before relaunch", log)
            return run.record()
    if outcome == "relaunch":
        _finish(run, "failed", f"timing lock kept appearing inside the machine lock ({MAX_TIMING_LOCK_RELAUNCHES} relaunches)", log)
    elif outcome == "cancelled":
        _finish(run, "cancelled", f"cancelled (rc={run.rc})", log)
    elif outcome == "done":
        _finish(run, "done", "", log)
    elif run.proc is None:
        _finish(run, "failed", errors[0] if errors else "launch failed", log)
    else:
        _finish(run, "failed", f"rc={run.rc}" + (f"; {errors[0]}" if errors else ""), log)
    return run.record()


def run_sync(request: launch.Request | dict, run_dir: Path | str | None = None, log=print,
             binary: binaries_mod.Binary | None = None, allow_share_during_bench: bool | None = None) -> dict:
    """Blocking entry point for example drivers and the CLI: new_run + execute, returns the run record.
    Raises PlanError (with .violations) when the request cannot be launched."""
    run = new_run(request, run_dir, binary, allow_share_during_bench=allow_share_during_bench)
    REGISTRY.add(run)
    if log:
        log(f"  [abrun] {run.id}: preset {run.request.preset} mode {run.request.mode} binary {run.binary.id} -> {run.run_dir}")
    return execute(run, log)


# --------------------------------------------------------------------------- registry, persistence, pins

class Registry:
    """In-memory map of live Run objects plus loaders for run.json records on disk."""

    def __init__(self):
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def add(self, run: Run) -> None:
        with self._lock:
            self._runs[run.id] = run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def live(self) -> list[Run]:
        with self._lock:
            return list(self._runs.values())

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def records(self, runs_dir: Path | None = None) -> list[dict]:
        """Every run record: live objects first (fresh), then run.json files under runs_dir not loaded."""
        out, seen = [], set()
        for r in self.live():
            out.append(r.record()); seen.add(r.id)
        for rec in load_records(runs_dir):
            if rec["id"] not in seen:
                out.append(rec)
        out.sort(key=lambda r: r.get("created") or "", reverse=True)
        return out

    def record(self, run_id: str, runs_dir: Path | None = None) -> dict | None:
        r = self.get(run_id)
        if r is not None:
            return r.record()
        p = Path(runs_dir or RUNS_DIR) / run_id / "run.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                return None
        return None


REGISTRY = Registry()


def load_records(runs_dir: Path | None = None) -> list[dict]:
    d = Path(runs_dir or RUNS_DIR)
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*/run.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def load_record(run_id_or_dir: str | Path, runs_dir: Path | None = None) -> dict | None:
    p = Path(run_id_or_dir)
    if p.is_dir():
        p = p / "run.json"
    elif not p.exists():
        p = Path(runs_dir or RUNS_DIR) / str(run_id_or_dir) / "run.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run_from_record(rec: dict) -> Run:
    """Rebuild a Run object (for re-enqueueing a persisted queued job after a restart)."""
    req = launch.Request.from_dict(rec["request"])
    binary = binaries_mod.Binary.from_dict(rec["binary"])
    plan = launch.Plan(**{k: v for k, v in rec["plan"].items() if k in launch.Plan.__dataclass_fields__})
    prepared = {"input_sha256": rec.get("input_sha256"), "source_sha256": rec.get("source_sha256"),
                "source": rec.get("input_source"), "pseudos": rec.get("pseudos"), "meta": rec.get("input_meta") or {},
                "expected_E_Ry": (rec.get("result") or {}).get("expected_E_Ry"),
                "expected_source": (rec.get("result") or {}).get("expected_source")}
    pr = presets.PRESETS.get(req.preset)
    if pr:
        prepared["expected_E_Ry"], prepared["expected_source"] = pr["expected_E_Ry"], pr["expected_source"]
    try:
        prepared["input_text"] = (Path(rec["run_dir"]) / "input.in").read_text()
    except OSError:
        prepared["input_text"] = None
    run = Run(rec["id"], req, Path(rec["run_dir"]), binary, prepared, plan)
    run.created = rec.get("created") or run.created
    run.state, run.state_why = rec.get("state", "queued"), rec.get("state_why", "")
    run.started, run.finished, run.rc = rec.get("started"), rec.get("finished"), rec.get("rc")
    run.etiquette = rec.get("etiquette") or run.etiquette
    run.result, run.ab = rec.get("result"), rec.get("ab")
    run.pinned_reference = bool(rec.get("pinned_reference"))
    run.wall_runner_s = rec.get("wall_runner_s")
    run.wall_total_s = rec.get("wall_total_s")
    run.allow_share_during_bench = bool(rec.get("allow_share_during_bench"))
    if isinstance(rec.get("process"), dict):
        run.process = {**run.process, **rec["process"]}
    return run


_PINS_LOCK = threading.Lock()


class Pins:
    """pins.json: input sha256 -> the run id that is the reference for that input (read-modify-write under a lock)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or (RUNS_DIR / "pins.json"))

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def pin(self, input_sha256: str, run_id: str) -> dict:
        with _PINS_LOCK:
            d = self.load()
            d[input_sha256] = run_id
            _write_json(self.path, d)
            return d

    def unpin(self, run_id: str) -> bool:
        with _PINS_LOCK:
            d = self.load()
            changed = False
            for k in [k for k, v in d.items() if v == run_id]:
                del d[k]; changed = True
            if changed:
                _write_json(self.path, d)
            return changed


def _is_run_subdir(d: Path, runs_dir: Path) -> bool:
    """True only for a direct child of the runs root (never the root, its parents, or anything reached by '..')."""
    try:
        return d.resolve().parent == Path(runs_dir).resolve() and d.resolve() != Path(runs_dir).resolve()
    except OSError:
        return False


def delete_run(run_id: str, runs_dir: Path | None = None) -> bool:
    """Remove a terminal run's directory (and its pin).  Live non-terminal runs are refused, and so is any id
    that does not name a direct subdirectory of the runs root ('.', '..', a path)."""
    if not run_id or run_id in (".", "..") or "/" in run_id or os.sep in run_id:
        return False
    root = Path(runs_dir or RUNS_DIR)
    r = REGISTRY.get(run_id)
    if r is not None and not r.terminal:
        return False
    d = root / run_id
    if r is not None:
        d = r.run_dir
    if not _is_run_subdir(d, root):
        return False
    if r is not None:
        REGISTRY.remove(run_id)
    if d.is_dir() and (d / "run.json").exists():
        shutil.rmtree(d, ignore_errors=True)
        Pins().unpin(run_id)
        return True
    return False


# --------------------------------------------------------------------------- the worker

class Worker:
    """Exactly one daemon thread drains a FIFO of Run objects, so abrun never runs two solvers at once.
    The queue is persisted to runs/queue.json after every change; restore() re-enqueues queued ids and
    marks a job that was running at the previous shutdown failed with reason 'server restarted', unless its
    solver tree (run.json 'process') is still alive: then the worker holds (paused with a reason, nothing
    dequeued) until that tree has exited, and the record is finished as failed or, after a cancel, cancelled."""

    HOLD_REASON = "orphaned solver"

    def __init__(self, registry: Registry | None = None, runs_dir: Path | None = None, log=None):
        self.registry = registry or REGISTRY
        self.runs_dir = Path(runs_dir or RUNS_DIR)
        self.queue: deque[Run] = deque()
        self.paused = False
        self.pause_reason: str | None = None
        self.current: Run | None = None
        self.hold: Run | None = None
        self.log = log
        self._cv = threading.Condition()
        self._persist_lock = threading.Lock()
        self._stop = threading.Event()
        self._hold_checked = 0.0
        self._thread = threading.Thread(target=self._loop, name="abrun-worker", daemon=True)

    @property
    def queue_path(self) -> Path:
        return self.runs_dir / "queue.json"

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, cancel_running: bool = True) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if cancel_running and self.current is not None and not self.current.terminal:
            cancel(self.current)
        self.persist()

    def persist(self) -> None:
        """Snapshot and write under one lock: concurrent callers (worker, handlers, stop) serialise and the
        newest snapshot always lands last.  A hold keeps its id in 'running'; its pause is not the user's."""
        with self._persist_lock:
            with self._cv:
                running = self.current.id if (self.current and not self.current.terminal) else None
                if running is None and self.hold is not None and not self.hold.terminal:
                    running = self.hold.id
                data = {"running": running, "queued": [r.id for r in self.queue],
                        "paused": bool(self.paused and self.pause_reason is None), "saved": _now()}
            try:
                _write_json(self.queue_path, data)
            except OSError:
                pass

    def restore(self) -> dict:
        """Re-enqueue persisted queued runs; fail the one that was running at shutdown, or hold while its
        solver tree is still alive (SIGHUP/kill -9 of the server leaves pw.x running in its own session)."""
        try:
            data = json.loads(self.queue_path.read_text())
        except (OSError, ValueError):
            return {"requeued": [], "failed": None, "held": None}
        failed = held = None
        if data.get("running"):
            rec = load_record(data["running"], self.runs_dir)
            if rec and rec.get("state") not in TERMINAL:
                run = run_from_record(rec)
                alive = live_pids(run.process, run.run_dir)
                if alive:
                    run.state_why = f"server restarted; solver processes still alive: {alive}"
                    run.save()
                    self.registry.add(run)
                    with self._cv:
                        self.hold = run
                        self.paused = True
                        self.pause_reason = (f"{self.HOLD_REASON}: run {run.id} from before the restart still has live "
                                             f"solver processes {alive}; the queue resumes when they have exited (or cancel {run.id})")
                    held = run.id
                    if self.log:
                        self.log(f"  [restore] holding the queue: {self.pause_reason}")
                else:
                    run.rc = rec.get("rc")
                    run.finished = _now()
                    run.state, run.state_why = "failed", "server restarted"
                    run.etiquette["timing_grade"], run.etiquette["timing_grade_reason"] = "untimed", "server restarted"
                    run.process["exited"] = True
                    run.save()
                    self.registry.add(run)
                    failed = run.id
        requeued = []
        for rid in data.get("queued", []):
            rec = load_record(rid, self.runs_dir)
            if rec and rec.get("state") == "queued":
                run = run_from_record(rec)
                self.registry.add(run)
                self.submit(run)
                requeued.append(rid)
        if data.get("paused"):
            self.paused, self.pause_reason = True, None
        self.persist()
        return {"requeued": requeued, "failed": failed, "held": held}

    def _check_hold(self) -> None:
        """Every HOLD_POLL_S: is the orphaned solver tree still alive?  When it is gone, finish its record and
        lift the hold's pause (never a pause the user set in the meantime)."""
        run = self.hold
        if run is None or time.time() - self._hold_checked < HOLD_POLL_S:
            return
        self._hold_checked = time.time()
        alive = live_pids(run.process, run.run_dir)
        if alive:
            return
        run.process["exited"] = True
        run.finished = _now()
        run.etiquette["timing_grade"], run.etiquette["timing_grade_reason"] = "untimed", "server restarted"
        if run.cancel_event.is_set():
            run.set_state("cancelled", "cancelled after a server restart (the orphaned solver tree was stopped)")
            run.publish("execution_interrupted", {"run_id": run.id})
            run.publish("muppet.done", {"run_id": run.id, "status": "interrupted", "seconds": run.wall_runner_s})
        else:
            run.set_state("failed", "server restarted; the orphaned solver tree has exited")
            run.publish("execution_error", {"run_id": run.id, "message": run.state_why, "tail": []})
            run.publish("muppet.done", {"run_id": run.id, "status": "error", "seconds": run.wall_runner_s})
        with self._cv:
            self.hold = None
            if self.pause_reason is not None and self.pause_reason.startswith(self.HOLD_REASON):
                self.paused, self.pause_reason = False, None
            self._cv.notify_all()
        if self.log:
            self.log(f"  [restore] orphaned run {run.id} is {run.state}; queue released")
        self.persist()
        global_bus.publish("status", {"queue_remaining": len(self.queue)})

    def submit(self, run: Run) -> int:
        with self._cv:
            self.queue.append(run)
            run.position = len(self.queue) - 1
            self._cv.notify_all()
            pos = run.position
        self.registry.add(run)
        self.persist()
        global_bus.publish("status", {"queue_remaining": len(self.queue)})
        return pos

    def pause(self) -> bool:
        with self._cv:
            self.paused, self.pause_reason = True, None
        self.persist()
        return True

    def resume(self) -> bool:
        with self._cv:
            if self.hold is not None:                    # the hold outlives a resume; the reason stays visible
                self.pause_reason = self.pause_reason or f"{self.HOLD_REASON}: run {self.hold.id} still alive"
                return True
            self.paused, self.pause_reason = False, None
            self._cv.notify_all()
        self.persist()
        return False

    def reorder(self, ids: list[str]) -> list[str]:
        with self._cv:
            by_id = {r.id: r for r in self.queue}
            new = [by_id[i] for i in ids if i in by_id] + [r for r in self.queue if r.id not in ids]
            self.queue = deque(new)
            for i, r in enumerate(self.queue):
                r.position = i
            out = [r.id for r in self.queue]
        self.persist()
        return out

    def dequeue_cancel(self, run_id: str) -> bool:
        """Cancel a still-queued run (removed from the queue, marked cancelled)."""
        found = None
        with self._cv:
            for r in list(self.queue):
                if r.id == run_id:
                    self.queue.remove(r)
                    found = r
                    break
        if found is None:
            return False
        found.cancel_event.set()
        found.finished = _now()
        found.set_state("cancelled", "cancelled while queued")
        found.publish("execution_interrupted", {"run_id": found.id})
        found.publish("muppet.done", {"run_id": found.id, "status": "interrupted", "seconds": 0})
        self.persist()                              # outside _cv: persist() takes _persist_lock, then _cv
        return True

    def status(self) -> dict:
        with self._cv:
            busy = self.current is not None and not self.current.terminal
            held = self.hold.id if (self.hold is not None and not self.hold.terminal) else None
            return {"running": self.current.id if busy else None,
                    "queued": [r.id for r in self.queue], "paused": self.paused, "paused_reason": self.pause_reason,
                    "held": held, "state": "busy" if busy else ("held" if held else "idle")}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._check_hold()
            with self._cv:
                while not self._stop.is_set() and (self.paused or not self.queue or self.hold is not None):
                    self._cv.wait(1.0)
                    if self.hold is not None:
                        break                           # re-check the orphan outside the lock
                if self._stop.is_set():
                    return
                if self.paused or not self.queue or self.hold is not None:
                    continue
                run = self.queue.popleft()
                for i, r in enumerate(self.queue):
                    r.position = i
                self.current = run
            self.persist()
            try:
                execute(run, self.log)
            except Exception as exc:  # noqa: BLE001  (a worker must survive any single job)
                try:
                    if run.proc is not None and run.proc.poll() is None:     # never leave a solver behind
                        run.cancel_event.set()
                        _terminate(run)
                    run.rc = run.rc if run.rc is not None else -1
                    _finish(run, "failed", f"runner error: {exc!r}", self.log)
                except Exception:  # noqa: BLE001
                    run.state, run.state_why = "failed", f"runner error: {exc!r}"
            finally:
                with self._cv:
                    self.current = None
                self.persist()
                global_bus.publish("status", {"queue_remaining": len(self.queue)})
