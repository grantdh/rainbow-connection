"""abrun HTTP server: ThreadingHTTPServer on 127.0.0.1:8766 (ABRUN_PORT / --port), a regex route table,
JSON errors {error, violations[]}, the static single-page UI, Server-Sent Events with Last-Event-ID /
?since= replay and ': ping' heartbeats, a 5 s machine-status thread and a graceful SIGINT.

One process, four thread kinds: HTTP handler threads (daemon), the single Worker thread that drains the
persisted queue (runner.Worker), the machine-status thread, and one SSE writer per open EventSource.
No auth, and binding to the loopback interface is not origin isolation: any web page the user opens can
send a preflight-free POST to 127.0.0.1.  So every request must carry Host 127.0.0.1:<port> or
localhost:<port> (403 otherwise: a DNS-rebound hostname fails here), POST/DELETE must be
application/json with Origin absent or same-origin (403/415), OPTIONS is 403 with no CORS headers, and
bodies are capped at 1 MiB (413).  The browser UI and the curl cookbook satisfy all of that.

    bin/abrun serve [--port 8766] [--open] [--fake-pw]

In-process (tests): srv = Server(port=0, fake_pw=True); srv.start(); ... srv.stop().  --fake-pw registers
tests/fake_pw.py as binary ids 'fake' (apple-bottom, dynamic) and 'reference-fake', points launch.MACHLOCK
at /nonexistent so fake runs never take the real machlock, and redirects the runs directory and the shared
solver-runs.log to a scratch directory (unless ABRUN_RUNS_DIR or an in-process monkeypatch already moved
them) so fake runs never land beside real ones; the real ~/Dev/.timing-lock is still honoured (nothing
bypasses it; tests monkeypatch abrun.runner.TIMING_LOCK in-process).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import DEFAULT_FROZEN, EXLIB_DIR, INTERPRETER, PKG_DIR, VERSION, exlib
from . import binaries as binaries_mod
from . import compare as compare_mod
from . import launch, parse, presets, runner
from .events import TERMINAL_EVENT, sse_frame, sse_stream

HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("ABRUN_PORT", "8766") or 8766)
STATIC_DIR = PKG_DIR / "static"
MACHINE_POLL_S = 5.0
HEARTBEAT_S = 10.0                      # module attribute: tests shorten it
LOG_MAX_DEFAULT = 2000
DISK_WARN_GB = 20.0
BODY_MAX_BYTES = 1 << 20                # 413 above this
HANDLER_TIMEOUT_S = 60.0                # idle socket timeout per handler thread (SSE writes keep flowing: pings)
REAL_RUNS_DIR = PKG_DIR / "runs"        # the default that --fake-pw must never write into
FILE_WHITELIST = ("input.in", "out.txt", "err.txt", "ab_profile.json", "machine.json", "summary.json", "run.json")
RE_TRACE_FILE = re.compile(r"^ab_trace\.\d+$")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
                ".json": "application/json", ".txt": "text/plain; charset=utf-8"}
IDENTITY_KEYS = ("preset", "input_path", "binary_id", "mode", "ranks", "nk", "omp", "min_gpu_dim", "profile_json",
                 "report", "trace", "verify_dylib", "lock", "timed", "allow_omp_gt1", "allow_dev", "force_profile_json")
DEFAULT_LEGS = ("reference", "ab_cpu", "ab_gpu_all")


class ApiError(Exception):
    def __init__(self, status: int, message: str, violations: list[str] | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.violations, self.extra = status, message, list(violations or []), extra or {}


# --------------------------------------------------------------------------- record helpers

def summary_from_record(rec: dict) -> dict:
    """RunSummary (the /api/runs row) from a run.json dict (disk-only runs; live runs use Run.summary())."""
    r, ab, e = rec.get("result") or {}, rec.get("ab") or {}, rec.get("etiquette") or {}
    req, b, totals = rec.get("request") or {}, rec.get("binary") or {}, ab.get("totals") or {}
    return {
        "id": rec.get("id"), "state": rec.get("state"), "state_why": rec.get("state_why"), "created": rec.get("created"),
        "started": rec.get("started"), "finished": rec.get("finished"), "preset": rec.get("preset"),
        "input_sha256": rec.get("input_sha256"), "binary_id": b.get("id"), "binary_short": compare_mod.binary_short(rec),
        "mode": req.get("mode"), "ranks": req.get("ranks"), "nk": req.get("nk"), "omp": (rec.get("plan") or {}).get("omp_effective"),
        "lock_effective": e.get("lock_effective"), "timing_grade": e.get("timing_grade"), "timing_grade_reason": e.get("timing_grade_reason"),
        "total_energy_Ry": r.get("total_energy_Ry"), "scf_iterations": r.get("scf_iterations"), "converged": r.get("converged"),
        "rc": rec.get("rc"), "wall_qe_s": r.get("wall_qe_s"), "wall_runner_s": rec.get("wall_runner_s"),
        "wall_total_s": rec.get("wall_total_s"),
        "gemm_calls": totals.get("calls"), "gpu_calls": totals.get("gpu_calls"), "fallbacks": totals.get("fallbacks"),
        "gpu_verified": ab.get("gpu_verified"), "gpu_badge": ab.get("gpu_badge"), "evidence_source": ab.get("evidence_source"),
        "lib_ok": ab.get("lib_ok"), "pinned_reference": bool(rec.get("pinned_reference")), "derived_from": rec.get("derived_from"),
        "label": rec.get("label") or req.get("label") or "", "expected_E_Ry": r.get("expected_E_Ry"),
        "dE_vs_expected_Ry": r.get("dE_vs_expected_Ry"), "progress": rec.get("progress") or {},
    }


def synth_log_lines(run_dir: Path) -> tuple[list, int]:
    """(lines [[seq, stream, text]...], suppressed AB_PROFILE lines) rebuilt from out.txt/err.txt for a run that is
    no longer in memory (after a server restart), filtered like the live reader (dyld noise, per-call lines)."""
    lines, seq, suppressed = [], 0, 0
    for name, stream in (("out.txt", "out"), ("err.txt", "err")):
        p = run_dir / name
        if not p.exists():
            continue
        streamed = 0
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if stream == "err":
                if parse.RE_PER_CALL.match(line):
                    streamed += 1
                    if streamed > runner.PROFILE_STREAM_FIRST:
                        suppressed += 1
                        continue
                elif line.startswith("dyld[") and "libapplebottom" not in line:
                    continue
            seq += 1
            lines.append([seq, stream, line])
    if suppressed:
        seq += 1
        lines.append([seq, "err", f"+{suppressed} AB_PROFILE per-call lines suppressed, see err.txt"])
    return lines, suppressed


def identity_of(req: launch.Request) -> tuple:
    d = req.to_dict()
    return tuple((k, d.get(k)) for k in IDENTITY_KEYS)


def power_state() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if "AC Power" in out:
        return "AC"
    if "Battery Power" in out:
        return "Battery"
    return "unknown"


# --------------------------------------------------------------------------- the server object

class Server:
    """Owns the HTTP server, the worker, the machine-status thread and the fake registration."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = HOST, fake_pw: bool = False, log=None, verbose: bool = False):
        self.host, self.port_requested, self.fake_pw, self.log, self.verbose = host, port, fake_pw, log, verbose
        self.stop_event = threading.Event()
        self.reference_binary_id = "ref"
        self.machine: dict = {}
        self.httpd: ThreadingHTTPServer | None = None
        self.started: str | None = None
        self._saved_machlock = None
        self._saved_paths: tuple | None = None
        self.fake_dir: Path | None = None
        self._threads: list[threading.Thread] = []
        self._fake_ids: list[str] = []
        if fake_pw:
            self._register_fakes()
        self.runs_dir = Path(runner.RUNS_DIR)
        self.worker = runner.Worker(runs_dir=self.runs_dir, log=self._worker_log)

    # ---- lifecycle
    def _register_fakes(self) -> None:
        from .tests.fake_pw import fake_binary
        for kind in ("ab", "reference"):
            b = fake_binary(kind)
            try:
                os.chmod(b.path, 0o755)
            except OSError:
                pass
            binaries_mod.register_fake(b)
            self._fake_ids.append(b.id)
        self.reference_binary_id = "reference-fake"
        self._saved_machlock = launch.MACHLOCK
        launch.MACHLOCK = Path("/nonexistent/machlock.py")      # fake runs never take the real machlock
        # fake runs never land in the real runs directory or the shared solver-runs.log (other sessions read it):
        # both move to a scratch directory unless already redirected (ABRUN_RUNS_DIR, or a test's monkeypatch)
        runs_real = Path(runner.RUNS_DIR).resolve() == REAL_RUNS_DIR.resolve()
        log_real = Path(runner.RUN_LOG).resolve() == Path(exlib.RUN_LOG).resolve()
        if runs_real or log_real:
            self.fake_dir = Path(tempfile.gettempdir()) / f"abrun-fake-{os.getuid()}"
            self.fake_dir.mkdir(parents=True, exist_ok=True)
            self._saved_paths = (runner.RUNS_DIR, runner.RUN_LOG)
            if runs_real:
                runner.RUNS_DIR = self.fake_dir / "runs"
            if log_real:
                runner.RUN_LOG = self.fake_dir / "solver-runs.log"

    def _worker_log(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    def start(self) -> "Server":
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        restored = self.worker.restore()
        if self.log and (restored["requeued"] or restored["failed"]):
            self.log(f"  [restore] requeued {restored['requeued']}; interrupted run failed 'server restarted': {restored['failed']}")
        self.machine = self._probe_machine()
        self.worker.start()
        self.httpd = ThreadingHTTPServer((self.host, self.port_requested), Handler)
        self.httpd.app = self                       # type: ignore[attr-defined]
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")
        t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="abrun-http", daemon=True)
        m = threading.Thread(target=self._machine_loop, name="abrun-machine", daemon=True)
        self._threads = [t, m]
        t.start(); m.start()
        return self

    def stop(self, cancel_running: bool = True, wait_s: float = 30.0) -> None:
        """Graceful: cancel the running job (and wait for it to reach a terminal state), persist the queue,
        close every SSE stream and the listening socket."""
        self.stop_event.set()
        self.worker.stop(cancel_running=cancel_running)
        if cancel_running:
            t0 = time.time()
            while time.time() - t0 < wait_s:
                cur = self.worker.current
                if cur is None or cur.terminal:
                    break
                time.sleep(0.1)
            self.worker.persist()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self._saved_machlock is not None:
            launch.MACHLOCK = self._saved_machlock
            self._saved_machlock = None
        if self._saved_paths is not None:
            runner.RUNS_DIR, runner.RUN_LOG = self._saved_paths
            self._saved_paths = None
        for fid in self._fake_ids:                  # the fakes exist only while a fake server runs
            binaries_mod._FAKES.pop(fid, None)
        self._fake_ids = []

    @property
    def port(self) -> int:
        return self.httpd.server_address[1] if self.httpd else self.port_requested

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ---- machine status
    def _machine_loop(self) -> None:
        while not self.stop_event.wait(MACHINE_POLL_S):
            try:
                self.machine = self._probe_machine()
            except Exception as exc:  # noqa: BLE001  (status must never kill the thread)
                self.machine = {**self.machine, "error": repr(exc), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def _probe_machine(self) -> dict:
        load1 = round(os.getloadavg()[0], 2)
        tl = Path(runner.TIMING_LOCK)
        tl_present, tl_text = tl.exists(), ""
        if tl_present:
            try:
                tl_text = tl.read_text(errors="ignore").strip()[:200]
            except OSError:
                pass
        machlock = Path(launch.MACHLOCK)
        bench_held = runner.machlock_bench_held()
        log_tail: list[str] = []
        mlog = machlock.parent / "machine.log"
        if machlock.exists() and mlog.exists():
            try:
                with open(mlog, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - 4096))
                    log_tail = fh.read().decode(errors="replace").splitlines()[-3:]
            except OSError:
                pass
        table = runner.ps_table()
        own: set[int] = set()
        for r in runner.REGISTRY.live():            # our launcher and its ranks (own pgids) are never "foreign"
            root = r.proc.pid if (r.proc is not None and r.proc.poll() is None) else None
            if root is None and not r.terminal and r.process.get("pid") and not r.process.get("exited"):
                root = r.process["pid"]             # an orphan adopted after a restart
            if root:
                own |= {root} | runner.descendants(root, table) | set(r.process.get("descendants") or [])
        foreign = [{"pid": pid, "cmd": cmd} for pid, cmd in runner.foreign_solvers(own, table=table)]
        power = power_state()
        disk_free_gb = None
        d = self.runs_dir
        while d and not d.exists():
            d = d.parent
        try:
            disk_free_gb = round(shutil.disk_usage(d).free / 1e9, 1)
        except OSError:
            pass
        blockers = []
        if load1 >= runner.LOAD1_TIMED_MAX:
            blockers.append(f"load1 {load1:.2f} >= {runner.LOAD1_TIMED_MAX}")
        if tl_present:
            blockers.append(f"timing lock present: {tl}")
        if bench_held:
            blockers.append("machlock bench held by another session")
        for f in foreign[:3]:
            blockers.append(f"foreign solver {f['pid']}: {f['cmd'][:60]}")
        if power != "AC":
            blockers.append(f"power {power} (not AC)")
        return {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "load1": load1,
            "timing_lock": {"path": str(tl), "present": tl_present, "text": tl_text},
            "machlock": {"path": str(machlock.parent / "machine.lock"), "available": machlock.exists() and os.environ.get("MUPPET_NO_MACHLOCK") != "1",
                         "bench_held": bench_held, "log_tail": log_tail},
            "foreign_solvers": foreign, "power": power, "disk_free_gb": disk_free_gb, "disk_warn": (disk_free_gb is not None and disk_free_gb < DISK_WARN_GB),
            "timed_possible": not blockers, "timed_blockers": blockers,
            "worker": self._worker_status(),
            "fake_pw": self.fake_pw,
        }

    def _worker_status(self) -> dict:
        st = self.worker.status()
        return {"state": st["state"], "current": st["running"], "queued": len(st["queued"]), "paused": st["paused"]}

    # ---- run helpers
    def record(self, rid: str) -> dict:
        rec = runner.REGISTRY.record(rid, self.runs_dir)
        if rec is None:
            raise ApiError(404, f"unknown run id {rid!r}")
        return rec

    def run_dir_of(self, rid: str, rec: dict | None = None) -> Path:
        live = runner.REGISTRY.get(rid)
        if live is not None:
            return live.run_dir
        rec = rec or self.record(rid)
        return Path(rec.get("run_dir") or (self.runs_dir / rid))

    def duplicate_of(self, req: launch.Request) -> str | None:
        key = identity_of(req)
        st = self.worker.status()
        for rid in ([st["running"]] if st["running"] else []) + list(st["queued"]):
            r = runner.REGISTRY.get(rid)
            if r is not None and not r.terminal and identity_of(r.request) == key:
                return rid
        return None

    def submit(self, req: launch.Request, allow_share_during_bench: bool = False) -> runner.Run:
        dup = self.duplicate_of(req)
        if dup:
            raise ApiError(409, f"an identical plan is already queued or running: {dup}", extra={"id": dup})
        try:
            run = runner.new_run(req, allow_share_during_bench=allow_share_during_bench)
        except runner.PlanError as exc:
            raise ApiError(400, "invalid plan", exc.violations) from exc
        self.worker.submit(run)
        if self.log:
            self.log(f"  [queued] {run.id} (position {run.position}): preset {req.preset} mode {req.mode} binary {req.binary_id}")
        return run

    def summaries(self) -> list[dict]:
        pins = runner.Pins().load()
        pinned_ids = set(pins.values())
        st = self.worker.status()
        positions = {rid: i for i, rid in enumerate(st["queued"])}
        out = []
        for rec in runner.REGISTRY.records(self.runs_dir):
            live = runner.REGISTRY.get(rec["id"])
            s = live.summary() if live is not None else summary_from_record(rec)
            s["pinned_reference"] = s["id"] in pinned_ids
            s["position"] = positions.get(s["id"])
            s["run_dir"] = str(live.run_dir) if live is not None else rec.get("run_dir")
            out.append(s)
        return out


# --------------------------------------------------------------------------- routes

def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern)


RID = r"(?P<rid>(?!\.+(?:/|$))[A-Za-z0-9_.-]+)"   # never '.' or '..': they would name the runs root or its parent
ROUTES = [
    ("GET", _rx(r"^/$"), "index"),
    ("GET", _rx(r"^/static/(?P<name>[A-Za-z0-9_.-]+)$"), "static"),
    ("GET", _rx(r"^/api/health$"), "health"),
    ("GET", _rx(r"^/api/machine$"), "machine"),
    ("GET", _rx(r"^/api/binaries$"), "binaries"),
    ("GET", _rx(r"^/api/presets$"), "presets"),
    ("GET", _rx(r"^/api/presets/(?P<pid>[A-Za-z0-9_-]+)/input$"), "preset_input"),
    ("GET", _rx(r"^/api/modes$"), "modes"),
    ("POST", _rx(r"^/api/plan$"), "plan"),
    ("POST", _rx(r"^/api/runs$"), "runs_post"),
    ("GET", _rx(r"^/api/runs$"), "runs_get"),
    ("POST", _rx(r"^/api/ab_set$"), "ab_set"),
    ("GET", _rx(r"^/api/events$"), "events_global"),
    ("GET", _rx(rf"^/api/runs/{RID}$"), "run_get"),
    ("DELETE", _rx(rf"^/api/runs/{RID}$"), "run_delete"),
    ("GET", _rx(rf"^/api/runs/{RID}/log$"), "run_log"),
    ("GET", _rx(rf"^/api/runs/{RID}/events$"), "run_events"),
    ("GET", _rx(rf"^/api/runs/{RID}/files/(?P<name>[A-Za-z0-9_.-]+)$"), "run_file"),
    ("POST", _rx(rf"^/api/runs/{RID}/cancel$"), "run_cancel"),
    ("POST", _rx(rf"^/api/runs/{RID}/rerun$"), "run_rerun"),
    ("POST", _rx(rf"^/api/runs/{RID}/pin$"), "run_pin"),
    ("DELETE", _rx(rf"^/api/runs/{RID}/pin$"), "run_unpin"),
    ("GET", _rx(r"^/api/queue$"), "queue_get"),
    ("POST", _rx(r"^/api/queue/pause$"), "queue_pause"),
    ("POST", _rx(r"^/api/queue/resume$"), "queue_resume"),
    ("POST", _rx(r"^/api/queue/reorder$"), "queue_reorder"),
    ("GET", _rx(r"^/api/compare$"), "compare"),
    ("GET", _rx(r"^/api/export$"), "export"),
]


class Handler(BaseHTTPRequestHandler):
    server_version = f"abrun/{VERSION}"
    protocol_version = "HTTP/1.0"
    timeout = HANDLER_TIMEOUT_S

    @property
    def app(self) -> Server:
        return self.server.app  # type: ignore[attr-defined]

    # ---- plumbing
    def log_message(self, fmt, *args):          # quiet unless verbose / ABRUN_HTTP_LOG
        if self.app.verbose or os.environ.get("ABRUN_HTTP_LOG"):
            super().log_message(fmt, *args)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_OPTIONS(self):                       # no CORS: a cross-origin preflight is refused, never answered
        self.send_json(403, {"error": "forbidden: no cross-origin access to abrun", "violations": []})

    def _allowed_hosts(self) -> set[str]:
        port = self.app.port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts |= {"127.0.0.1", "localhost"}
        return hosts

    def _origin_check(self, method: str) -> ApiError | None:
        """Loopback binding is not origin isolation: a page on any site can POST to 127.0.0.1 and a DNS-rebound
        name can read.  Host must be exactly 127.0.0.1:<port> or localhost:<port>; POST/DELETE must be JSON with
        Origin absent (curl, http.client) or same-origin (the UI)."""
        host = (self.headers.get("Host") or "").strip().lower()
        allowed = self._allowed_hosts()
        if host not in allowed:
            return ApiError(403, f"forbidden: Host {host!r} is not one of {sorted(allowed)}")
        if method in ("POST", "DELETE"):
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return ApiError(415, f"POST/DELETE need Content-Type: application/json (got {ctype or 'none'!r})")
            origin = self.headers.get("Origin")
            if origin is not None and origin.strip().lower() not in {f"http://{h}" for h in allowed}:
                return ApiError(403, f"forbidden: cross-origin request from {origin!r}")
        return None

    def _dispatch(self, method: str) -> None:
        err = self._origin_check(method)
        if err is not None:
            self.send_json(err.status, {"error": err.message, "violations": []})
            return
        url = urlsplit(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        path_matched = False
        for m_method, rx, name in ROUTES:
            m = rx.match(url.path)
            if not m:
                continue
            path_matched = True
            if m_method != method:
                continue
            try:
                body = self._read_body() if method in ("POST", "DELETE") else {}
                res = getattr(self, "h_" + name)(m, q, body)
            except ApiError as exc:
                self.send_json(exc.status, {"error": exc.message, "violations": exc.violations, **exc.extra})
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                self.send_json(500, {"error": f"server error: {exc!r}", "violations": []})
                return
            if res is None:
                return
            if len(res) == 3:
                self.send_text(res[0], res[1], res[2])
            else:
                self.send_json(res[0], res[1])
            return
        if path_matched:
            self.send_json(405, {"error": f"method {method} not allowed on {url.path}", "violations": []})
        else:
            self.send_json(404, {"error": f"no route for {url.path}", "violations": []})

    def _read_body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ApiError(400, "invalid Content-Length") from exc
        if n > BODY_MAX_BYTES:
            raise ApiError(413, f"request body {n} bytes > {BODY_MAX_BYTES}")
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw.strip():
            return {}
        try:
            d = json.loads(raw)
        except ValueError as exc:
            raise ApiError(400, f"invalid JSON body: {exc}") from exc
        if not isinstance(d, dict):
            raise ApiError(400, "JSON body must be an object")
        return d

    def send_json(self, status: int, obj) -> None:
        data = json.dumps(obj, default=str).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def send_text(self, status: int, text: str, ctype: str = "text/plain; charset=utf-8") -> None:
        data = text.encode() if isinstance(text, str) else text
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _since(self, q: dict) -> int:
        raw = self.headers.get("Last-Event-ID") or q.get("since") or "0"
        try:
            return max(0, int(raw))
        except ValueError:
            return 0

    def _stream(self, bus, since: int, close_on_terminal: bool) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for frame in sse_stream(bus, since, self.app.stop_event, HEARTBEAT_S, close_on_terminal):
                self.wfile.write(frame)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ---- static
    def h_index(self, m, q, body):
        return self._static_file("index.html")

    def h_static(self, m, q, body):
        return self._static_file(m.group("name"))

    def _static_file(self, name: str):
        root = STATIC_DIR.resolve()
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file():
            raise ApiError(404, f"no static file {name}")
        data = p.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", STATIC_TYPES.get(p.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return None

    # ---- status
    def h_health(self, m, q, body):
        st = self.app.worker.status()
        return 200, {"ok": True, "version": VERSION, "interpreter": INTERPRETER, "exlib": str(EXLIB_DIR / "exlib.py"),
                     "runs_dir": str(self.app.runs_dir), "port": self.app.port, "started": self.app.started,
                     "queue": {"running": st["running"], "queued": len(st["queued"]), "paused": st["paused"]},
                     "fake_pw": self.app.fake_pw, "reference_binary_id": self.app.reference_binary_id,
                     "timing_lock": str(runner.TIMING_LOCK), "run_log": str(runner.RUN_LOG)}

    def h_machine(self, m, q, body):
        return 200, {**self.app.machine, "worker": self.app._worker_status()}

    def h_binaries(self, m, q, body):
        inv = binaries_mod.inventory(refresh=q.get("refresh") in ("1", "true"), include_private=q.get("all") in ("1", "true"))
        return 200, {"generated": inv["generated"], "binaries": [b.to_dict() for b in inv["binaries"]],
                     "other_solvers": [b.to_dict() for b in inv["other_solvers"]], "hidden_private": inv["hidden_private"],
                     "hidden_dev": inv["hidden_dev"], "reference_binary_id": self.app.reference_binary_id,
                     "default_binary_id": "fake" if self.app.fake_pw else DEFAULT_FROZEN}

    def h_presets(self, m, q, body):
        return 200, {"presets": presets.list_presets()}

    def h_preset_input(self, m, q, body):
        pid = m.group("pid")
        if pid not in presets.PRESETS:
            raise ApiError(404, f"unknown preset {pid!r}")
        try:
            rec = presets.prepare(pid, self.app.runs_dir / "_preview", q.get("input_path") or None, write=False)
        except (presets.PresetError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc
        return 200, rec["input_text"], "text/plain; charset=utf-8"

    def h_modes(self, m, q, body):
        return 200, {"modes": launch.modes_list()}

    # ---- plan / runs
    def h_plan(self, m, q, body):
        req = launch.Request.from_dict(body)
        res = runner.plan_only(req)
        if res["violations"]:
            return 400, {"error": "invalid plan", "violations": res["violations"], "plan": res["plan"], "run_id_would_be": res["run_id_would_be"]}
        return 200, res

    def h_runs_post(self, m, q, body):
        req = launch.Request.from_dict(body)
        run = self.app.submit(req, allow_share_during_bench=bool(body.get("allow_share_during_bench", False)))
        return 201, {"id": run.id, "state": run.state, "position": run.position, "plan": run.plan.to_dict(), "run_dir": str(run.run_dir)}

    def h_ab_set(self, m, q, body):
        legs = body.get("legs") or list(DEFAULT_LEGS)
        bad = [l for l in legs if l not in launch.MODES]
        if bad:
            raise ApiError(400, f"unknown legs {bad}; one of {list(launch.MODES)}")
        common = {k: v for k, v in body.items() if k in launch.Request.__dataclass_fields__ and k not in ("mode", "derived_from")}
        if body.get("timed"):
            common["lock"] = "bench"                  # timed A/B sets take machlock bench on every leg
        ref_bin = body.get("reference_binary_id") or self.app.reference_binary_id
        reqs = []
        for leg in legs:
            d = dict(common, mode=leg)
            if leg == "reference":
                d["binary_id"] = ref_bin
            reqs.append(launch.Request.from_dict(d))
        # validate every leg before queueing any
        for r in reqs:
            res = runner.plan_only(r)
            if res["violations"]:
                raise ApiError(400, f"leg {r.mode} invalid", res["violations"])
            dup = self.app.duplicate_of(r)
            if dup:
                raise ApiError(409, f"leg {r.mode}: an identical plan is already queued or running: {dup}", extra={"id": dup})
        allow_share = bool(body.get("allow_share_during_bench", False))
        runs = [self.app.submit(r, allow_share_during_bench=allow_share) for r in reqs]
        ref_run = next((r for r in runs if r.request.mode == "reference"), None)
        pinned = False
        if ref_run is not None and ref_run.prepared.get("input_sha256"):
            runner.Pins().pin(ref_run.prepared["input_sha256"], ref_run.id)
            ref_run.pinned_reference = True
            ref_run.save()
            pinned = True
        return 201, {"ids": [r.id for r in runs], "ref": ref_run.id if ref_run else None, "pinned": pinned}

    def h_runs_get(self, m, q, body):
        rows = self.app.summaries()
        for key in ("state", "preset", "mode"):
            if q.get(key):
                rows = [r for r in rows if r.get(key) == q[key]]
        if q.get("binary"):
            rows = [r for r in rows if r.get("binary_id") == q["binary"]]
        try:
            limit = int(q.get("limit") or 200)
        except ValueError:
            limit = 200
        return 200, {"runs": rows[:limit], "total": len(rows)}

    def h_run_get(self, m, q, body):
        return 200, self.app.record(m.group("rid"))

    def h_run_delete(self, m, q, body):
        rid = m.group("rid")
        live = runner.REGISTRY.get(rid)
        if live is not None and not live.terminal:
            raise ApiError(409, f"run {rid} is {live.state}; only terminal runs can be deleted")
        self.app.record(rid)
        if not runner.delete_run(rid, self.app.runs_dir):
            raise ApiError(404, f"run {rid} has no directory to delete")
        return 200, {"deleted": True, "id": rid}

    def h_run_log(self, m, q, body):
        rid = m.group("rid")
        try:
            since = max(0, int(q.get("since") or 0))
        except ValueError:
            since = 0
        try:
            mx = max(1, min(int(q.get("max") or LOG_MAX_DEFAULT), 20000))
        except ValueError:
            mx = LOG_MAX_DEFAULT
        stream = q.get("stream") or "both"
        if stream not in ("both", "out", "err", "guard", "all"):
            raise ApiError(400, "stream must be both|out|err|guard|all")
        live = runner.REGISTRY.get(rid)
        if live is not None:
            lines, nxt = live.log.since(since, mx, None if stream in ("both", "all") else stream)
            return 200, {"next": nxt, "state": live.state, "state_why": live.state_why, "state_since": live.state_since,
                         "lines": lines, "progress": dict(live.progress), "suppressed_profile_lines": live.suppressed_profile_lines,
                         "event_seq": live.bus.seq, "live": True}
        rec = self.app.record(rid)
        all_lines, suppressed = synth_log_lines(self.app.run_dir_of(rid, rec))
        sel = [l for l in all_lines if l[0] > since and (stream in ("both", "all") or l[1] == stream)][:mx]
        nxt = sel[-1][0] if sel else since
        return 200, {"next": nxt, "state": rec.get("state"), "state_why": rec.get("state_why"), "state_since": rec.get("finished"),
                     "lines": sel, "progress": rec.get("progress") or {}, "suppressed_profile_lines": suppressed,
                     "event_seq": 0, "live": False}

    def h_run_events(self, m, q, body):
        rid = m.group("rid")
        since = self._since(q)
        live = runner.REGISTRY.get(rid)
        if live is not None:
            self._stream(live.bus, since, close_on_terminal=True)
            return None
        rec = self.app.record(rid)
        status = {"done": "success", "cancelled": "interrupted"}.get(rec.get("state"), "error")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(sse_frame(1, TERMINAL_EVENT, {"run_id": rid, "status": status, "seconds": rec.get("wall_runner_s"), "replayed": True}))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return None

    def h_events_global(self, m, q, body):
        self._stream(runner.global_bus, self._since(q), close_on_terminal=False)
        return None

    def h_run_file(self, m, q, body):
        rid, name = m.group("rid"), m.group("name")
        if name not in FILE_WHITELIST and not RE_TRACE_FILE.match(name):
            raise ApiError(404, f"{name} is not a downloadable run file")
        p = self.app.run_dir_of(rid) / name
        if not p.is_file():
            raise ApiError(404, f"{name} not found for run {rid}")
        ctype = "application/json; charset=utf-8" if p.suffix == ".json" else "text/plain; charset=utf-8"
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise ApiError(500, f"cannot read {name}: {exc}") from exc
        return 200, data, ctype

    def h_run_cancel(self, m, q, body):
        rid = m.group("rid")
        live = runner.REGISTRY.get(rid)
        if live is None:
            self.app.record(rid)
            raise ApiError(409, f"run {rid} is not live (terminal)")
        if live.terminal:
            raise ApiError(409, f"run {rid} is already {live.state}")
        if self.app.worker.dequeue_cancel(rid):
            return 200, {"state": "cancelled", "id": rid}
        return 200, {"state": runner.cancel(live), "id": rid}

    def h_run_rerun(self, m, q, body):
        rid = m.group("rid")
        parent = self.app.record(rid)
        d = dict(parent.get("request") or {})
        for k in ("mode", "binary_id", "lock", "timed", "label", "ranks", "nk", "omp", "min_gpu_dim", "profile_json", "report",
                  "trace", "verify_dylib", "allow_omp_gt1", "allow_dev", "force_profile_json", "purge_tmp"):
            if k in body:
                d[k] = body[k]
        d["derived_from"] = rid
        if "label" not in body:
            d["label"] = parent.get("label") or d.get("label") or ""
        allow_share = bool(body.get("allow_share_during_bench", parent.get("allow_share_during_bench", False)))
        run = self.app.submit(launch.Request.from_dict(d), allow_share_during_bench=allow_share)
        return 201, {"id": run.id, "state": run.state, "position": run.position, "plan": run.plan.to_dict(),
                     "derived_from": rid, "run_dir": str(run.run_dir)}

    def h_run_pin(self, m, q, body):
        rid = m.group("rid")
        rec = self.app.record(rid)
        sha = rec.get("input_sha256")
        if not sha:
            raise ApiError(409, f"run {rid} has no input sha256 to pin")
        runner.Pins().pin(sha, rid)
        live = runner.REGISTRY.get(rid)
        if live is not None:
            live.pinned_reference = True
            live.save()
        else:
            p = self.app.run_dir_of(rid, rec) / "run.json"
            try:
                rec["pinned_reference"] = True
                runner._write_json(p, rec)
            except OSError:
                pass
        return 200, {"input_sha256": sha, "reference": rid}

    def h_run_unpin(self, m, q, body):
        rid = m.group("rid")
        changed = runner.Pins().unpin(rid)
        live = runner.REGISTRY.get(rid)
        if live is not None and changed:
            live.pinned_reference = False
            live.save()
        return 200, {"unpinned": changed, "id": rid}

    # ---- queue
    def h_queue_get(self, m, q, body):
        return 200, self.app.worker.status()

    def h_queue_pause(self, m, q, body):
        return 200, {"paused": self.app.worker.pause()}

    def h_queue_resume(self, m, q, body):
        return 200, {"paused": self.app.worker.resume()}

    def h_queue_reorder(self, m, q, body):
        ids = body.get("ids")
        if not isinstance(ids, list):
            raise ApiError(400, "body must be {\"ids\": [...]}")
        return 200, {"queued": self.app.worker.reorder([str(i) for i in ids])}

    # ---- compare / export
    def _compare(self, q: dict) -> dict:
        ids = [i for i in (q.get("ids") or "").split(",") if i]
        if not ids:
            raise ApiError(400, "ids=a,b,c required")
        recs = [self.app.record(i) for i in ids]
        tol = None
        if q.get("energy_tol"):
            try:
                tol = float(q["energy_tol"])
            except ValueError as exc:
                raise ApiError(400, "energy_tol must be a number") from exc
        return compare_mod.compare(recs, q.get("ref") or None, tol, runner.Pins().load())

    def h_compare(self, m, q, body):
        return 200, self._compare(q)

    def h_export(self, m, q, body):
        fmt = q.get("format") or "md"
        if fmt not in ("md", "csv", "json"):
            raise ApiError(400, "format must be md|csv|json")
        res = self._compare(q)
        ctype = {"md": "text/markdown; charset=utf-8", "csv": "text/csv; charset=utf-8", "json": "application/json; charset=utf-8"}[fmt]
        return 200, compare_mod.export(res, fmt), ctype


# --------------------------------------------------------------------------- CLI entry

def main(args) -> int:
    port = getattr(args, "port", None) or DEFAULT_PORT
    fake = bool(getattr(args, "fake_pw", False))
    srv = Server(port=port, fake_pw=fake, log=lambda s: print(s, flush=True), verbose=bool(os.environ.get("ABRUN_HTTP_LOG")))
    try:
        srv.start()
    except OSError as exc:
        print(f"abrun serve: cannot bind {HOST}:{port}: {exc}", flush=True)
        return 1
    print(f"abrun {VERSION} serving on {srv.url}  runs: {srv.runs_dir}  fake pw.x: {fake}  "
          f"timing lock: {runner.TIMING_LOCK}  machlock: {launch.MACHLOCK}  (Ctrl-C stops: cancels the running job, persists the queue)",
          flush=True)
    if getattr(args, "open", False):
        try:
            webbrowser.open(srv.url)
        except Exception:  # noqa: BLE001
            pass
    stop = threading.Event()

    def _on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)     # a closed terminal stops the server the same graceful way
    while not stop.wait(0.5):
        pass
    print("abrun serve: stopping (cancelling the running job, persisting the queue)", flush=True)
    srv.stop(cancel_running=True)
    print("abrun serve: stopped", flush=True)
    return 0
