"""Live inverse-design server: the optimizer runs continuously and streams frames to the browser.

    ~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/live/server.py [--port 8770]

Standard library HTTP only (numpy/scipy for the physics):
    GET  /         the page (index.html next to this file)
    GET  /events   server-sent events, one JSON frame per optimizer step (throttled)
    POST /params   JSON patch of parameters; applied by the worker before its next step
    GET  /export   current design (density, permittivity, parameters, metrics) as JSON
    POST /verify   {"res": 50}: export the current design and run meep_check.py on it in a subprocess (~20-60 s);
                   progress and results arrive in the frames ("meep"); the plot is served under /runs/

Parameter classes, by what a change costs:
    live      radius, beta, lr, ratio, crosstalk, auto_beta, paused, view  -> next step (tens of ms)
    rebuild   kind, wavelength, wavelength2, width, width_out, design, separation
              -> new condensed operators (~1 s per wavelength; cached, so returning to a setting is instant);
                 the current density is kept as a warm start
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
sys.path.insert(0, str(HERE.parent))
import devices  # noqa: E402
import optimize  # noqa: E402

GEOMETRY = ("wavelength", "wavelength2", "width", "width_out", "design", "separation", "bandwidth")
TARGETS = ("ratio", "crosstalk")
FRAME_DT = 0.05          # at most 20 frames/s to the browser
HISTORY = 600            # points kept for the chart
CACHE = 8                # cached Objectives (each holds the exterior LU per wavelength)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.pending: dict = {}
        self.frame = "{}"
        self.version = 0
        self.cache: OrderedDict = OrderedDict()
        self.kind = "bend"
        self.params = dict(devices.DEFAULTS[self.kind])
        self.ctl = dict(radius=0.12, beta=4.0, lr=0.03, auto_beta=True, paused=False, view=0, init="uniform")
        self.status = "building"
        self.iter = 0                     # global iteration count across device switches (chart x axis)
        self.hist: list[dict] = []
        self.binary: dict | None = None
        self.opt: optimize.Optimizer | None = None
        self.cont = optimize.Continuation()
        self.meep: dict = dict(state="idle")
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- MEEP verification
    def verify(self, res: float):
        if self.meep.get("state") == "running" or self.opt is None:
            return
        design = self.export()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = RUNS / "live" / f"{design['kind']}-{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "design.json").write_text(json.dumps(design))
        self.meep = dict(state="running", res=res, iter=self.iter, kind=design["kind"], started=time.time())
        self._republish()

        def job():
            cmd = [sys.executable, str(HERE.parent / "meep_check.py"), str(out / "design.json"),
                   "--res", f"{res:g}", "--out", str(out)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            rep_path = out / "meep_check.json"
            if rep_path.exists():
                rep = json.loads(rep_path.read_text())
                self.meep.update(state="done", rows=rep["rows"], wall=rep["meep_wall_s"],
                                 png=f"/runs/{(out / 'meep_check.png').relative_to(RUNS)}")
            else:
                self.meep.update(state="error", error=(p.stderr or p.stdout)[-400:])
            self._republish()

        threading.Thread(target=job, daemon=True).start()

    def _republish(self):
        with self.lock:
            self._publish()

    # ---------------------------------------------------------------- parameter handling
    def update(self, patch: dict):
        with self.cv:
            self.pending.update(patch)
            self.cv.notify_all()

    def _objective(self, dev: devices.Device) -> optimize.Objective:
        key = (dev.kind, tuple((k, dev.params.get(k)) for k in GEOMETRY))
        if key in self.cache:
            self.cache.move_to_end(key)
            obj = self.cache[key]
            obj.dev = dev
            obj.retarget(dev.terms)
            return obj
        self.status = "building"
        self._publish()
        obj = optimize.Objective(dev)
        self.cache[key] = obj
        while len(self.cache) > CACHE:
            self.cache.popitem(last=False)[1].pool.shutdown(wait=False)
        return obj

    def _apply(self, patch: dict):
        """Apply a control patch; True if it changes the optimization (so convergence restarts)."""
        rebuild, retarget, new_kind = False, False, False
        kind = patch.pop("kind", self.kind)
        if kind != self.kind:
            self.kind, self.params = kind, dict(devices.DEFAULTS[kind])
            rebuild = new_kind = True
        for k, v in patch.items():
            if k in GEOMETRY and k in self.params and self.params[k] != v:
                self.params[k] = float(v)
                rebuild = True
            elif k in TARGETS and k in self.params:
                self.params[k] = float(v)
                retarget = True
            elif k in self.ctl:
                self.ctl[k] = v
        o = self.opt
        if o is None:
            dev = devices.build(self.kind, **self.params)
            self.opt = optimize.Optimizer(dev, radius=self.ctl["radius"], beta=self.ctl["beta"], lr=self.ctl["lr"],
                                          obj=self._objective(dev))
            return True
        if rebuild:
            dev = devices.build(self.kind, **self.params)
            o.set_device(dev, self._objective(dev))
            self.binary = None
            if new_kind:  # a different device: start over rather than from the last device's layout
                o.reset(self.ctl["init"], seed=int(time.time()))
                self.ctl["beta"] = 4.0
                self.hist.clear()
            elif self.ctl["auto_beta"]:  # same device, new geometry: warm start, but soft enough to move
                self.ctl["beta"] = min(float(self.ctl["beta"]), 8.0)
        elif retarget:
            o.dev = devices.build(self.kind, **self.params)
            o.obj.retarget(o.dev.terms)
        if abs(o.radius - float(self.ctl["radius"])) > 1e-9:
            o.set_radius(float(self.ctl["radius"]))
        o.beta, o.lr = float(self.ctl["beta"]), float(self.ctl["lr"])
        if patch.get("reset"):
            o.reset(patch.get("init", self.ctl["init"]), seed=int(time.time()))
            self.ctl["beta"] = o.beta = float(patch.get("beta", 4.0))
            self.binary = None
        if patch.get("check"):
            self.binary = o.binary_check()
        restart = rebuild or retarget or bool(patch.keys() & {"radius", "beta", "lr", "auto_beta", "reset"})
        if restart:
            self.cont.reset()
        return restart

    # ---------------------------------------------------------------- worker
    def _worker(self):
        last_frame = 0.0
        while True:
            with self.cv:
                idle = self.status in ("paused", "converged")
                if idle and not self.pending:
                    self.cv.wait(timeout=1.0)
                patch, self.pending = self.pending, {}
            try:
                with self.lock:
                    if patch or self.opt is None:
                        # view/check/pause patches leave a converged design converged
                        if self._apply(patch) or self.status == "paused":
                            self.status = "running"
                    if self.ctl["paused"]:
                        self.status = "paused"
                        self._publish()
                        continue
                    if self.status == "converged":
                        continue
                    rec = self.opt.step()
                    self.iter += 1
                    self.hist.append(dict(it=self.iter, loss=rec["loss"], T=rec["T"]))
                    del self.hist[:-HISTORY]
                    self._convergence(rec)
                now = time.perf_counter()
                if now - last_frame > FRAME_DT or self.status != "running":
                    last_frame = now
                    with self.lock:
                        self._publish()
            except Exception as e:  # keep the server alive; surface the error in the page
                self.status = f"error: {e!r}"
                self.ctl["paused"] = True
                self._publish()

    def _convergence(self, rec: dict):
        """Idle when the loss stops moving; with auto_beta, first sharpen the projection toward a binary design."""
        state = self.cont.update(self.opt, rec, self.ctl["auto_beta"])
        if state == "raised":
            self.ctl["beta"] = self.opt.beta
        elif state == "converged":
            self.binary = self.opt.binary_check()
            self.status = "converged"

    def _publish(self):
        o = self.opt
        if o is None:
            frame = dict(status=self.status)
        else:
            d, g = o.dev, o.dev.grid
            last = getattr(o, "last", None)
            rp = last["rp"] if last is not None else o.density()[1]  # the density the shown field came from
            epsn = (o.eps_full(rp) - d.eps_min) / (d.eps_max - d.eps_min)
            k = min(int(self.ctl["view"]), len(o.obj.sims) - 1)
            fld = None
            if last is not None and len(last["fields"]) == len(o.obj.sims):
                ez = o.obj.field(k, last["fields"][k])
                ez = ez / (np.abs(ez[g.npml:-g.npml, g.npml:-g.npml]).max() + 1e-30)
                fld = b64(np.clip(127.5 + 127.5 * ez.real, 0, 255))
            sx, sy = d.design
            frame = dict(
                status=self.status, iter=self.iter, kind=self.kind, params=self.params, ctl=self.ctl,
                nx=g.nx, ny=g.ny, dx=g.dx, npml=g.npml, design=[sx.start, sx.stop, sy.start, sy.stop],
                eps=b64(np.clip(epsn * 255, 0, 255)), field=fld,
                views=[f"{s.lap.wavelength * 1e3:.0f} nm from {s.src.port.name}" for s in o.obj.sims],
                ports={n: dict(axis=p.axis, index=p.index, lo=p.lo, hi=p.hi) for n, p in d.ports.items()},
                terms=[dict(label=t.label, target=t.target, wavelength=t.wavelength) for t in d.terms],
                last=None if last is None else dict(loss=last["loss"], T=last["T"], gray=last["gray"], ms=last["ms"]),
                binary=self.binary, hist=self.hist, meep=self.meep,
            )
        with self.cv:
            self.frame = json.dumps(frame)
            self.version += 1
            self.cv.notify_all()

    def export(self) -> dict:
        with self.lock:
            return dict(self.opt.export(), ctl=self.ctl)


SESSION: Session


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with SESSION.cv:
                        if SESSION.version == seen:
                            SESSION.cv.wait(timeout=10)
                        frame, ver = SESSION.frame, SESSION.version
                    if ver == seen:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        seen = ver
                        self.wfile.write(b"data: " + frame.encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith("/runs/") and self.path.endswith(".png"):
            f = (RUNS / self.path[len("/runs/"):]).resolve()
            if RUNS.resolve() not in f.parents or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, f.read_bytes(), "image/png")
        elif self.path == "/export":
            body = json.dumps(SESSION.export()).encode()
            self._send(200, body, "application/json",
                       {"Content-Disposition": f'attachment; filename="{SESSION.kind}_design.json"'})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/params", "/verify"):
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            patch = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b"bad json", "text/plain")
        if self.path == "/verify":
            SESSION.verify(float(patch.get("res", 50)))
        else:
            SESSION.update(patch)
        self._send(200, b'{"ok":true}', "application/json")


def main():
    global SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    SESSION = Session()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"inverse-design live server on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
