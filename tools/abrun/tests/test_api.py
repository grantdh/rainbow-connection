"""server.py in-process on an ephemeral port with the fake pw.x: health, static, binaries, plan, run + SSE,
log, files, compare/export, rerun, pin, cancel, timing-lock wait, failure tail, duplicates/404s, queue
pause/resume/reorder, A/B set, global SSE replay with Last-Event-ID, and the persisted-queue restart.

Never touches the real locks or logs: TIMING_LOCK, RUN_LOG, RUNS_DIR and launch.MACHLOCK are monkeypatched
in-process (the Server points MACHLOCK at /nonexistent itself in fake mode) and the fake binary is launched
directly (no mpirun, no machlock).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from abrun import launch, runner, server

FAKE_ENV = ("ABRUN_FAKE_SLEEP", "ABRUN_FAKE_RC", "ABRUN_FAKE_SCHEMA")


class ApiCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="abrun-api-"))
        cls.saved = (runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK, server.HEARTBEAT_S)
        runner.TIMING_LOCK = cls.tmp / "timing-lock"
        runner.RUN_LOG = cls.tmp / "solver-runs.log"
        runner.RUNS_DIR = cls.tmp / "runs"
        launch.MACHLOCK = Path("/nonexistent/machlock.py")
        server.HEARTBEAT_S = 0.5
        (cls.tmp / "libapplebottom.dylib").write_text("fake")
        os.environ["ABRUN_FAKE_DYLIB"] = str(cls.tmp / "libapplebottom.dylib")
        cls.srv = server.Server(port=0, fake_pw=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop(cancel_running=True)
        runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK, server.HEARTBEAT_S = cls.saved
        for k in FAKE_ENV + ("ABRUN_FAKE_DYLIB",):
            os.environ.pop(k, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tearDown(self):
        for k in FAKE_ENV:
            os.environ.pop(k, None)
        try:
            runner.TIMING_LOCK.unlink()
        except OSError:
            pass

    # ---- http helpers
    def req(self, method, path, body=None, headers=None, raw_body=None):
        """Like the UI's fetch wrapper: every POST/DELETE carries Content-Type: application/json (the server
        refuses them otherwise, see test_13); http.client sends Host: 127.0.0.1:<port> and no Origin."""
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=30)
        data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
        h = {"Content-Type": "application/json"} if method in ("POST", "DELETE") else {}
        h.update(headers or {})
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        ctype = resp.getheader("Content-Type") or ""
        payload = json.loads(raw) if "json" in ctype else raw.decode(errors="replace")
        return resp.status, payload, dict(resp.getheaders())

    def get(self, path, headers=None):
        return self.req("GET", path, headers=headers)

    def post(self, path, body=None):
        return self.req("POST", path, body or {})

    def delete(self, path):
        return self.req("DELETE", path)

    def wait_state(self, rid, states, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st, rec, _ = self.get(f"/api/runs/{rid}")
            if st == 200 and rec["state"] in states:
                return rec
            time.sleep(0.1)
        self.fail(f"{rid} did not reach {states} within {timeout}s (last: {rec.get('state')} {rec.get('state_why')})")

    def sse(self, path, timeout=15.0, stop_type="muppet.done", run_id=None, headers=None, max_events=5000):
        """Read SSE frames with http.client until stop_type (for run_id) or timeout; returns [(id, type, data)]."""
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=timeout)
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertIn("text/event-stream", resp.getheader("Content-Type"))
        events, cur = [], None
        t0 = time.time()
        try:
            while time.time() - t0 < timeout and len(events) < max_events:
                line = resp.readline()
                if not line:
                    break
                line = line.rstrip(b"\n")
                if line.startswith(b"id: "):
                    cur = int(line[4:])
                elif line.startswith(b"data: "):
                    d = json.loads(line[6:])
                    events.append((cur, d["type"], d["data"]))
                    if d["type"] == stop_type and (run_id is None or d["data"].get("run_id") == run_id):
                        break
                elif line.startswith(b":"):
                    events.append((None, "heartbeat", {}))
                    if stop_type == "heartbeat":
                        break
        finally:
            conn.close()
        return events

    def wait_idle(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st, q, _ = self.get("/api/queue")
            if q["running"] is None and not q["queued"]:
                return
            time.sleep(0.2)
        self.fail("worker did not become idle")

    def run_and_wait(self, body, timeout=30.0):
        st, r, _ = self.post("/api/runs", body)
        self.assertEqual(st, 201, r)
        return self.wait_state(r["id"], ("done", "failed", "cancelled"), timeout)

    # ---- tests (numbered: later ones compare runs made by earlier ones)
    def test_01_health_static_catalog(self):
        st, h, hdr = self.get("/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(h["ok"])
        self.assertEqual(h["version"], "0.1")
        self.assertTrue(h["fake_pw"])
        self.assertEqual(h["reference_binary_id"], "reference-fake")
        self.assertEqual(h["runs_dir"], str(runner.RUNS_DIR))
        self.assertEqual(h["queue"], {"running": None, "queued": 0, "paused": False})
        self.assertEqual(hdr.get("Cache-Control"), "no-store")
        st, page, hdr = self.get("/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", hdr["Content-Type"])
        self.assertIn("Wall times are observations unless a run shows grade=bench", page)
        for f in ("app.js", "style.css"):
            st, _, hdr = self.get(f"/static/{f}")
            self.assertEqual(st, 200, f)
        self.assertEqual(self.get("/static/../server.py")[0], 404)
        self.assertEqual(self.get("/static/nope.js")[0], 404)
        st, m, _ = self.get("/api/machine")
        self.assertEqual(st, 200)
        for k in ("updated", "load1", "timing_lock", "machlock", "foreign_solvers", "power", "disk_free_gb", "timed_possible", "timed_blockers", "worker"):
            self.assertIn(k, m)
        self.assertFalse(m["timing_lock"]["present"])
        self.assertFalse(m["machlock"]["bench_held"])
        self.assertEqual(m["worker"]["state"], "idle")
        st, b, _ = self.get("/api/binaries")
        ids = {x["id"] for x in b["binaries"]}
        self.assertIn("fake", ids)
        self.assertIn("reference-fake", ids)
        self.assertEqual(b["reference_binary_id"], "reference-fake")
        self.assertEqual(b["default_binary_id"], "fake")
        st, mo, _ = self.get("/api/modes")
        self.assertEqual([m["id"] for m in mo["modes"]], ["reference", "ab_cpu", "ab_auto", "ab_gpu", "ab_gpu_all"])
        st, p, _ = self.get("/api/presets")
        self.assertEqual([x["id"] for x in p["presets"]], ["si2", "C", "B", "A", "custom"])
        si2 = p["presets"][0]
        self.assertTrue(si2["available"])
        self.assertEqual(si2["nat"], 2)
        st, text, hdr = self.get("/api/presets/si2/input")
        self.assertEqual(st, 200)
        self.assertIn("text/plain", hdr["Content-Type"])
        self.assertIn("outdir = './tmp'", text)
        self.assertIn("pseudo_dir = './pseudo'", text)
        self.assertEqual(self.get("/api/presets/nope/input")[0], 404)
        self.assertEqual(self.get("/api/presets/custom/input")[0], 400)

    def test_02_plan(self):
        st, res, _ = self.post("/api/plan", {"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all", "ranks": 1})
        self.assertEqual(st, 200, res)
        self.assertEqual(res["violations"], [])
        p = res["plan"]
        self.assertEqual(p["env_set"]["AB_MODE"], "gpu")
        self.assertEqual(p["env_set"]["AB_MIN_GPU_DIM"], "0")
        self.assertEqual(p["env_set"]["AB_PROFILE"], "1")
        self.assertTrue(p["env_set"]["AB_PROFILE_JSON"].endswith("/ab_profile.json"))
        self.assertEqual(p["env_set"]["DYLD_PRINT_LIBRARIES"], "1")
        self.assertEqual(p["env_set"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(p["launcher"], "direct")
        self.assertEqual(p["shim_argv"][1], str(launch.SHIM))
        self.assertEqual(p["wrapper_argv"], [])
        self.assertNotIn("/usr/bin/nice", p["shell"])
        self.assertTrue(res["run_id_would_be"].endswith("-si2-ab_gpu_all-fake"))
        st, res, _ = self.post("/api/plan", {"preset": "si2", "binary_id": "fake", "mode": "reference"})
        self.assertEqual(st, 400)
        self.assertEqual(res["error"], "invalid plan")
        self.assertTrue(any("apple-bottom-free" in v for v in res["violations"]))
        st, res, _ = self.post("/api/plan", {"preset": "si2", "binary_id": "nope", "mode": "ab_cpu"})
        self.assertEqual(st, 400)
        self.assertTrue(any("unknown binary" in v for v in res["violations"]))
        st, res, _ = self.req("POST", "/api/plan", headers={"Content-Type": "application/json"})
        self.assertEqual(st, 200)      # empty body = defaults (preset si2, default frozen id) -> 200 or 400, never 500
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=10)
        conn.request("POST", "/api/plan", body=b"{not json", headers={"Content-Type": "application/json"})
        r = conn.getresponse(); self.assertEqual(r.status, 400); r.read(); conn.close()

    def test_03_gpu_all_run_sse_log_record(self):
        os.environ["ABRUN_FAKE_SCHEMA"] = "new"
        st, r, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all", "lock": "share", "label": "api gpu"})
        self.assertEqual(st, 201, r)
        rid = r["id"]
        self.assertEqual(r["state"], "queued")
        self.assertIn("plan", r)
        events = self.sse(f"/api/runs/{rid}/events?since=0", timeout=20, run_id=rid)
        types = [e[1] for e in events if e[1] != "heartbeat"]
        for t in ("execution_start", "muppet.log", "progress", "executed", "muppet.done"):
            self.assertIn(t, types, types)
        order = [types.index(t) for t in ("execution_start", "muppet.log", "progress", "executed", "muppet.done")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(types[-1], "muppet.done")
        ids = [e[0] for e in events if e[0] is not None]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(e[2].get("run_id") == rid for e in events if e[1] != "heartbeat"))
        rec = self.wait_state(rid, ("done",), 10)
        res, ab = rec["result"], rec["ab"]
        self.assertAlmostEqual(res["total_energy_Ry"], -22.83836927)
        self.assertAlmostEqual(res["wall_qe_s"], 121.49)
        self.assertEqual(res["scf_iterations"], 3)
        self.assertEqual(ab["schema"], "v2")
        self.assertEqual(ab["evidence_source"], "profile_json")
        self.assertTrue(ab["gpu_verified"], ab["gpu_verified_reason"])
        self.assertEqual(ab["totals"]["gpu_calls"], ab["totals"]["calls"])
        self.assertEqual(ab["totals"]["calls"], 1200)
        self.assertIsNotNone(ab["lib_loaded"])
        self.assertTrue(ab["lib_ok"])
        self.assertEqual(rec["etiquette"]["lock_effective"], "share")
        self.assertEqual(rec["etiquette"]["timing_grade"], "untimed")
        self.assertIn("machine_at_acquisition", rec["etiquette"])
        self.assertEqual(rec["etiquette"]["machine_at_acquisition"]["nice"], 10)
        # log endpoint
        st, lg, _ = self.get(f"/api/runs/{rid}/log?since=0")
        self.assertEqual(st, 200)
        self.assertEqual(lg["state"], "done")
        self.assertTrue(lg["live"])
        texts = [l[2] for l in lg["lines"]]
        self.assertTrue(any("iteration #" in t for t in texts))
        self.assertTrue(any("AB_REPORT[" in t for t in texts))
        self.assertTrue(any("suppressed" in t for t in texts), "the +N AB_PROFILE marker is in the stream")
        self.assertEqual(lg["suppressed_profile_lines"], 1200 - runner.PROFILE_STREAM_FIRST)
        self.assertLessEqual(sum(1 for t in texts if t.startswith("AB_PROFILE ")), runner.PROFILE_STREAM_FIRST)
        st, lg2, _ = self.get(f"/api/runs/{rid}/log?since={lg['next']}")
        self.assertEqual(lg2["lines"], [])
        st, lg3, _ = self.get(f"/api/runs/{rid}/log?since=0&stream=out&max=2")
        self.assertEqual(len(lg3["lines"]), 2)
        self.assertTrue(all(l[1] == "out" for l in lg3["lines"]))
        self.assertEqual(self.get(f"/api/runs/{rid}/log?stream=bogus")[0], 400)
        # files whitelist
        for f in ("input.in", "out.txt", "err.txt", "ab_profile.json", "machine.json", "summary.json", "run.json"):
            st, _, _ = self.get(f"/api/runs/{rid}/files/{f}")
            self.assertEqual(st, 200, f)
        self.assertEqual(self.get(f"/api/runs/{rid}/files/pseudo")[0], 404)
        self.assertEqual(self.get(f"/api/runs/{rid}/files/..%2Frun.json")[0], 404)
        st, prof, _ = self.get(f"/api/runs/{rid}/files/ab_profile.json")
        self.assertIn("routines", prof)
        # runs list row
        st, lst, _ = self.get("/api/runs?mode=ab_gpu_all")
        row = next(x for x in lst["runs"] if x["id"] == rid)
        self.assertEqual(row["label"], "api gpu")
        self.assertEqual(row["gpu_calls"], 1200)
        self.assertEqual(row["gpu_badge"], "green")
        self.assertTrue(row["run_dir"].endswith(rid))
        # the per-run stream after the terminal event replays and closes
        events2 = self.sse(f"/api/runs/{rid}/events", timeout=5, run_id=rid, headers={"Last-Event-ID": str(ids[-2])})
        self.assertEqual([e[1] for e in events2 if e[1] != "heartbeat"], ["muppet.done"])
        # solver-runs.log line in exlib's format
        line = runner.RUN_LOG.read_text().splitlines()[-1]
        self.assertRegex(line, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d rc=0 wall=[\d.]+s load1=[\d.]+ idle_at_start=(True|False) lock=share "
                               r"label=abrun si2 ab_gpu_all fake cwd=" + re.escape(rec["run_dir"]) + "$")
        self.__class__.gpu_id = rid

    def test_04_gpu_all_old_schema(self):
        os.environ["ABRUN_FAKE_SCHEMA"] = "old"
        rec = self.run_and_wait({"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all", "lock": "share"})
        self.assertEqual(rec["state"], "done", rec["state_why"])
        self.assertEqual(rec["ab"]["schema"], "v1")
        self.assertTrue(rec["ab"]["gpu_verified"])
        self.assertIsNone(rec["ab"]["totals"]["fallback_breakdown"])

    def test_05_cpu_reference_compare_export(self):
        cpu = self.run_and_wait({"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share"})
        self.assertEqual(cpu["state"], "done", cpu["state_why"])
        self.assertEqual(cpu["ab"]["totals"]["gpu_calls"], 0)
        self.assertTrue(cpu["ab"]["gpu_verified"])
        ref = self.run_and_wait({"preset": "si2", "binary_id": "reference-fake", "mode": "reference", "lock": "share"})
        self.assertEqual(ref["state"], "done", ref["state_why"])
        self.assertEqual(ref["ab"]["evidence_source"], "none")
        self.assertIsNone(ref["ab"]["gpu_verified"])
        self.assertFalse(any(k.startswith("AB_") for k in ref["plan"]["env_set"]))
        gpu = self.gpu_id
        st, res, _ = self.get(f"/api/compare?ids={cpu['id']},{gpu},{ref['id']}&ref={ref['id']}")
        self.assertEqual(st, 200, res)
        self.assertEqual(res["ref"], ref["id"])
        self.assertEqual([r["id"] for r in res["rows"]], [ref["id"], cpu["id"], gpu])
        for r in res["rows"][1:]:
            self.assertEqual(r["dE_vs_ref_Ry"], 0.0)
            self.assertFalse(r["same_pw_sha"])
            self.assertEqual(r["tolerance_Ry"], 1e-5)
            self.assertFalse(r["wall_comparable"])
        self.assertEqual(res["sections"]["same_binary"], [])
        self.assertEqual(res["sections"]["cross_build"], [cpu["id"], gpu])
        self.assertTrue(res["markdown"].startswith("| run | binary | mode | E_tot (Ry) | dE vs ref | tol | iters | QE wall (s), observation | grade | GEMM calls | GPU calls | fallbacks |"))
        st, res2, _ = self.get(f"/api/compare?ids={cpu['id']},{gpu},{ref['id']}&ref={cpu['id']}")
        self.assertEqual([r["id"] for r in res2["rows"]], [cpu["id"], gpu, ref["id"]])   # same-binary row first
        self.assertTrue(res2["rows"][1]["same_pw_sha"])
        self.assertEqual(res2["rows"][1]["tolerance_Ry"], 1e-8)
        self.assertEqual(res2["sections"]["same_binary"], [gpu])
        self.assertEqual(self.get("/api/compare")[0], 400)
        self.assertEqual(self.get("/api/compare?ids=nope")[0], 404)
        self.assertEqual(self.get(f"/api/compare?ids={gpu}&energy_tol=x")[0], 400)
        st, md, hdr = self.get(f"/api/export?ids={cpu['id']},{gpu}&ref={cpu['id']}&format=md")
        self.assertIn("text/markdown", hdr["Content-Type"])
        self.assertTrue(md.startswith("| run |"))
        self.assertIn(gpu, md)
        self.assertNotIn(ref["id"], md)
        st, csv_text, hdr = self.get(f"/api/export?ids={cpu['id']},{gpu}&ref={cpu['id']}&format=csv")
        self.assertIn("text/csv", hdr["Content-Type"])
        self.assertTrue(csv_text.startswith("id,binary_short,mode,"))
        st, js, hdr = self.get(f"/api/export?ids={cpu['id']},{gpu}&ref={cpu['id']}&format=json")
        self.assertEqual(js["ref"], cpu["id"])
        self.assertEqual(self.get(f"/api/export?ids={gpu}&format=xml")[0], 400)
        self.__class__.cpu_id, self.__class__.ref_id = cpu["id"], ref["id"]

    def test_06_rerun_pin_unpin_delete(self):
        st, r, _ = self.post(f"/api/runs/{self.cpu_id}/rerun", {"mode": "ab_auto"})
        self.assertEqual(st, 201, r)
        self.assertEqual(r["derived_from"], self.cpu_id)
        rec = self.wait_state(r["id"], ("done",), 30)
        self.assertEqual(rec["derived_from"], self.cpu_id)
        self.assertEqual(rec["request"]["mode"], "ab_auto")
        self.assertEqual(rec["input_sha256"], self.get(f"/api/runs/{self.cpu_id}")[1]["input_sha256"])
        self.assertIn("calibrated table", rec["ab"]["gpu_verified_reason"])
        self.assertEqual(self.post("/api/runs/nope/rerun", {"mode": "ab_cpu"})[0], 404)
        self.assertEqual(self.post(f"/api/runs/{self.cpu_id}/rerun", {"mode": "reference"})[0], 400)
        # pin / unpin
        st, p, _ = self.post(f"/api/runs/{self.ref_id}/pin")
        self.assertEqual(st, 200, p)
        self.assertEqual(p["reference"], self.ref_id)
        self.assertEqual(runner.Pins().load()[p["input_sha256"]], self.ref_id)
        st, lst, _ = self.get("/api/runs?mode=reference")
        self.assertTrue(next(x for x in lst["runs"] if x["id"] == self.ref_id)["pinned_reference"])
        st, res, _ = self.get(f"/api/compare?ids={self.cpu_id},{self.gpu_id},{self.ref_id}")   # no ref: the pin wins
        self.assertEqual(res["ref"], self.ref_id)
        st, u, _ = self.delete(f"/api/runs/{self.ref_id}/pin")
        self.assertTrue(u["unpinned"])
        self.assertNotIn(self.ref_id, runner.Pins().load().values())
        # delete
        run_dir = Path(rec["run_dir"])
        self.assertTrue(run_dir.exists())
        st, d, _ = self.delete(f"/api/runs/{r['id']}")
        self.assertEqual(st, 200, d)
        self.assertTrue(d["deleted"])
        self.assertFalse(run_dir.exists())
        self.assertEqual(self.get(f"/api/runs/{r['id']}")[0], 404)
        self.assertEqual(self.delete(f"/api/runs/{r['id']}")[0], 404)

    def test_07_cancel_sleeping(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "60"
        st, r, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share"})
        self.assertEqual(st, 201, r)
        rid = r["id"]
        rec = self.wait_state(rid, ("running",), 20)
        pgid = rec["etiquette"]["machine_at_acquisition"]["pgid"]
        t0 = time.time()
        st, c, _ = self.post(f"/api/runs/{rid}/cancel")
        self.assertEqual(st, 200, c)
        self.assertEqual(c["state"], "cancelling")
        rec = self.wait_state(rid, ("cancelled",), 15)
        self.assertLess(time.time() - t0, 15)
        self.assertEqual(runner.pgid_members(pgid), [], "no survivor in the job's pgid")
        d = Path(rec["run_dir"])
        self.assertTrue((d / "out.txt").exists() and (d / "err.txt").exists())
        self.assertIn("iteration #", (d / "out.txt").read_text())
        self.assertEqual(self.post(f"/api/runs/{rid}/cancel")[0], 409)
        st, lg, _ = self.get(f"/api/runs/{rid}/log?since=0")
        self.assertEqual(lg["state"], "cancelled")

    def test_08_timing_lock_wait(self):
        runner.TIMING_LOCK.write_text("other session timing")
        st, r, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share"})
        self.assertEqual(st, 201, r)
        rid = r["id"]
        rec = self.wait_state(rid, ("waiting_timing_lock",), 10)
        st, m, _ = self.get("/api/machine")
        time.sleep(0.5)
        st, lg, _ = self.get(f"/api/runs/{rid}/log?since=0&stream=guard")
        self.assertTrue(any("present" in l[2] for l in lg["lines"]), lg)
        self.assertEqual(lg["state"], "waiting_timing_lock")
        time.sleep(1.0)
        self.assertEqual(self.get(f"/api/runs/{rid}")[1]["state"], "waiting_timing_lock")
        runner.TIMING_LOCK.unlink()
        t0 = time.time()
        rec = self.wait_state(rid, ("done",), 15)
        self.assertLess(time.time() - t0, 12)
        self.assertGreater(rec["etiquette"]["timing_lock_waited_s"], 0)
        st, lg, _ = self.get(f"/api/runs/{rid}/log?since=0&stream=guard")
        self.assertTrue(any("released" in l[2] for l in lg["lines"]))

    def test_09_failure_tail(self):
        os.environ["ABRUN_FAKE_RC"] = "2"
        st, r, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share"})
        self.assertEqual(st, 201, r)
        events = self.sse(f"/api/runs/{r['id']}/events", timeout=20, run_id=r["id"])
        errs = [e for e in events if e[1] == "execution_error"]
        self.assertEqual(len(errs), 1)
        self.assertTrue(1 <= len(errs[0][2]["tail"]) <= 25)
        self.assertIn("rc=2", errs[0][2]["message"])
        self.assertEqual(events[-1][2]["status"], "error")
        rec = self.wait_state(r["id"], ("failed",), 5)
        self.assertEqual(rec["rc"], 2)

    def test_10_duplicates_and_errors(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "20"
        body = {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share", "label": "dup"}
        st, r, _ = self.post("/api/runs", body)
        self.assertEqual(st, 201, r)
        st, r2, _ = self.post("/api/runs", body)
        self.assertEqual(st, 409, r2)
        self.assertEqual(r2["id"], r["id"])
        st, r3, _ = self.post("/api/runs", {**body, "mode": "ab_auto"})      # a different plan is not a duplicate
        self.assertEqual(st, 201, r3)
        self.assertEqual(self.delete(f"/api/runs/{r['id']}")[0], 409)         # non-terminal
        self.assertEqual(self.post(f"/api/runs/{r3['id']}/cancel")[1]["state"], "cancelled")   # still queued: dequeued
        st, c, _ = self.post(f"/api/runs/{r['id']}/cancel")         # possibly before the launch: both paths end cancelled
        self.assertEqual(st, 200, c)
        self.wait_state(r["id"], ("cancelled",), 15)
        self.wait_state(r3["id"], ("cancelled",), 5)
        self.wait_idle(30)
        self.assertEqual(self.get("/api/runs/nope")[0], 404)
        self.assertEqual(self.get("/api/nope")[0], 404)
        self.assertEqual(self.post("/api/health")[0], 405)
        self.assertEqual(self.get("/api/runs/nope/log")[0], 404)
        self.assertEqual(self.get("/api/runs/nope/files/out.txt")[0], 404)
        self.assertEqual(self.post("/api/runs/nope/pin")[0], 404)
        st, e, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "reference"})
        self.assertEqual(st, 400)
        self.assertEqual(e["error"], "invalid plan")
        self.assertTrue(e["violations"])

    def test_11_queue_abset_pause_reorder(self):
        st, p, _ = self.post("/api/queue/pause")
        self.assertTrue(p["paused"])
        st, s, _ = self.post("/api/ab_set", {"preset": "si2", "binary_id": "fake", "ranks": 1, "lock": "share",
                                              "legs": ["reference", "ab_cpu", "ab_gpu_all"], "label": "set"})
        self.assertEqual(st, 201, s)
        self.assertEqual(len(s["ids"]), 3)
        self.assertEqual(s["ref"], s["ids"][0])
        self.assertTrue(s["pinned"])
        self.assertEqual(self.post("/api/ab_set", {"preset": "si2", "binary_id": "fake", "lock": "share", "legs": ["ab_cpu"]})[0], 409)  # duplicate leg
        self.assertEqual(self.post("/api/ab_set", {"preset": "si2", "binary_id": "fake", "legs": ["nope"]})[0], 400)
        st, q, _ = self.get("/api/queue")
        self.assertEqual(q["queued"], s["ids"])
        self.assertTrue(q["paused"])
        self.assertIsNone(q["running"])
        st, rq, _ = self.post("/api/queue/reorder", {"ids": list(reversed(s["ids"]))})
        self.assertEqual(rq["queued"], list(reversed(s["ids"])))
        self.assertEqual(self.post("/api/queue/reorder", {"ids": "x"})[0], 400)
        st, m, _ = self.get("/api/machine")
        self.assertEqual(m["worker"]["queued"], 3)
        self.assertTrue(m["worker"]["paused"])
        st, h, _ = self.get("/api/health")
        self.assertEqual(h["queue"]["queued"], 3)
        st, p, _ = self.post("/api/queue/resume")
        self.assertFalse(p["paused"])
        for rid in s["ids"]:
            rec = self.wait_state(rid, ("done",), 60)
            self.assertEqual(rec["state"], "done", rec["state_why"])
        st, lst, _ = self.get("/api/runs?mode=reference")
        ref_row = next(x for x in lst["runs"] if x["id"] == s["ref"])
        self.assertTrue(ref_row["pinned_reference"])
        st, res, _ = self.get(f"/api/compare?ids={','.join(s['ids'])}")
        self.assertEqual(res["ref"], s["ref"])
        self.assertEqual(res["sections"]["cross_build"], [i for i in s["ids"] if i != s["ref"]])
        st, q, _ = self.get("/api/queue")
        self.assertEqual(q, {"running": None, "queued": [], "paused": False, "paused_reason": None, "held": None, "state": "idle"})

    def test_13_host_origin_content_type_options_body_cap(self):
        port = self.srv.port
        # a DNS-rebound hostname (or any non-loopback Host) is refused on every route
        self.assertEqual(self.get("/api/health", headers={"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.get("/", headers={"Host": f"evil.example:{port}"})[0], 403)
        self.assertEqual(self.get("/api/health", headers={"Host": f"localhost:{port}"})[0], 200)
        self.assertEqual(self.get("/api/health", headers={"Host": f"127.0.0.1:{port}"})[0], 200)
        # a preflight-free form/text POST from a hostile page is not JSON
        st, e, _ = self.req("POST", "/api/plan", headers={"Content-Type": "text/plain"},
                            raw_body=b'{"preset":"si2","binary_id":"fake","mode":"ab_cpu"}')
        self.assertEqual(st, 415, e)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("DELETE", "/api/runs/nope/pin")                     # no Content-Type at all
        r = conn.getresponse(); self.assertEqual(r.status, 415); r.read(); conn.close()
        # a JSON POST with a foreign Origin is refused; the UI's same-origin one is not
        body = {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu"}
        self.assertEqual(self.req("POST", "/api/plan", body, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.req("POST", "/api/plan", body, headers={"Origin": f"http://127.0.0.1:{port}"})[0], 200)
        self.assertEqual(self.req("POST", "/api/plan", body, headers={"Origin": f"http://localhost:{port}"})[0], 200)
        self.assertEqual(self.req("POST", "/api/plan", body)[0], 200)   # no Origin: curl, http.client
        # OPTIONS (a CORS preflight) is refused without any CORS header
        st, e, hdr = self.req("OPTIONS", "/api/runs")
        self.assertEqual(st, 403)
        self.assertFalse(any(k.lower().startswith("access-control") for k in hdr))
        # a body above the cap is refused before it is read
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", "/api/plan")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(server.BODY_MAX_BYTES + 1))
        conn.endheaders()
        r = conn.getresponse(); self.assertEqual(r.status, 413); r.read(); conn.close()
        st, h, _ = self.get("/api/queue")
        self.assertIn("paused_reason", h)
        self.assertIn("held", h)

    def test_14_delete_never_names_the_runs_root(self):
        self.wait_idle()
        planted = runner.RUNS_DIR / "run.json"
        planted.write_text("{}")                                         # the precondition of the old deletion
        try:
            for rid in (".", "..", "..."):
                st, e, _ = self.delete(f"/api/runs/{rid}")
                self.assertEqual(st, 404, (rid, e))
            self.assertEqual(self.get("/api/runs/../log")[0], 404)
            self.assertEqual(self.get("/api/runs/./files/run.json")[0], 404)
            self.assertTrue(runner.RUNS_DIR.is_dir() and planted.exists() and (runner.RUNS_DIR / "queue.json").exists())
        finally:
            planted.unlink()

    def test_12_global_sse_replay(self):
        self.wait_idle()
        events = self.sse("/api/events?since=0", timeout=10, stop_type="heartbeat", max_events=20000)   # the whole replay, then the first ping
        real = [e for e in events if e[1] != "heartbeat"]
        self.assertTrue(real, "the global bus has history from the earlier runs")
        self.assertEqual(events[-1][1], "heartbeat")
        ids = [e[0] for e in real]
        self.assertEqual(ids, sorted(ids))
        first, last = ids[0], ids[-1]
        events2 = self.sse("/api/events", timeout=10, stop_type="heartbeat", headers={"Last-Event-ID": str(first)}, max_events=20000)
        real2 = [e for e in events2 if e[1] != "heartbeat"]
        self.assertTrue(real2)
        self.assertGreater(real2[0][0], first)                   # Last-Event-ID replays only what was missed
        self.assertEqual(real2[-1][0], last)
        events3 = self.sse(f"/api/events?since={last}", timeout=3, stop_type="heartbeat", max_events=5)
        self.assertEqual([e[1] for e in events3], ["heartbeat"])   # nothing newer than the newest seq: only the ping

    def test_99_persisted_queue_restart(self):
        self.wait_idle()
        os.environ["ABRUN_FAKE_SLEEP"] = "30"
        st, a, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_cpu", "lock": "share", "label": "restart-a"})
        st, b, _ = self.post("/api/runs", {"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all", "lock": "share", "label": "restart-b"})
        self.assertEqual((a.get("state"), b.get("state")), ("queued", "queued"))
        self.wait_state(a["id"], ("running",), 20)
        q = json.loads((runner.RUNS_DIR / "queue.json").read_text())
        self.assertEqual((q["running"], q["queued"]), (a["id"], [b["id"]]))
        # graceful stop: the running job is cancelled, the queued one stays queued
        self.srv.stop(cancel_running=True)
        rec_a = json.loads((runner.RUNS_DIR / a["id"] / "run.json").read_text())
        self.assertEqual(rec_a["state"], "cancelled")
        q = json.loads((runner.RUNS_DIR / "queue.json").read_text())
        self.assertEqual((q["running"], q["queued"]), (None, [b["id"]]))
        # simulate a crash instead: run.json left in state running and the persisted queue naming it
        rec_a["state"], rec_a["state_why"] = "running", ""
        (runner.RUNS_DIR / a["id"] / "run.json").write_text(json.dumps(rec_a))
        (runner.RUNS_DIR / "queue.json").write_text(json.dumps({"running": a["id"], "queued": [b["id"]], "paused": False}))
        runner.REGISTRY.remove(a["id"]); runner.REGISTRY.remove(b["id"])
        os.environ.pop("ABRUN_FAKE_SLEEP")
        self.__class__.srv = server.Server(port=0, fake_pw=True).start()
        st, ra, _ = self.get(f"/api/runs/{a['id']}")
        self.assertEqual((ra["state"], ra["state_why"]), ("failed", "server restarted"))
        rb = self.wait_state(b["id"], ("done",), 40)
        self.assertEqual(rb["state"], "done", rb["state_why"])
        self.assertEqual(rb["ab"]["totals"]["gpu_calls"], 1200)
        st, lst, _ = self.get("/api/runs")
        self.assertIn(a["id"], [x["id"] for x in lst["runs"]])
        # a disk-only run (not in memory) still serves its log and a closed event stream
        runner.REGISTRY.remove(b["id"])
        st, lg, _ = self.get(f"/api/runs/{b['id']}/log?since=0")
        self.assertFalse(lg["live"])
        self.assertTrue(any("iteration #" in l[2] for l in lg["lines"]))
        self.assertEqual(lg["suppressed_profile_lines"], 1200 - runner.PROFILE_STREAM_FIRST)
        ev = self.sse(f"/api/runs/{b['id']}/events", timeout=5, run_id=b["id"])
        self.assertEqual([e[1] for e in ev], ["muppet.done"])
        self.assertEqual(ev[0][2]["status"], "success")


class FakeModePathsCase(unittest.TestCase):
    """--fake-pw never writes beside real runs: with the real RUNS_DIR / solver-runs.log in force, Server(fake_pw=True)
    redirects both to a scratch directory and restores them on stop() (runs after ApiCase: it pops the fake ids)."""

    def test_fake_server_redirects_runs_dir_and_run_log(self):
        from abrun import exlib
        saved = (runner.RUNS_DIR, runner.RUN_LOG)
        runner.RUNS_DIR, runner.RUN_LOG = server.REAL_RUNS_DIR, Path(exlib.RUN_LOG)
        try:
            srv = server.Server(port=0, fake_pw=True)                   # not started: no socket, no restore
            try:
                self.assertIsNotNone(srv.fake_dir)
                self.assertNotEqual(Path(runner.RUNS_DIR).resolve(), server.REAL_RUNS_DIR.resolve())
                self.assertNotEqual(Path(runner.RUN_LOG).resolve(), Path(exlib.RUN_LOG).resolve())
                self.assertEqual(srv.runs_dir, srv.fake_dir / "runs")
                self.assertEqual(Path(runner.RUN_LOG), srv.fake_dir / "solver-runs.log")
                self.assertEqual(srv.worker.runs_dir, srv.fake_dir / "runs")
            finally:
                srv.stop(cancel_running=False)
            self.assertEqual(Path(runner.RUNS_DIR), server.REAL_RUNS_DIR)
            self.assertEqual(Path(runner.RUN_LOG), Path(exlib.RUN_LOG))
        finally:
            runner.RUNS_DIR, runner.RUN_LOG = saved
        # already redirected (a test's monkeypatch, or ABRUN_RUNS_DIR): left alone
        scratch = Path(tempfile.mkdtemp(prefix="abrun-fakemode-"))
        runner.RUNS_DIR, runner.RUN_LOG = scratch / "runs", scratch / "solver-runs.log"
        try:
            srv = server.Server(port=0, fake_pw=True)
            try:
                self.assertIsNone(srv.fake_dir)
                self.assertEqual(srv.runs_dir, scratch / "runs")
            finally:
                srv.stop(cancel_running=False)
            self.assertEqual(Path(runner.RUNS_DIR), scratch / "runs")
        finally:
            runner.RUNS_DIR, runner.RUN_LOG = saved
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
