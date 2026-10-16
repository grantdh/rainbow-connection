"""runner.py end to end on the fake pw.x: run_sync, both profile schemas, every mode's verdict, cancel,
timing-lock wait, failure tail, worker persistence/restore, compare ordering (RunnerCase, direct launch);
and the machlock + mpirun process tree on tests/fake_machlock.py + tests/fake_mpirun.py (TreeCase):
descendant walk, own ranks never "foreign", cancel without orphaning ranks that sit in their own process
groups and ignore SIGINT, SIGKILL escalation, a cancel landing before Popen, the shim's exit 75 on a late
timing lock, guard max-wait vs the timing lock, share-during-bench, persist() under contention, restore()
holding on a live orphan, delete_run/new_run root refusals.

Never touches the real locks, logs, machlock or mpirun: TIMING_LOCK, RUN_LOG, RUNS_DIR, launch.MACHLOCK,
launch.MACHLOCK_PYTHON and launch.MPIRUN are monkeypatched in-process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from abrun import SHIM, binaries, compare, launch, runner
from abrun.tests.fake_pw import fake_binary

TESTS = Path(__file__).resolve().parent
FAKE_ENV = ("ABRUN_FAKE_SLEEP", "ABRUN_FAKE_RC", "ABRUN_FAKE_SCHEMA", "ABRUN_FAKE_DYLIB", "ABRUN_FAKE_BUILD",
            "ABRUN_FAKE_IGNORE_SIGINT", "ABRUN_FAKE_IGNORE_SIGTERM", "ABRUN_FAKE_LAUNCHER_DEAF", "MUPPET_GUARD_MAX_WAIT_S")


class RunnerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="abrun-test-"))
        cls.saved = (runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK)
        runner.TIMING_LOCK = cls.tmp / "timing-lock"
        runner.RUN_LOG = cls.tmp / "solver-runs.log"
        runner.RUNS_DIR = cls.tmp / "runs"
        launch.MACHLOCK = Path("/nonexistent/machlock.py")
        (cls.tmp / "libapplebottom.dylib").write_text("fake")
        os.environ["ABRUN_FAKE_DYLIB"] = str(cls.tmp / "libapplebottom.dylib")
        cls.ab = fake_binary("ab")
        cls.ref = fake_binary("reference")
        binaries.register_fake(cls.ab)
        binaries.register_fake(cls.ref)
        os.chmod(cls.ab.path, 0o755)

    @classmethod
    def tearDownClass(cls):
        runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK = cls.saved
        for k in FAKE_ENV:
            os.environ.pop(k, None)
        for b in (cls.ab, cls.ref):                   # the fakes exist only while this class runs (test_binaries walks _FAKES)
            binaries._FAKES.pop(b.id, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tearDown(self):
        for k in ("ABRUN_FAKE_SLEEP", "ABRUN_FAKE_RC", "ABRUN_FAKE_SCHEMA"):
            os.environ.pop(k, None)
        try:
            runner.TIMING_LOCK.unlink()
        except OSError:
            pass

    def run_sync(self, **kw):
        req = launch.Request(**{"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all", "lock": "share", **kw})
        return runner.run_sync(req, log=None)

    # ---- the four modes on the fake, both schemas
    def test_gpu_all_v2(self):
        rec = self.run_sync()
        self.assertEqual(rec["state"], "done", rec["state_why"])
        r, ab, e = rec["result"], rec["ab"], rec["etiquette"]
        self.assertAlmostEqual(r["total_energy_Ry"], -22.83836927)
        self.assertAlmostEqual(r["wall_qe_s"], 121.49)
        self.assertEqual(r["scf_iterations"], 3)
        self.assertTrue(r["job_done"] and r["converged"])
        self.assertEqual(ab["evidence_source"], "profile_json")
        self.assertEqual(ab["schema"], "v2")
        self.assertEqual(ab["totals"]["calls"], 1200)
        self.assertEqual(ab["totals"]["gpu_calls"], 1200)
        self.assertTrue(ab["gpu_verified"], ab["gpu_verified_reason"])
        self.assertEqual(ab["gpu_badge"], "green")
        self.assertTrue(ab["build_matches_dylib"])
        self.assertEqual(ab["lib_loaded"], os.environ["ABRUN_FAKE_DYLIB"])
        self.assertTrue(ab["lib_ok"])
        self.assertEqual(ab["per_call_lines"], 1200)
        self.assertEqual(rec["suppressed_profile_lines"], 1200 - runner.PROFILE_STREAM_FIRST)
        self.assertEqual(e["lock_effective"], "share")
        self.assertEqual(e["timing_grade"], "untimed")
        self.assertEqual(e["gpu_idle"], "unverified")
        m = e["machine_at_acquisition"]
        for k in ("t_acquired", "load1", "power", "timing_lock_present", "foreign_solvers", "pgid"):
            self.assertIn(k, m)
        self.assertFalse(m["timing_lock_present"])
        self.assertEqual(m["nice"], 10)
        d = Path(rec["run_dir"])
        for f in ("input.in", "out.txt", "err.txt", "machine.json", "ab_profile.json", "summary.json", "run.json"):
            self.assertTrue((d / f).exists(), f)
        self.assertFalse((d / "tmp").exists(), "tmp purged")
        self.assertTrue((d / "pseudo" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF").is_symlink())
        self.assertNotIn("AB_MODE", rec["plan"]["env_removed"])
        self.assertEqual(rec["plan"]["launcher"], "direct")
        line = runner.RUN_LOG.read_text().splitlines()[-1]
        self.assertRegex(line, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d rc=0 wall=[\d.]+s load1=[\d.]+ idle_at_start=(True|False) lock=share "
                               r"label=abrun si2 ab_gpu_all fake cwd=" + re.escape(str(d)) + "$")
        summ = json.loads((d / "summary.json").read_text())
        self.assertEqual(summ["gpu_calls"], 1200)
        self.assertEqual(summ["lib_loaded"], ab["lib_loaded"])
        # the per-run bus saw the whole event sequence in order
        run = runner.REGISTRY.get(rec["id"])
        types = [e[2] for e in run.bus.since(0)]
        for t in ("execution_start", "executing", "abrun.state", "muppet.log", "progress", "executed", "muppet.done"):
            self.assertIn(t, types)
        self.assertLess(types.index("execution_start"), types.index("executed"))
        self.assertEqual(types[-1], "muppet.done")
        self.assertTrue(any("suppressed" in e[3].get("text", "") for e in run.bus.since(0) if e[2] == "muppet.log"))

    def test_gpu_all_v1(self):
        os.environ["ABRUN_FAKE_SCHEMA"] = "old"
        rec = self.run_sync()
        self.assertEqual(rec["state"], "done")
        self.assertEqual(rec["ab"]["schema"], "v1")
        self.assertTrue(rec["ab"]["gpu_verified"])
        self.assertIsNone(rec["ab"]["totals"]["fallback_breakdown"])

    def test_cpu_auto_gpu(self):
        rec = self.run_sync(mode="ab_cpu")
        self.assertEqual(rec["ab"]["totals"]["gpu_calls"], 0)
        self.assertTrue(rec["ab"]["gpu_verified"])
        self.assertEqual(rec["ab"]["gpu_badge"], "grey")
        rec = self.run_sync(mode="ab_auto")
        self.assertTrue(rec["ab"]["gpu_verified"])
        self.assertIn("calibrated table", rec["ab"]["gpu_verified_reason"])
        rec = self.run_sync(mode="ab_gpu")
        self.assertEqual(rec["ab"]["totals"]["gpu_calls"], 320)
        self.assertTrue(rec["ab"]["gpu_verified"])
        self.assertEqual(rec["ab"]["min_gpu_dim_effective"], "library default (32)")

    def test_reference(self):
        rec = self.run_sync(binary_id="reference-fake", mode="reference")
        self.assertEqual(rec["state"], "done")
        self.assertEqual(rec["ab"]["evidence_source"], "none")
        self.assertIsNone(rec["ab"]["gpu_verified"])
        self.assertTrue(rec["ab"]["lib_ok"])
        self.assertFalse(any(k.startswith("AB_") for k in rec["plan"]["env_set"]))

    # ---- failure, cancel, timing lock
    def test_failure_tail(self):
        os.environ["ABRUN_FAKE_RC"] = "2"
        rec = self.run_sync(mode="ab_cpu")
        self.assertEqual(rec["state"], "failed")
        self.assertEqual(rec["rc"], 2)
        run = runner.REGISTRY.get(rec["id"])
        errs = [e for e in run.bus.since(0) if e[2] == "execution_error"]
        self.assertEqual(len(errs), 1)
        self.assertTrue(1 <= len(errs[0][3]["tail"]) <= 25)
        self.assertTrue(runner.RUN_LOG.read_text().splitlines()[-1].split()[1] == "rc=2")

    def test_cancel(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "60"
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        t = threading.Thread(target=runner.execute, args=(run, None), daemon=True)
        t.start()
        t0 = time.time()
        while run.state != "running" and time.time() - t0 < 20:
            time.sleep(0.1)
        self.assertEqual(run.state, "running")
        pgid = run.pgid
        self.assertEqual(runner.cancel(run), "cancelling")
        t.join(timeout=15)
        self.assertFalse(t.is_alive())
        self.assertEqual(run.state, "cancelled")
        self.assertEqual(runner.pgid_members(pgid), [], "no survivor in the pgid")
        self.assertTrue((run.run_dir / "out.txt").exists())
        self.assertIn("iteration #", (run.run_dir / "out.txt").read_text())
        types = [e[2] for e in run.bus.since(0)]
        self.assertIn("execution_interrupted", types)
        self.assertEqual(types[-1], "muppet.done")

    def test_timing_lock_wait(self):
        runner.TIMING_LOCK.write_text("other session timing")
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        t = threading.Thread(target=runner.execute, args=(run, None), daemon=True)
        t.start()
        t0 = time.time()
        while run.state != "waiting_timing_lock" and time.time() - t0 < 10:
            time.sleep(0.1)
        self.assertEqual(run.state, "waiting_timing_lock")
        t1 = time.time()
        while not any(l[1] == "guard" and "present" in l[2] for l in run.log.since(0)[0]) and time.time() - t1 < 5:
            time.sleep(0.05)      # the guard line is logged just after the state flips; poll instead of asserting immediately
        self.assertTrue(any(l[1] == "guard" and "present" in l[2] for l in run.log.since(0)[0]))
        time.sleep(1.0)
        self.assertEqual(run.state, "waiting_timing_lock")
        runner.TIMING_LOCK.unlink()
        t.join(timeout=15)
        self.assertEqual(run.state, "done", run.state_why)
        self.assertGreater(run.etiquette["timing_lock_waited_s"], 0)
        self.assertTrue(any("released" in l[2] for l in run.log.since(0)[0] if l[1] == "guard"))

    def test_cancel_while_waiting(self):
        runner.TIMING_LOCK.write_text("x")
        n_log = len(runner.RUN_LOG.read_text().splitlines()) if runner.RUN_LOG.exists() else 0
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        t = threading.Thread(target=runner.execute, args=(run, None), daemon=True)
        t.start()
        time.sleep(0.5)
        self.assertEqual(runner.cancel(run), "cancelling")
        t.join(timeout=10)
        self.assertEqual(run.state, "cancelled")
        self.assertIsNone(run.proc)
        # no process was started: no solver-runs.log line (exlib's wall is a subprocess wall), no runner wall
        self.assertEqual(len(runner.RUN_LOG.read_text().splitlines()) if runner.RUN_LOG.exists() else 0, n_log)
        self.assertIsNone(run.wall_runner_s)
        self.assertGreater(run.wall_total_s, 0)

    def test_wall_excludes_waits(self):
        runner.TIMING_LOCK.write_text("other session timing")
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        t = threading.Thread(target=runner.execute, args=(run, None), daemon=True)
        t.start()
        time.sleep(2.0)
        runner.TIMING_LOCK.unlink()
        t.join(timeout=20)
        self.assertEqual(run.state, "done", run.state_why)
        self.assertGreaterEqual(run.etiquette["timing_lock_waited_s"], 1.5)
        self.assertLess(run.wall_runner_s, run.etiquette["timing_lock_waited_s"], "runner wall counts from acquisition, not from the wait")
        self.assertGreaterEqual(run.wall_total_s, run.etiquette["timing_lock_waited_s"])
        self.assertRegex(runner.RUN_LOG.read_text().splitlines()[-1], rf"rc=0 wall={run.wall_runner_s:.1f}s ")

    def test_delete_and_new_run_refuse_the_runs_root(self):
        root = runner.RUNS_DIR
        root.mkdir(parents=True, exist_ok=True)
        planted = root / "run.json"
        planted.write_text("{}")
        try:
            self.assertFalse(runner.delete_run(".", root))
            self.assertFalse(runner.delete_run("..", root))
            self.assertFalse(runner.delete_run("../" + root.name, root))
            self.assertTrue(root.is_dir() and planted.exists())
            for bad in (root, root.parent):
                with self.assertRaises(runner.PlanError) as cm:
                    runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu"), run_dir=bad)
                self.assertIn("runs root", cm.exception.violations[0])
        finally:
            planted.unlink()

    def test_timing_grade_reads_timing_lock_present(self):
        b = fake_binary("ab")
        b.citable, b.live_library = True, False
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="bench"), binary=b)
        run.request.timed = True
        m = {"load1": 0.5, "power": "AC", "foreign_solvers": [], "timing_lock_present": True}
        run.etiquette["machine_at_acquisition"] = m
        grade, why = runner._timing_grade(run, "bench", True)
        self.assertEqual(grade, "untimed")
        self.assertIn("timing lock present at acquisition", why)
        m["timing_lock_present"] = False
        self.assertEqual(runner._timing_grade(run, "bench", True)[0], "bench")

    def test_shim_exec_failure_is_exit_127(self):
        d = self.tmp / "shim-exec"
        d.mkdir(exist_ok=True)
        p = subprocess.run([sys.executable, str(SHIM), "--run-dir", str(d), "--nice", "0", "--", "/nonexistent/mpirun", "-np", "1"],
                           capture_output=True, text=True, env={**os.environ, "ABRUN_TIMING_LOCK": str(runner.TIMING_LOCK)})
        self.assertEqual(p.returncode, 127, p.stderr)
        self.assertIn("cannot exec /nonexistent/mpirun", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertTrue((d / "machine.json").exists())
        # and the timing lock inside the lock: exit 75 before machine.json
        (d / "machine.json").unlink()
        runner.TIMING_LOCK.write_text("late")
        p = subprocess.run([sys.executable, str(SHIM), "--run-dir", str(d), "--nice", "0", "--", "/bin/true"],
                           capture_output=True, text=True, env={**os.environ, "ABRUN_TIMING_LOCK": str(runner.TIMING_LOCK)})
        self.assertEqual(p.returncode, runner.SHIM_TIMING_LOCK_RC, p.stderr)
        self.assertIn("timing lock present inside the machine lock", p.stderr)
        self.assertFalse((d / "machine.json").exists())

    # ---- plan errors, worker persistence, compare
    def test_plan_error(self):
        with self.assertRaises(runner.PlanError) as cm:
            self.run_sync(mode="reference")
        self.assertTrue(any("apple-bottom-free" in v for v in cm.exception.violations))
        res = runner.plan_only({"preset": "si2", "binary_id": "fake", "mode": "ab_gpu_all"})
        self.assertEqual(res["violations"], [])
        self.assertEqual(res["plan"]["env_set"]["AB_MIN_GPU_DIM"], "0")

    def test_worker_persist_and_restore(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "30"
        w = runner.Worker(runs_dir=runner.RUNS_DIR)
        a = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        b = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_gpu_all", lock="share"))
        w.submit(a); w.submit(b)
        self.assertEqual(w.status()["queued"], [a.id, b.id])
        w.start()
        t0 = time.time()
        while a.state not in ("waiting_guard", "running") and time.time() - t0 < 20:
            time.sleep(0.1)
        q = json.loads(w.queue_path.read_text())
        self.assertEqual(q["running"], a.id)
        self.assertEqual(q["queued"], [b.id])
        w.stop(cancel_running=True)
        t0 = time.time()
        while a.state != "cancelled" and time.time() - t0 < 20:
            time.sleep(0.1)
        self.assertEqual(a.state, "cancelled")
        # simulate a crash: a run.json left in state running, and the persisted queue naming it
        a.state = "running"; a.save()
        runner.REGISTRY.remove(a.id); runner.REGISTRY.remove(b.id)
        w.queue_path.write_text(json.dumps({"running": a.id, "queued": [b.id], "paused": False}))
        os.environ.pop("ABRUN_FAKE_SLEEP")
        w2 = runner.Worker(runs_dir=runner.RUNS_DIR)
        res = w2.restore()
        self.assertEqual(res["failed"], a.id)
        self.assertEqual(res["requeued"], [b.id])
        self.assertEqual(json.loads((a.run_dir / "run.json").read_text())["state_why"], "server restarted")
        self.assertEqual(w2.status()["queued"], [b.id])
        w2.start()
        b2 = runner.REGISTRY.get(b.id)
        t0 = time.time()
        while b2.state != "done" and time.time() - t0 < 30:
            time.sleep(0.2)
        self.assertEqual(b2.state, "done", b2.state_why)
        w2.stop(cancel_running=False)
        self.assertTrue(runner.delete_run(a.id, runner.RUNS_DIR))
        self.assertFalse(a.run_dir.exists())

    def test_compare(self):
        ref = self.run_sync(binary_id="reference-fake", mode="reference")
        cpu = self.run_sync(mode="ab_cpu")
        gpu = self.run_sync(mode="ab_gpu_all")
        res = compare.compare([cpu, gpu, ref], ref_id=ref["id"])
        self.assertEqual(res["ref"], ref["id"])
        self.assertEqual([r["id"] for r in res["rows"]], [ref["id"], cpu["id"], gpu["id"]])   # same-binary (vs each other) after ref
        for r in res["rows"][1:]:
            self.assertEqual(r["dE_vs_ref_Ry"], 0.0)
            self.assertFalse(r["same_pw_sha"])
            self.assertEqual(r["tolerance_Ry"], 1e-5)
            self.assertIn("not a BLAS A/B", r["tolerance_note"])
            self.assertFalse(r["wall_comparable"])
            self.assertTrue(r["wall_note"].startswith("untimed"))
        res2 = compare.compare([cpu, gpu, ref], ref_id=cpu["id"])
        self.assertEqual([r["id"] for r in res2["rows"]], [cpu["id"], gpu["id"], ref["id"]])
        self.assertTrue(res2["rows"][1]["same_pw_sha"])
        self.assertEqual(res2["rows"][1]["tolerance_Ry"], 1e-8)
        self.assertTrue(res["markdown"].startswith(compare.MD_HEADER))
        self.assertIn("1200/1200", res["markdown"])
        self.assertEqual(compare.select_reference([cpu, gpu, ref]), ref["id"])
        pins = runner.Pins(self.tmp / "pins.json")
        pins.pin(cpu["input_sha256"], cpu["id"])
        self.assertEqual(compare.select_reference([cpu, gpu, ref], pins=pins.load()), cpu["id"])
        self.assertTrue(pins.unpin(cpu["id"]))


class TreeCase(unittest.TestCase):
    """The machlock + mpirun process tree, on stand-ins: launch.MACHLOCK -> tests/fake_machlock.py (runs the job
    with subprocess.call like the real wrapper), launch.MPIRUN -> tests/fake_mpirun.py (puts each rank in its
    own process group like PRRTE and aborts ~0.5 s after SIGINT), binary 'fake-mpi' (links.libmpi set).
    Escalation deadlines are shortened to 2 s / 4 s; the contention sampler to 1 s; the hold poll to 1 s."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="abrun-tree-"))
        cls.saved = (runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK, launch.MACHLOCK_PYTHON, launch.MPIRUN,
                     runner.CANCEL_SIGTERM_AFTER_S, runner.CANCEL_SIGKILL_AFTER_S, runner.CONTENTION_POLL_S, runner.HOLD_POLL_S)
        runner.TIMING_LOCK = cls.tmp / "timing-lock"
        runner.RUN_LOG = cls.tmp / "solver-runs.log"
        runner.RUNS_DIR = cls.tmp / "runs"
        launch.MACHLOCK = TESTS / "fake_machlock.py"
        launch.MACHLOCK_PYTHON = Path(sys.executable)
        launch.MPIRUN = TESTS / "fake_mpirun.py"
        runner.CANCEL_SIGTERM_AFTER_S, runner.CANCEL_SIGKILL_AFTER_S = 2.0, 4.0
        runner.CONTENTION_POLL_S, runner.HOLD_POLL_S = 1.0, 1.0
        (cls.tmp / "libapplebottom.dylib").write_text("fake")
        os.environ["ABRUN_FAKE_DYLIB"] = str(cls.tmp / "libapplebottom.dylib")
        cls.mpi, cls.ab = fake_binary("mpi"), fake_binary("ab")
        binaries.register_fake(cls.mpi)
        binaries.register_fake(cls.ab)
        for p in (cls.ab.path, launch.MPIRUN, launch.MACHLOCK):
            os.chmod(p, 0o755)

    @classmethod
    def tearDownClass(cls):
        (runner.TIMING_LOCK, runner.RUN_LOG, runner.RUNS_DIR, launch.MACHLOCK, launch.MACHLOCK_PYTHON, launch.MPIRUN,
         runner.CANCEL_SIGTERM_AFTER_S, runner.CANCEL_SIGKILL_AFTER_S, runner.CONTENTION_POLL_S, runner.HOLD_POLL_S) = cls.saved
        for k in FAKE_ENV:
            os.environ.pop(k, None)
        for b in (cls.mpi, cls.ab):
            binaries._FAKES.pop(b.id, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tearDown(self):
        for k in FAKE_ENV:
            if k != "ABRUN_FAKE_DYLIB":
                os.environ.pop(k, None)
        try:
            runner.TIMING_LOCK.unlink()
        except OSError:
            pass

    # ---- helpers
    def start(self, binary_id="fake-mpi", **kw):
        req = launch.Request(**{"preset": "si2", "binary_id": binary_id, "mode": "ab_cpu", "lock": "build", "ranks": 2, **kw})
        run = runner.new_run(req)
        t = threading.Thread(target=runner.execute, args=(run, None), daemon=True)
        t.start()
        return run, t

    def wait_state(self, run, states, timeout=20.0):
        t0 = time.time()
        while run.state not in states and time.time() - t0 < timeout:
            time.sleep(0.1)
        self.assertIn(run.state, states, f"{run.state}: {run.state_why}")

    def wait_ranks(self, run, n=2, timeout=15.0):
        """Wait until n descendants of the launched pid sit in their own process groups (the ranks)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            table = runner.ps_table()
            desc = runner.descendants(run.proc.pid, table)
            ranks = [p for p in desc if table[p][1] == p]
            if len(ranks) >= n:
                return table, desc, sorted(ranks)
            time.sleep(0.1)
        self.fail(f"ranks never appeared under {run.proc.pid}")

    def survivors(self, pids):
        table = runner.ps_table()
        return sorted(p for p in pids if p in table)

    def lines(self, run, stream):
        return [l[2] for l in run.log.since(0)[0] if l[1] == stream]

    # ---- the blocker: cancel on the wrapper path must not orphan the ranks
    def test_wrapper_cancel_leaves_no_orphan(self):
        os.environ["ABRUN_FAKE_SLEEP"], os.environ["ABRUN_FAKE_IGNORE_SIGINT"] = "60", "1"
        run, t = self.start()
        self.wait_state(run, ("running",))
        self.assertEqual(run.etiquette["lock_effective"], "machlock-build")
        self.assertTrue(run.process["wrapper"])
        self.assertEqual(run.plan.launcher, "mpirun")
        table, desc, ranks = self.wait_ranks(run)
        own = {run.proc.pid} | desc
        self.assertGreaterEqual(len(desc), 3, "launcher + 2 ranks")
        self.assertFalse(set(ranks) & set(runner.pgid_members(run.pgid)), "the ranks are outside the launched pgid: killpg cannot see them")
        t0 = time.time()
        self.assertEqual(runner.cancel(run), "cancelling")
        t.join(timeout=25)
        self.assertFalse(t.is_alive())
        self.assertLess(time.time() - t0, 10, "SIGINT to the launcher aborted the ranks; no escalation needed")
        self.assertEqual(run.state, "cancelled", run.state_why)
        time.sleep(0.3)
        self.assertEqual(self.survivors(own), [], "nothing of the tree survives the cancel")
        self.assertTrue(set(ranks) <= set(run.process["descendants_at_cancel"]))
        self.assertTrue(run.process["exited"])
        err = self.lines(run, "err")
        self.assertTrue(any("fake_mpirun" in l and "abort on signal 2" in l for l in err), err[-5:])
        self.assertTrue(any(l.startswith("fake_machlock: release build rc=") for l in err), "the wrapper released its lock normally")
        self.assertFalse(any("outlived the launcher" in l for l in self.lines(run, "guard")), "no orphan to sweep")
        self.assertRegex(runner.RUN_LOG.read_text().splitlines()[-1], r" rc=-?\d+ wall=[\d.]+s .* lock=machlock label=abrun si2 ab_cpu fake-mpi ")

    def test_group_sigint_orphans_the_ranks_and_the_exit_sweep_catches_them(self):
        """The pre-fix behaviour, as a control: a SIGINT to the wrapper's process group makes machlock's
        subprocess.call SIGKILL the launcher before it can abort the ranks (own pgids, ignoring SIGINT); the
        runner's exit sweep of the descendant snapshot is what cleans up."""
        os.environ["ABRUN_FAKE_SLEEP"], os.environ["ABRUN_FAKE_IGNORE_SIGINT"] = "60", "1"
        run, t = self.start()
        self.wait_state(run, ("running",))
        table, desc, ranks = self.wait_ranks(run)
        time.sleep(1.5)                                         # one contention sample: the snapshot holds the ranks
        self.assertTrue(set(ranks) <= set(run.process["descendants"]))
        os.killpg(run.pgid, signal.SIGINT)
        t.join(timeout=25)
        self.assertFalse(t.is_alive())
        self.assertEqual(run.state, "failed", run.state_why)
        self.assertEqual(self.survivors(ranks), [])
        self.assertTrue(any("outlived the launcher" in l for l in self.lines(run, "guard")), self.lines(run, "guard"))

    def test_wrapper_cancel_escalates_to_sigkill(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "60"
        os.environ["ABRUN_FAKE_IGNORE_SIGINT"] = os.environ["ABRUN_FAKE_IGNORE_SIGTERM"] = os.environ["ABRUN_FAKE_LAUNCHER_DEAF"] = "1"
        run, t = self.start()
        self.wait_state(run, ("running",))
        table, desc, ranks = self.wait_ranks(run)
        own = {run.proc.pid} | desc
        t0 = time.time()
        runner.cancel(run)
        t.join(timeout=30)
        self.assertFalse(t.is_alive())
        self.assertGreaterEqual(time.time() - t0, runner.CANCEL_SIGKILL_AFTER_S - 0.5, "SIGINT and SIGTERM were ignored")
        self.assertEqual(run.state, "cancelled", run.state_why)
        time.sleep(0.3)
        self.assertEqual(self.survivors(own), [])

    def test_direct_mpirun_cancel(self):
        os.environ["ABRUN_FAKE_SLEEP"], os.environ["ABRUN_FAKE_IGNORE_SIGINT"] = "60", "1"
        run, t = self.start(lock="share")
        self.wait_state(run, ("running",))
        self.assertFalse(run.process["wrapper"])
        table, desc, ranks = self.wait_ranks(run)
        own = {run.proc.pid} | desc
        runner.cancel(run)
        t.join(timeout=25)
        self.assertEqual(run.state, "cancelled", run.state_why)
        time.sleep(0.3)
        self.assertEqual(self.survivors(own), [])

    # ---- own ranks are never foreign solvers
    def test_own_tree_excluded_from_foreign_solvers(self):
        # a synthetic ps table shaped like the live chain: machlock (session leader 500) -> prterun (same pgid)
        # -> two pw.x ranks with pgid == pid (PRRTE setpgid); 900 is another session's pw.x
        table = {500: (1, 500, "/opt/homebrew/bin/python3 machlock.py bench -- ..."),
                 501: (500, 500, "prterun -np 2 -x AB_MODE=cpu /x/pw.x -in input.in"),
                 502: (501, 502, "/x/pw.x -in input.in"), 503: (501, 503, "/x/pw.x -in input.in"),
                 900: (1, 900, "/y/pw.x -in other.in"), 901: (1, 901, "/bin/zsh")}
        desc = runner.descendants(500, table)
        self.assertEqual(desc, {501, 502, 503})
        own = {500} | desc
        self.assertEqual(runner.foreign_solvers(own, table=table), [[900, "/y/pw.x -in other.in"]])
        pgid_only = {pid for pid, (ppid, pgid, cmd) in table.items() if pgid == 500}
        self.assertEqual([f[0] for f in runner.foreign_solvers(pgid_only, table=table)], [502, 503, 900],
                         "a pgid-based exclusion would count the run's own ranks as foreign")
        proc = {"pid": 500, "pgid": 500, "descendants": [501, 502, 503], "exited": False}
        self.assertEqual(runner.live_pids(proc, "/runs/x", table), [500, 501, 502, 503])
        self.assertEqual(runner.live_pids(proc, "/runs/x", {502: (1, 502, "/bin/sleep 5")}), [], "a reused pid is not ours")
        self.assertEqual(runner.live_pids({**proc, "exited": True}, "/runs/x", table), [])
        # and live: the 1 s sampler never records our own launcher/ranks as contention
        os.environ["ABRUN_FAKE_SLEEP"] = "4"
        run, t = self.start(lock="share")
        self.wait_state(run, ("running",))
        table, desc, ranks = self.wait_ranks(run)
        t.join(timeout=30)
        self.assertEqual(run.state, "done", run.state_why)
        self.assertGreaterEqual(len(run.etiquette["load1_samples"]), 1)
        own = {run.proc.pid} | desc
        for c in run.etiquette["contention"]:
            self.assertFalse(own & {f[0] for f in c["foreign_solvers"]}, c)
        self.assertTrue(set(ranks) <= set(run.process["descendants"]))

    # ---- a cancel between the last check and Popen
    def test_cancel_between_check_and_popen(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "60"
        orig, fired = subprocess.Popen, []

        def hooked(*a, **k):
            if k.get("start_new_session") and not fired:
                fired.append(1)
                runner.cancel(run)
            return orig(*a, **k)

        run = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        runner.subprocess.Popen = hooked
        try:
            t0 = time.time()
            rec = runner.execute(run, None)
        finally:
            runner.subprocess.Popen = orig
        self.assertTrue(fired)
        self.assertEqual(rec["state"], "cancelled", rec["state_why"])
        self.assertLess(time.time() - t0, 15, "the poll loop stopped the solver; it did not run its 60 s")
        self.assertEqual(self.survivors([run.proc.pid]), [])

    # ---- the timing lock inside the machine lock: shim exit 75 -> wait again -> relaunch
    def test_shim_timing_lock_relaunch(self):
        orig, fired = subprocess.Popen, []

        def hooked(*a, **k):
            if k.get("start_new_session") and not fired:
                fired.append(1)
                runner.TIMING_LOCK.write_text("late lock")
                threading.Timer(1.5, runner.TIMING_LOCK.unlink).start()
            return orig(*a, **k)

        n_log = len(runner.RUN_LOG.read_text().splitlines()) if runner.RUN_LOG.exists() else 0
        run = runner.new_run(launch.Request(preset="si2", binary_id="fake-mpi", mode="ab_cpu", lock="build", ranks=1))
        runner.subprocess.Popen = hooked
        try:
            rec = runner.execute(run, None)
        finally:
            runner.subprocess.Popen = orig
        self.assertEqual(rec["state"], "done", rec["state_why"])
        self.assertEqual(rec["etiquette"]["timing_lock_relaunches"], 1)
        self.assertTrue(any("shim found" in l for l in self.lines(run, "guard")), self.lines(run, "guard"))
        self.assertTrue(any("timing lock present inside the machine lock" in l for l in self.lines(run, "err")))
        self.assertTrue(any("timing lock released" in l for l in self.lines(run, "guard")))
        self.assertLess(rec["wall_runner_s"], rec["wall_total_s"])
        self.assertEqual(len(runner.RUN_LOG.read_text().splitlines()), n_log + 1, "one log line for one solver run")
        self.assertFalse(rec["etiquette"]["machine_at_acquisition"]["timing_lock_present"])

    def test_guard_max_wait_never_discharges_the_timing_lock(self):
        os.environ["MUPPET_GUARD_MAX_WAIT_S"] = "1"
        saved_machlock, launch.MACHLOCK = launch.MACHLOCK, Path("/nonexistent/machlock.py")      # guard mode
        orig = runner.wait_for_timing_lock

        def late(run, log=None):
            ok = orig(run, log)
            runner.TIMING_LOCK.write_text("appeared after step 1")
            return ok

        runner.wait_for_timing_lock = late
        try:
            run, t = self.start(binary_id="fake", lock="build", ranks=1)
            self.wait_state(run, ("waiting_guard",), 10)
            time.sleep(3.0)
            self.assertEqual(run.state, "waiting_guard", "max-wait (1 s) must not discharge the timing lock")
            self.assertTrue(any("max-wait never discharges" in l for l in self.lines(run, "guard")))
            self.assertFalse(any("-> proceeding" in l for l in self.lines(run, "guard")))
            runner.TIMING_LOCK.unlink()
            t.join(timeout=30)
            self.assertEqual(run.state, "done", run.state_why)
        finally:
            runner.wait_for_timing_lock = orig
            launch.MACHLOCK = saved_machlock

    # ---- explicit share while another session holds bench
    def test_share_during_bench_waits_as_build_unless_allowed(self):
        saved = runner.machlock_bench_held
        runner.machlock_bench_held = lambda machlock=None: True
        try:
            run, t = self.start(lock="share", ranks=1)
            t.join(timeout=30)
            self.assertEqual(run.state, "done", run.state_why)
            self.assertEqual(run.etiquette["lock_effective"], "machlock-build")
            self.assertTrue(run.etiquette["lock_share_promoted_to_build"])
            self.assertTrue(run.etiquette["bench_held_at_dequeue"])
            self.assertTrue(any("waiting as machlock build" in l for l in self.lines(run, "guard")))
            self.assertTrue(any(l.startswith("fake_machlock: got build") for l in self.lines(run, "err")))
            run2 = runner.new_run({"preset": "si2", "binary_id": "fake-mpi", "mode": "ab_cpu", "lock": "share", "ranks": 1,
                                   "allow_share_during_bench": True})
            self.assertTrue(run2.allow_share_during_bench)
            rec = runner.execute(run2, None)
            self.assertEqual(rec["state"], "done", rec["state_why"])
            self.assertEqual(rec["etiquette"]["lock_effective"], "share")
            self.assertFalse(rec["etiquette"]["lock_share_promoted_to_build"])
            self.assertEqual(rec["plan"]["wrapper_argv"], [])
            self.assertTrue(rec["allow_share_during_bench"])
            self.assertTrue(any("launching anyway (allow_share_during_bench=true)" in l for l in self.lines(run2, "guard")))
            self.assertTrue(runner.run_from_record(rec).allow_share_during_bench)
        finally:
            runner.machlock_bench_held = saved

    # ---- persistence under contention; restore holding on a live orphan
    def test_persist_concurrent(self):
        w = runner.Worker(runs_dir=runner.RUNS_DIR)
        a = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        w.submit(a)
        errors = []

        def hammer():
            try:
                for _ in range(150):
                    w.persist()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        q = json.loads(w.queue_path.read_text())
        self.assertEqual(q["queued"], [a.id])
        self.assertEqual([p.name for p in runner.RUNS_DIR.glob(".queue.json.*")], [], "no temp file left behind")
        w.dequeue_cancel(a.id)
        self.assertEqual(json.loads(w.queue_path.read_text())["queued"], [])

    def test_restore_holds_while_the_orphan_lives(self):
        os.environ["ABRUN_FAKE_SLEEP"] = "60"
        orphan = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_cpu", lock="share"))
        # the solver outlives its server: start it through an intermediate that exits at once, so launchd reaps it
        spawn = ("import subprocess, sys; p = subprocess.Popen(sys.argv[1:], start_new_session=True, stdin=subprocess.DEVNULL, "
                 "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)")
        out = subprocess.run([sys.executable, "-c", spawn, sys.executable, self.ab.path, "-in", "input.in"], cwd=str(orphan.run_dir),
                             capture_output=True, text=True, env={**os.environ, "AB_MODE": "cpu"}).stdout
        pid = int(out.strip())
        os.environ.pop("ABRUN_FAKE_SLEEP")
        orphan.state = "running"
        orphan.process.update({"pid": pid, "pgid": pid, "descendants": [], "wrapper": False, "exited": False})
        orphan.save()
        b = runner.new_run(launch.Request(preset="si2", binary_id="fake", mode="ab_gpu_all", lock="share"))
        (runner.RUNS_DIR / "queue.json").write_text(json.dumps({"running": orphan.id, "queued": [b.id], "paused": False}))
        runner.REGISTRY.remove(orphan.id); runner.REGISTRY.remove(b.id)
        w = runner.Worker(runs_dir=runner.RUNS_DIR)
        try:
            res = w.restore()
            self.assertEqual(res["held"], orphan.id)
            self.assertIsNone(res["failed"])
            self.assertEqual(res["requeued"], [b.id])
            st = w.status()
            self.assertEqual((st["state"], st["held"], st["paused"]), ("held", orphan.id, True))
            self.assertIn("orphaned solver", st["paused_reason"])
            held = runner.REGISTRY.get(orphan.id)
            self.assertEqual(held.state, "running")
            self.assertIn("still alive", held.state_why)
            self.assertEqual(json.loads(w.queue_path.read_text())["running"], orphan.id)
            w.start()
            time.sleep(2.5)
            b2 = runner.REGISTRY.get(b.id)
            self.assertEqual(b2.state, "queued", "nothing starts beside a live orphan")
            self.assertEqual(self.survivors([pid]), [pid])
            self.assertTrue(w.resume(), "resume cannot lift a hold")
            self.assertEqual(runner.cancel(held), "cancelling")
            t0 = time.time()
            while b2.state != "done" and time.time() - t0 < 30:
                time.sleep(0.2)
            self.assertEqual(held.state, "cancelled", held.state_why)
            self.assertEqual(b2.state, "done", b2.state_why)
            self.assertEqual(self.survivors([pid]), [])
            st = w.status()
            self.assertEqual((st["state"], st["held"], st["paused"]), ("idle", None, False))
            self.assertIsNone(json.loads(w.queue_path.read_text())["running"])
        finally:
            w.stop(cancel_running=False)
            if pid in runner.ps_table():
                os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
