"""Golden argv/env matrix and the violation matrix for launch.plan()/validate() (pure, no process), the mode
label wording, and compare.py's pair classes / notes / binary_short on synthetic run records."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from abrun import AB_FROZEN_ROOT, INTERPRETER, MACHLOCK_PYTHON, SHIM, compare, launch
from abrun.binaries import Binary

CWD = Path("/private/tmp/abrun-test-cwd")


def frozen(bid="33fbaa8c8944", **kw):
    d = dict(id=bid, group="frozen", path=str(AB_FROZEN_ROOT / bid / "pw.x"), sha256="2f0f029a56073a88", sha_verified=True,
             qe_version="7.5", links={"applebottom": True, "openblas": True, "accelerate": False, "fftw3_omp": True,
                                      "libmpi": "/opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib"},
             ab_symbols={"U": 2, "T": 0}, ab_linkage="dynamic", rpaths=["@executable_path"],
             dylib_path=str(AB_FROZEN_ROOT / bid / "libapplebottom.dylib"), dylib_sha256="33fbaa", ab_build_info="apple-bottom 1.4.0 (c5a1fa6a8166)",
             features=["AB_DISPATCH_TABLE", "AB_MIN_GPU_DIM", "AB_MODE", "AB_PROFILE", "AB_PROFILE_JSON", "AB_PROFILE_THRESHOLD_NS", "AB_REPORT"],
             profile_schema="v2", omp1_only=True, cls="citable", citable=True, launchable=True)
    d.update(kw)
    return Binary(**d)


REF = Binary(id="ref", group="reference", path="/usr/local/bin/pw.x", sha256="e70ae3ba", qe_version="7.4.1",
             links={"applebottom": False, "openblas": True, "accelerate": False, "fftw3_omp": False,
                    "libmpi": "/opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib"},
             ab_symbols={"U": 0, "T": 0}, ab_linkage="none", cls="reference", citable=True, launchable=True)
FZ = frozen()
FZ3 = frozen("3ae144a9cadd", features=FZ.features + ["AB_TRACE", "AB_TRACE_HASH", "AB_TEST_FAIL_STAGE"], cls="experiments", citable=False)
E56 = frozen("e56d1b46c753", sha256="55213db2", features=["AB_MIN_GPU_DIM", "AB_MODE", "AB_PROFILE", "AB_PROFILE_JSON"],
             ab_build_info=None, profile_schema="v1", omp1_only=False, cls="parity-only", citable=False)
DEV = frozen("builds:mpi-gpu-ab", group="dev", path=str(Path.home() / "qe-test/builds/mpi-gpu-ab/bin/pw.x"), sha_verified=None,
             dylib_path=str(Path.home() / "Dev/Claude/apple-bottom/build/libapplebottom.dylib"), live_library=True,
             omp1_only=False, cls="dev", citable=False)
SEEDED_ENV = {"PATH": "/usr/bin", "AB_MODE": "gpu", "AB_TEST_FAIL_STAGE": "alloc", "DYLD_PRINT_LIBRARIES": "1", "HOME": "/x"}


class TestGolden(unittest.TestCase):
    def setUp(self):
        self._old = launch.MACHLOCK
        launch.MACHLOCK = Path(__file__)          # an existing file so build/bench produce a wrapper
        os.environ.pop("MUPPET_NO_MACHLOCK", None)

    def tearDown(self):
        launch.MACHLOCK = self._old

    def x_pairs(self, argv):
        return [argv[i + 1] for i, a in enumerate(argv) if a == "-x"]

    def test_env_matrix(self):
        expect_ab = {"reference": None, "ab_cpu": "cpu", "ab_auto": "auto", "ab_gpu": "gpu", "ab_gpu_all": "gpu"}
        for mode, ab in expect_ab.items():
            for ranks, nk in ((1, 1), (8, 4)):
                for lock in launch.LOCKS:
                    b = REF if mode == "reference" else FZ
                    req = launch.Request(preset="si2", binary_id=b.id, mode=mode, ranks=ranks, nk=nk, lock=lock)
                    with self.subTest(mode=mode, ranks=ranks, lock=lock):
                        self.assertEqual(launch.validate(req, b, CWD), [])
                        p = launch.plan(req, b, CWD, environ=SEEDED_ENV)
                        env = p.env_set
                        self.assertEqual(p.env_removed, ["AB_MODE", "AB_TEST_FAIL_STAGE", "DYLD_PRINT_LIBRARIES"])
                        self.assertEqual(env.get("AB_MODE"), ab)
                        self.assertNotIn("off", env.values())
                        self.assertEqual(env["OMP_NUM_THREADS"], "1")
                        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
                        self.assertEqual(env["VECLIB_MAXIMUM_THREADS"], "1")
                        self.assertEqual(env["DYLD_PRINT_LIBRARIES"], "1")
                        self.assertEqual("AB_MIN_GPU_DIM" in env, mode == "ab_gpu_all")
                        if mode == "ab_gpu_all":
                            self.assertEqual(env["AB_MIN_GPU_DIM"], "0")
                        if ab:
                            self.assertEqual(env["AB_REPORT"], "1")
                            self.assertEqual("AB_PROFILE_JSON" in env, ranks == 1)
                            self.assertEqual("AB_PROFILE" in env, ranks == 1)
                            if ranks == 1:
                                self.assertEqual(env["AB_PROFILE_JSON"], str(CWD / "ab_profile.json"))
                        else:
                            self.assertFalse(any(k.startswith("AB_") for k in env))
                        # every exported variable is an -x NAME=value, in order
                        self.assertEqual(self.x_pairs(p.solver_argv), [f"{k}={v}" for k, v in env.items()])
                        self.assertEqual(p.solver_argv[:3], ["/opt/homebrew/bin/mpirun", "-np", str(ranks)])
                        tail = [b.path, "-in", "input.in"] + (["-nk", str(nk)] if nk > 1 else [])
                        self.assertEqual(p.solver_argv[-len(tail):], tail)
                        self.assertEqual(p.shim_argv, [INTERPRETER, str(SHIM), "--run-dir", str(CWD), "--nice", "10", "--"])
                        eff = launch.resolve_lock(lock, False, bench_held=False)
                        self.assertEqual(p.lock_would_resolve_to, eff)
                        if eff in ("build", "bench"):
                            self.assertEqual(p.wrapper_argv, [str(MACHLOCK_PYTHON), str(launch.MACHLOCK), eff, "--"])
                        else:
                            self.assertEqual(p.wrapper_argv, [])
                        self.assertNotIn("/usr/bin/nice", p.argv)
                        self.assertNotIn("nice", [os.path.basename(a) for a in p.argv])
                        self.assertEqual(p.argv, p.wrapper_argv + p.shim_argv + p.solver_argv)
                        self.assertTrue(p.shell.startswith(f"cd {CWD} && env "))
                        for tok in p.argv:
                            self.assertIn(tok.split("=")[0] if " " not in tok else tok, p.shell)
                        self.assertFalse(p.timing_grade_possible)

    def test_golden_ab_gpu_all(self):
        req = launch.Request(preset="si2", binary_id="33fbaa8c8944", mode="ab_gpu_all")
        p = launch.plan(req, FZ, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.solver_argv, [
            "/opt/homebrew/bin/mpirun", "-np", "1", "-x", "OMP_NUM_THREADS=1", "-x", "OPENBLAS_NUM_THREADS=1",
            "-x", "VECLIB_MAXIMUM_THREADS=1", "-x", "AB_MODE=gpu", "-x", "AB_MIN_GPU_DIM=0", "-x", "AB_REPORT=1",
            "-x", "AB_PROFILE=1", "-x", f"AB_PROFILE_JSON={CWD}/ab_profile.json", "-x", "DYLD_PRINT_LIBRARIES=1",
            str(AB_FROZEN_ROOT / "33fbaa8c8944/pw.x"), "-in", "input.in"])
        self.assertEqual(p.min_gpu_dim_effective, 0)
        p = launch.plan(launch.Request(mode="ab_gpu", binary_id="33fbaa8c8944"), FZ, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.min_gpu_dim_effective, "library default (32)")
        self.assertNotIn("AB_MIN_GPU_DIM", p.env_set)
        p = launch.plan(launch.Request(mode="ab_gpu", binary_id="33fbaa8c8944", min_gpu_dim=64), FZ, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.env_set["AB_MIN_GPU_DIM"], "64")
        p = launch.plan(launch.Request(mode="reference", binary_id="ref"), REF, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.solver_argv, ["/opt/homebrew/bin/mpirun", "-np", "1", "-x", "OMP_NUM_THREADS=1", "-x", "OPENBLAS_NUM_THREADS=1",
                                         "-x", "VECLIB_MAXIMUM_THREADS=1", "-x", "DYLD_PRINT_LIBRARIES=1", "/usr/local/bin/pw.x", "-in", "input.in"])

    def test_timed_and_lock_resolution(self):
        self.assertEqual(launch.resolve_lock("auto", True, False), "bench")
        self.assertEqual(launch.resolve_lock("auto", False, True), "build")
        self.assertEqual(launch.resolve_lock("auto", False, False), "share")
        self.assertEqual(launch.resolve_lock("share", True, True), "share")
        req = launch.Request(mode="ab_cpu", binary_id="33fbaa8c8944", timed=True)
        self.assertEqual(launch.validate(req, FZ, CWD), [])
        p = launch.plan(req, FZ, CWD, environ=SEEDED_ENV)
        self.assertTrue(p.timing_grade_possible)
        self.assertEqual(p.wrapper_argv[2], "bench")
        p = launch.plan(launch.Request(mode="ab_cpu", binary_id="33fbaa8c8944"), FZ, CWD, environ=SEEDED_ENV, bench_held=True)
        self.assertEqual(p.lock_would_resolve_to, "build")
        self.assertTrue(any("bench held" in n for n in p.notes))

    def test_omp_and_profile_rules(self):
        p = launch.plan(launch.Request(mode="ab_cpu", binary_id="e56d1b46c753", omp=4), E56, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.env_set["OMP_NUM_THREADS"], "1")
        self.assertTrue(any("forced to 1" in n for n in p.notes))
        p = launch.plan(launch.Request(mode="ab_cpu", binary_id="e56d1b46c753", omp=4, allow_omp_gt1=True), E56, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.env_set["OMP_NUM_THREADS"], "4")
        self.assertNotIn("AB_REPORT", p.env_set)                       # e56d has no AB_REPORT feature
        self.assertTrue(any("AB_REPORT not exported" in n for n in p.notes))
        p = launch.plan(launch.Request(mode="reference", binary_id="ref", omp=4), REF, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.env_set["OMP_NUM_THREADS"], "4")
        p = launch.plan(launch.Request(mode="ab_cpu", binary_id="33fbaa8c8944", ranks=8, nk=4, force_profile_json=True), FZ, CWD, environ=SEEDED_ENV)
        self.assertIn("AB_PROFILE_JSON", p.env_set)
        self.assertTrue(any("WARNING" in n for n in p.notes))
        p = launch.plan(launch.Request(mode="ab_gpu", binary_id="3ae144a9cadd", trace=True), FZ3, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.env_set["AB_TRACE"], str(CWD / "ab_trace"))
        self.assertNotIn("AB_TEST_FAIL_STAGE", p.env_set)

    def test_direct_launcher(self):
        b = frozen(links={**FZ.links, "libmpi": None})
        p = launch.plan(launch.Request(mode="ab_cpu", binary_id=b.id, ranks=4), b, CWD, environ=SEEDED_ENV)
        self.assertEqual(p.launcher, "direct")
        self.assertEqual(p.solver_argv, [b.path, "-in", "input.in"])


class TestViolations(unittest.TestCase):
    def v(self, req, b, cwd=CWD):
        return launch.validate(req, b, cwd)

    def test_matrix(self):
        self.assertTrue(any("apple-bottom-free" in x for x in self.v(launch.Request(mode="reference", binary_id=FZ.id), FZ)))
        self.assertTrue(any("apple-bottom-linked" in x for x in self.v(launch.Request(mode="ab_cpu", binary_id="ref"), REF)))
        self.assertTrue(any("frozen directory" in x for x in self.v(launch.Request(mode="ab_cpu"), FZ, AB_FROZEN_ROOT / "33fbaa8c8944" / "x")))
        rel = frozen(path="pw.x")
        self.assertTrue(any("absolute" in x for x in self.v(launch.Request(mode="ab_cpu"), rel)))
        self.assertTrue(any("OMP_NUM_THREADS=1 only" in x for x in self.v(launch.Request(mode="ab_cpu", omp=2, allow_omp_gt1=True), FZ)))
        self.assertTrue(any("OMP_NUM_THREADS=1 only" in x for x in self.v(launch.Request(mode="ab_cpu", omp=2), FZ)), "omp1_only: refused with or without allow")
        self.assertEqual(self.v(launch.Request(mode="ab_cpu", binary_id="e56d1b46c753", omp=2), E56), [], "non-omp1_only without allow: forced to 1, not refused")
        self.assertTrue(any("AB_TRACE" in x for x in self.v(launch.Request(mode="ab_gpu", trace=True), FZ)))
        self.assertEqual(self.v(launch.Request(mode="ab_gpu", binary_id=FZ3.id, trace=True), FZ3), [])
        for b in (FZ3, E56, DEV):
            self.assertTrue(any("timed refused" in x for x in self.v(launch.Request(mode="ab_cpu", binary_id=b.id, timed=True, allow_dev=True), b)), b.id)
        self.assertTrue(any("timed runs need lock" in x for x in self.v(launch.Request(mode="ab_cpu", timed=True, lock="share"), FZ)))
        self.assertTrue(any("timed runs need lock" in x for x in self.v(launch.Request(mode="ab_cpu", timed=True, lock="build"), FZ)))
        self.assertTrue(any("dev build" in x for x in self.v(launch.Request(mode="ab_cpu", binary_id=DEV.id), DEV)))
        self.assertEqual([x for x in self.v(launch.Request(mode="ab_cpu", binary_id=DEV.id, allow_dev=True), DEV) if "dev build" in x], [])
        bad = frozen(sha_verified=False, launchable=False, launch_reason="SHA256SUMS mismatch")
        self.assertTrue(any("SHA256SUMS" in x for x in self.v(launch.Request(mode="ab_cpu"), bad)))
        self.assertTrue(any("ab_gpu_all fixes" in x for x in self.v(launch.Request(mode="ab_gpu_all", min_gpu_dim=16), FZ)))
        self.assertTrue(any("ab_gpu_all" in x for x in self.v(launch.Request(mode="ab_gpu", min_gpu_dim=0), FZ)))
        self.assertTrue(any("nk" in x for x in self.v(launch.Request(mode="ab_cpu", ranks=2, nk=4), FZ)))
        self.assertTrue(any("custom preset needs input_path" in x for x in self.v(launch.Request(mode="ab_cpu", preset="custom"), FZ)))
        self.assertTrue(any("unknown preset" in x for x in self.v(launch.Request(mode="ab_cpu", preset="zz"), FZ)))
        self.assertEqual(self.v(launch.Request(mode="zzz"), FZ), ["unknown mode 'zzz'; one of ['reference', 'ab_cpu', 'ab_auto', 'ab_gpu', 'ab_gpu_all']"])
        self.assertEqual(self.v(launch.Request(mode="ab_cpu"), None), ["unknown binary id '33fbaa8c8944'"])

    def test_request_from_dict(self):
        r = launch.Request.from_dict({"preset": "C", "binary_id": "33fbaa8c8944", "mode": "ab_gpu", "ranks": "8", "min_gpu_dim": "", "bogus": 1})
        self.assertEqual((r.preset, r.ranks, r.min_gpu_dim), ("C", 8, None))
        self.assertEqual(launch.child_env({"AB_MODE": "cpu"}, SEEDED_ENV), {"PATH": "/usr/bin", "HOME": "/x", "AB_MODE": "cpu"})

    def test_dev_violation_names_live_only_when_live(self):
        live = [x for x in self.v(launch.Request(mode="ab_cpu", binary_id=DEV.id), DEV) if "dev build" in x]
        self.assertEqual(len(live), 1); self.assertIn("LIVE", live[0])
        frozen_dylib = frozen("builds:mpi-opt-ab", group="dev", path=str(Path.home() / "qe-test/builds/mpi-opt-ab/bin/pw.x"),
                              sha_verified=None, dylib_path=str(AB_FROZEN_ROOT / "b8cd26bdaee1/libapplebottom.dylib"),
                              live_library=False, omp1_only=False, cls="dev", citable=False)
        v = [x for x in self.v(launch.Request(mode="ab_cpu", binary_id=frozen_dylib.id), frozen_dylib) if "dev build" in x]
        self.assertEqual(len(v), 1); self.assertNotIn("LIVE", v[0]); self.assertIn("never citable", v[0])


class TestModeLabels(unittest.TestCase):
    """The mode strings state only what was measured: the floor-mode si64 number sits on ab_gpu (the mode the
    2026-10-15 campaign ran), ab_gpu_all carries no wall claim, ab_cpu names no particular QE build."""

    def test_no_unmeasured_performance_claim(self):
        labels = {m["id"]: m["label"] + " " + m["short"] for m in launch.modes_list()}
        self.assertNotIn("slower", labels["ab_gpu_all"].lower())
        self.assertNotIn("si64", labels["ab_gpu_all"].lower().replace("si64 wall exists", ""))
        self.assertIn("no bench-grade si64 wall", labels["ab_gpu_all"])
        self.assertIn("1.14-1.40x", labels["ab_gpu"]); self.assertIn("1 rank", labels["ab_gpu"]); self.assertIn("2026-10-15", labels["ab_gpu"])
        self.assertNotIn("QE develop", labels["ab_cpu"])
        for mid, text in labels.items():
            self.assertNotIn("faster", text.lower(), mid)
            self.assertNotIn("speedup", text.lower(), mid)


# --------------------------------------------------------------------------- compare.py on synthetic records

SPIKE = frozen("5fdc1e839797", dylib_sha256="5fdc1e8397979e55", ab_build_info="apple-bottom 1.4.0 (9059433fbe51-dirty)",
               qe_build="QE develop 7.5 -O3 (OMP=1 only)", cls="experimental", citable=False)
FZ_B = frozen(qe_build="QE develop 7.5 -O3 (OMP=1 only)")
B8CD = frozen("b8cd26bdaee1", sha256="4e34b0442c00906f", dylib_sha256="b8cd26bdaee1abb3", ab_build_info="apple-bottom 1.3.1-dev (1dbdb52abeec)",
              qe_build="QE develop 7.5", omp1_only=False)
REF_B = Binary(**{**REF.__dict__, "qe_build": "QE 7.4.1"})


def record(rid, binary, mode, energy=-22.83836927, wall=1.0, grade="untimed", ranks=1, created="2026-10-16T00:00:00",
           totals=None, input_sha="in1"):
    return {"id": rid, "created": created, "input_sha256": input_sha, "preset": "si2",
            "request": {"mode": mode, "ranks": ranks, "nk": 1, "omp": 1}, "plan": {"launcher": "mpirun"},
            "binary": binary.to_dict(),
            "etiquette": {"timing_grade": grade, "timing_grade_reason": "" if grade == "bench" else "not timed (lock=share)", "contention": []},
            "result": {"total_energy_Ry": energy, "scf_iterations": 8, "wall_qe_s": wall, "forces": None, "stress_kbar": None},
            "ab": {"totals": totals, "gpu_verified": None, "evidence_source": "profile_json" if totals else "none",
                   "build": binary.ab_build_info}}


class TestCompareLabels(unittest.TestCase):
    def setUp(self):
        self.cpu = record("r-cpu", FZ_B, "ab_cpu", wall=2.0, totals={"calls": 100, "gpu_calls": 0, "cpu_calls": 100, "fallbacks": 0, "fallback_breakdown": None})
        self.gpu = record("r-gpu", FZ_B, "ab_gpu_all", wall=4.0, totals={"calls": 100, "gpu_calls": 100, "cpu_calls": 0, "fallbacks": 0, "fallback_breakdown": None})
        self.spike = record("r-spike", SPIKE, "ab_gpu_all", wall=5.0, totals={"calls": 100, "gpu_calls": 100, "cpu_calls": 0, "fallbacks": 0, "fallback_breakdown": None})
        self.ref = record("r-ref", REF_B, "reference", wall=1.0)
        self.b8 = record("r-b8cd", B8CD, "ab_cpu", wall=9.0, totals={"calls": 100, "gpu_calls": 0, "cpu_calls": 100, "fallbacks": 0, "fallback_breakdown": None})
        self.res = compare.compare([self.ref, self.b8, self.spike, self.gpu, self.cpu], ref_id="r-cpu")
        self.rows = {r["id"]: r for r in self.res["rows"]}

    def test_pair_classes_and_sections(self):
        self.assertEqual([r["id"] for r in self.res["rows"]], ["r-cpu", "r-gpu", "r-spike", "r-ref", "r-b8cd"])
        self.assertEqual(self.res["sections"]["same_binary"], ["r-gpu"])
        self.assertEqual(self.res["sections"]["cross_build"], ["r-spike", "r-ref", "r-b8cd"])     # never under same_binary
        g, s, b8, ref = self.rows["r-gpu"], self.rows["r-spike"], self.rows["r-b8cd"], self.rows["r-ref"]
        self.assertEqual((g["pair"], g["same_pw_sha"], g["same_dylib_sha"], g["same_binary"], g["tolerance_Ry"]), ("same_binary", True, True, True, 1e-8))
        self.assertEqual((s["pair"], s["same_pw_sha"], s["same_dylib_sha"], s["same_binary"], s["tolerance_Ry"]), ("same_pw_other_dylib", True, False, False, 1e-8))
        self.assertEqual((b8["pair"], b8["same_pw_sha"], b8["same_binary"], b8["tolerance_Ry"]), ("cross_build", False, False, 1e-5))
        self.assertEqual((ref["pair"], ref["tolerance_Ry"]), ("cross_build", 1e-5))
        self.assertTrue(all(r["within_tolerance"] for r in self.res["rows"] if not r["is_ref"]))

    def test_notes_come_from_the_records(self):
        g, s, b8, ref = self.rows["r-gpu"], self.rows["r-spike"], self.rows["r-b8cd"], self.rows["r-ref"]
        self.assertEqual(g["tolerance_note"], "same pw.x 2f0f02 and libapplebottom 33fbaa: BLAS A/B (only AB_MODE differs), 1e-8 Ry")
        self.assertIn("same pw.x 2f0f02, different libapplebottom 33fbaa vs 5fdc1e", s["tolerance_note"])
        self.assertIn("9059433fbe51-dirty", s["tolerance_note"]); self.assertIn("not the one-binary mode A/B", s["tolerance_note"])
        self.assertEqual(ref["tolerance_note"], "different pw.x (2f0f02 vs e70ae3): QE develop 7.5 -O3 (OMP=1 only) (apple-bottom 1.4.0 c5a1fa6 GEMMs + OpenBLAS) "
                                                "vs QE 7.4.1 (OpenBLAS, no apple-bottom): 1e-5 Ry, not a BLAS A/B")
        self.assertEqual(b8["tolerance_note"], "different pw.x (2f0f02 vs 4e34b0): QE develop 7.5 -O3 (OMP=1 only) (apple-bottom 1.4.0 c5a1fa6 GEMMs + OpenBLAS) "
                                               "vs QE develop 7.5 (apple-bottom 1.3.1-dev 1dbdb52 GEMMs + OpenBLAS): 1e-5 Ry, not a BLAS A/B")
        self.assertNotIn("7.4.1", b8["tolerance_note"])                 # no hardcoded "7.4.1 OpenBLAS vs develop 7.5 -O3"
        self.assertIn("not a BLAS A/B", b8["tolerance_note"])
        # fallback constant only when a record carries no build metadata
        bare = dict(self.b8, id="r-bare", binary={"id": "x", "sha256": "deadbeef"})
        self.assertEqual(compare.row_for(bare, self.cpu)["tolerance_note"], compare.CROSS_BUILD_NOTE_FALLBACK)
        self.assertEqual(compare.row_for(bare, self.cpu)["pair"], "cross_build")

    def test_binary_short_distinguishes_dylibs_on_one_pw(self):
        shorts = {bid: compare.binary_short({"binary": b.to_dict()}) for bid, b in
                  (("fz", FZ_B), ("spike", SPIKE), ("b8", B8CD), ("e56", E56), ("dev", DEV), ("ref", REF_B))}
        self.assertEqual(shorts["fz"], "frozen 33fbaa 1.4.0 c5a1fa6")
        self.assertEqual(shorts["spike"], "frozen 5fdc1e 1.4.0 9059433-dirty")
        self.assertEqual(shorts["b8"], "frozen b8cd26 1.3.1-dev 1dbdb52")
        self.assertEqual(shorts["e56"], "frozen e56d1b pre-1.4.0")
        self.assertEqual(shorts["dev"], "dev mpi-gpu-ab 1.4.0 c5a1fa6 LIVE")
        self.assertEqual(shorts["ref"], "reference 7.4.1")
        self.assertEqual(len(set(shorts.values())), len(shorts))
        self.assertEqual(compare.ab_version_commit("apple-bottom 1.4.0 (9059433fbe51-dirty)"), ("1.4.0", "9059433-dirty"))
        self.assertEqual(compare.ab_version_commit(None), (None, None))

    def test_wall_ratio_only_when_bench_on_one_pw(self):
        g = self.rows["r-gpu"]
        self.assertIsNone(g["wall_ratio_vs_ref"]); self.assertAlmostEqual(g["wall_quotient_untimed"], 2.0)
        self.assertFalse(g["wall_comparable"]); self.assertTrue(g["wall_note"].startswith("untimed"))
        cpu_b = record("b-cpu", FZ_B, "ab_cpu", wall=2.0, grade="bench")
        gpu_b = record("b-gpu", FZ_B, "ab_gpu_all", wall=4.0, grade="bench")
        spike_b = record("b-spike", SPIKE, "ab_gpu_all", wall=5.0, grade="bench")
        ref_b = record("b-ref", REF_B, "reference", wall=1.0, grade="bench")
        rows = {r["id"]: r for r in compare.compare([cpu_b, gpu_b, spike_b, ref_b], ref_id="b-cpu")["rows"]}
        self.assertAlmostEqual(rows["b-gpu"]["wall_ratio_vs_ref"], 2.0); self.assertIsNone(rows["b-gpu"]["wall_quotient_untimed"])
        self.assertEqual(rows["b-gpu"]["wall_note"], "both bench-grade")
        self.assertTrue(rows["b-spike"]["wall_comparable"]); self.assertIn("library A/B", rows["b-spike"]["wall_note"])
        self.assertIsNone(rows["b-ref"]["wall_ratio_vs_ref"]); self.assertEqual(rows["b-ref"]["wall_note"], "cross-build: not a BLAS A/B")

    def test_markdown_tags_and_legend(self):
        md = self.res["markdown"]
        self.assertTrue(md.startswith(compare.MD_HEADER))
        self.assertIn("| 1.0e-08 (same pw.x+dylib) |", md)
        self.assertIn("| 1.0e-08 (same pw.x, other dylib) |", md)
        self.assertIn("| 1.0e-05 (cross-build) |", md)
        table, legend = md.split("\n\n", 1)
        self.assertEqual(len(table.splitlines()), 2 + 5)
        self.assertTrue(legend.startswith("binaries: "))
        for short in ("frozen 33fbaa 1.4.0 c5a1fa6", "frozen 5fdc1e 1.4.0 9059433-dirty", "reference 7.4.1", "frozen b8cd26 1.3.1-dev 1dbdb52"):
            self.assertIn(f"{short} = ", legend)
        self.assertIn("pw.x 2f0f029a", legend); self.assertIn("libapplebottom 5fdc1e83", legend)
        self.assertIn(compare.MD_MODES_LEGEND, legend)
        self.assertFalse(any(rid in legend for rid in self.rows))
        csv_head = compare.export(self.res, "csv").splitlines()[0].split(",")
        self.assertIn("pair", csv_head); self.assertNotIn("wall_ratio_vs_ref", csv_head)


if __name__ == "__main__":
    unittest.main()
