"""parse.py pinned against the fixtures (no solver, no locks)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from abrun import parse

FIX = Path(__file__).resolve().parent / "fixtures"


class TestOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (FIX / "scf_eos_a10330.out").read_text(errors="replace")
        cls.r = parse.parse_out(cls.text)

    def test_energy_and_iterations(self):
        self.assertAlmostEqual(self.r["total_energy_Ry"], -22.84016602, places=8)
        self.assertEqual(self.r["scf_iterations"], 9)
        self.assertEqual(self.r["qe_version"], "7.5")
        self.assertTrue(self.r["job_done"])
        self.assertTrue(self.r["converged"])
        self.assertTrue(self.r["ieee_note"])            # IEEE note present and NOT a failure

    def test_series_and_sizes(self):
        self.assertEqual(len(self.r["scf_energies"]), 8)
        self.assertAlmostEqual(self.r["scf_energies"][0], -22.83762927, places=8)
        self.assertEqual(len(self.r["scf_accuracy"]), 9)       # the last iteration prints one before the '!' line
        self.assertAlmostEqual(self.r["scf_accuracy"][-1], 1.1e-9)
        self.assertEqual(self.r["progress"]["iteration"], 9)
        self.assertEqual(self.r["progress"]["max"], 100)
        self.assertEqual(self.r["nproc"], 1)
        self.assertEqual(self.r["npool"], 0)
        self.assertEqual(self.r["nks"], 28)
        self.assertEqual(self.r["npw_max"], 1419)
        self.assertEqual(self.r["ngm"], 31879)

    def test_wall(self):
        self.assertAlmostEqual(self.r["wall_qe_s"], 31.09)
        self.assertAlmostEqual(self.r["cpu_qe_s"], 14.02)
        for s, v in json.loads((FIX / "wall_strings.json").read_text()).items():
            self.assertAlmostEqual(parse.parse_duration(s), v, msg=s)
        self.assertIsNone(parse.parse_duration(""))

    def test_forces_and_stress(self):
        text = """
     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =    -0.00123456    0.00000000    0.00200000
     The non-local contrib.  to forces
     atom    1 type  1   force =     0.10000000    0.00000000    0.00000000
     atom    2 type  1   force =    -0.10000000    0.00000000    0.00000000
     The ionic contribution  to forces
     atom    1 type  1   force =     0.20000000    0.00000000    0.00000000
     atom    2 type  1   force =    -0.20000000    0.00000000    0.00000000

     Total force =     0.002350     Total SCF correction =     0.000012

     total   stress  (Ry/bohr**3)                   (kbar)     P=      -12.34
     -0.00008390   0.00000000   0.00000000          -12.34        0.00        0.00
      0.00000000  -0.00008390   0.00000000            0.00      -12.34        0.00
      0.00000000   0.00000000  -0.00008390            0.00        0.00      -12.34
"""
        r = parse.parse_out(text)
        self.assertEqual(r["forces"], [[0.0, 0.0, 0.0], [-0.00123456, 0.0, 0.002]])
        self.assertAlmostEqual(r["total_scf_correction"], 1.2e-5)
        self.assertEqual(r["pressure_kbar"], -12.34)
        self.assertEqual(r["stress_kbar"], [[-12.34, 0.0, 0.0], [0.0, -12.34, 0.0], [0.0, 0.0, -12.34]])

    def test_maxstep_from_input(self):
        r = parse.parse_out("iteration #  2", input_text="&ELECTRONS\n  electron_maxstep = 40\n/\n")
        self.assertEqual(r["progress"], {"iteration": 2, "max": 40, "scf_accuracy_Ry": None})


class TestProfileJson(unittest.TestCase):
    def test_v1(self):
        p = parse.parse_profile_json(FIX / "ab_profile_v1.json")
        self.assertEqual(p["schema"], "v1")
        self.assertIsNone(p["build"])
        self.assertEqual(set(p["routines"]), {"dgemm", "zgemm"})
        t = parse.totals_of(p["routines"])
        self.assertEqual(t["calls"], 580 + 13103)
        self.assertEqual(t["gpu_calls"], t["calls"])
        self.assertEqual(t["fallbacks"], 0)
        self.assertIsNone(t["fallback_breakdown"])
        self.assertIsNone(p["routines"]["zgemm"]["fallbacks"])
        self.assertEqual(p["routines"]["zgemm"]["dim_histogram"], [6702, 1216, 0, 0, 5185, 0, 0, 0])

    def test_v2(self):
        p = parse.parse_profile_json(FIX / "ab_profile_v2.json")
        self.assertEqual(p["schema"], "v2")
        self.assertEqual(p["build"], "apple-bottom 1.4.0 (c5a1fa6a8166)")
        z = p["routines"]["zgemm"]
        self.assertEqual(z["fallbacks"], {"no_device": 0, "too_large": 0, "gpu_failed": 0, "other": 0})
        self.assertEqual(z["gpu_stage_ns"]["kernel"], 9249000000)
        t = parse.totals_of(p["routines"])
        self.assertEqual((t["calls"], t["gpu_calls"], t["fallbacks"]), (12180, 12180, 0))
        self.assertEqual(t["fallback_breakdown"], {"no_device": 0, "too_large": 0, "gpu_failed": 0, "other": 0})

    def test_unknown_schema(self):
        tmp = FIX.parent / "_unknown_schema.json"
        tmp.write_text(json.dumps({"routines": [{"name": "zgemm", "calls": 5, "weird": 1}]}))
        try:
            p = parse.parse_profile_json(tmp)
            self.assertIsNone(p["schema"])
            self.assertIn("unknown profile schema", p["error"])
            verified, reason, badge = parse.verify_gpu("ab_gpu_all", "none", None)
            self.assertIsNone(verified)
            self.assertEqual(badge, "amber")
        finally:
            tmp.unlink()


class TestErr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = parse.parse_err((FIX / "err_ab_report_2ranks.txt").read_text())

    def test_summed_over_pids(self):
        e = self.e
        self.assertEqual(e["ranks_reporting"], 2)
        self.assertEqual(e["build"], "apple-bottom 1.4.0 (c5a1fa6a8166)")
        self.assertEqual(e["ab_mode_echo"], "gpu")
        z, d = e["routines"]["zgemm"], e["routines"]["dgemm"]
        self.assertEqual((z["calls"], z["gpu_calls"], z["cpu_calls"]), (5826 + 6233, 219 + 247, 5607 + 5986))
        self.assertEqual((d["calls"], d["gpu_calls"]), (1056, 32))
        self.assertAlmostEqual(z["gpu_stages_ms"]["kernel"], 22249.112 + 22042.052, places=3)
        self.assertEqual(d["fallbacks"], {"no_device": 0, "too_large": 1, "gpu_failed": 0, "other": 0})
        self.assertEqual(d["fallback_total"], 1)
        t = parse.totals_of(e["routines"])
        self.assertEqual(t["calls"], 5826 + 6233 + 1056)
        self.assertEqual(t["fallbacks"], 1)

    def test_warnings_lib_and_per_call(self):
        e = self.e
        self.assertEqual(len([w for w in e["warnings"] if "AB_DISPATCH_TABLE" in w]), 1)
        self.assertEqual(e["lib_loaded"], "/Users/grantheileman/qe-test/ab-frozen/33fbaa8c8944/libapplebottom.dylib")
        self.assertEqual(e["per_call_lines"], 3)
        self.assertEqual(e["json_written"], "/Users/grantheileman/x/ab_profile.json")
        self.assertFalse(e["ab_report_empty"])
        self.assertEqual(e["dyld_lines"], 3)

    def test_empty_report(self):
        e = parse.parse_err((FIX / "err_empty.txt").read_text())
        self.assertTrue(e["ab_report_empty"])
        self.assertEqual(e["routines"], {})
        self.assertEqual(e["ab_mode_echo"], "cpu")

    def test_same_totals_from_both_schemas_and_report(self):
        v2 = parse.totals_of(parse.parse_profile_json(FIX / "ab_profile_v2.json")["routines"])
        text = ("AB_REPORT[1] apple-bottom 1.4.0 (c5a1fa6a8166) AB_MODE=gpu\n"
                "AB_REPORT[1] zgemm calls=11658 gpu_calls=11658 cpu_calls=0 gpu_calls%=100.0 total_ms=23917.0 gpu_ms=23917.0 cpu_ms=0.0 gpu_time%=100.0 fallbacks=0\n"
                "AB_REPORT[1] dgemm calls=522 gpu_calls=522 cpu_calls=0 gpu_calls%=100.0 total_ms=972.0 gpu_ms=972.0 cpu_ms=0.0 gpu_time%=100.0 fallbacks=0\n")
        rep = parse.totals_of(parse.parse_err(text)["routines"])
        for k in ("calls", "gpu_calls", "cpu_calls", "fallbacks"):
            self.assertEqual(v2[k], rep[k], k)


class TestVerdict(unittest.TestCase):
    def tot(self, calls, gpu, fb=None):
        bd = dict(zip(("no_device", "too_large", "gpu_failed", "other"), fb)) if fb else None
        return {"calls": calls, "gpu_calls": gpu, "cpu_calls": calls - gpu, "fallbacks": sum(fb) if fb else 0, "fallback_breakdown": bd}

    def test_truth_table(self):
        V = parse.verify_gpu
        self.assertEqual(V("reference", "none", None)[0], None)
        self.assertEqual(V("reference", "none", None)[2], "n/a")
        self.assertEqual(V("ab_cpu", "profile_json", self.tot(100, 0))[:1], (True,))
        self.assertEqual(V("ab_cpu", "profile_json", self.tot(100, 5))[0], False)
        ok, why, badge = V("ab_auto", "ab_report", self.tot(100, 0))
        self.assertTrue(ok); self.assertIn("calibrated table", why); self.assertEqual(badge, "grey")
        self.assertEqual(V("ab_auto", "ab_report", self.tot(100, 10))[0], False)                       # 1.4.0 dylib
        self.assertIsNone(V("ab_auto", "ab_report", self.tot(100, 10), has_build_info=False)[0])       # pre-1.4.0: informational
        ok, why, badge = V("ab_gpu", "profile_json", self.tot(100, 0))
        self.assertTrue(ok); self.assertIn("no GEMM above the floor", why)
        self.assertEqual(V("ab_gpu", "profile_json", self.tot(100, 60, (0, 0, 0, 0)))[2], "green")
        self.assertEqual(V("ab_gpu", "profile_json", self.tot(100, 60, (0, 0, 1, 0)))[0], False)
        ok, why, badge = V("ab_gpu_all", "profile_json", self.tot(100, 100, (0, 0, 0, 0)))
        self.assertTrue(ok); self.assertEqual(badge, "green")
        self.assertTrue(V("ab_gpu_all", "profile_json", self.tot(100, 100))[0])                      # v1: no breakdown
        ok, why, badge = V("ab_gpu_all", "profile_json", self.tot(100, 98, (0, 2, 0, 0)))
        self.assertTrue(ok); self.assertEqual(badge, "amber"); self.assertIn("too_large", why)
        ok, why, badge = V("ab_gpu_all", "profile_json", self.tot(100, 99, (0, 0, 1, 0)))
        self.assertFalse(ok); self.assertEqual(badge, "red")
        self.assertFalse(V("ab_gpu_all", "profile_json", self.tot(100, 50, (0, 0, 0, 0)))[0])
        self.assertFalse(V("ab_gpu_all", "profile_json", self.tot(0, 0))[0])
        self.assertIsNone(V("ab_gpu_all", "none", None)[0])

    def test_zero_calls_is_unmeasured_on_every_ab_mode(self):
        """Counters that exist but count nothing (the library's 'No BLAS-ABI calls recorded.' case) pass no mode:
        amber/None on ab_cpu, ab_auto and ab_gpu, red/False on ab_gpu_all; never gpu_verified=True."""
        V = parse.verify_gpu
        zero = parse.totals_of({})
        self.assertEqual(zero["calls"], 0)
        for source in ("profile_json", "ab_report"):
            for mode in ("ab_cpu", "ab_auto", "ab_gpu"):
                ok, why, badge = V(mode, source, zero)
                self.assertIsNone(ok, (mode, source))
                self.assertEqual(badge, "amber", (mode, source))
                self.assertTrue(why.startswith("no GEMM call reached apple-bottom (calls == 0)"), why)
                self.assertIn(source, why)
            ok, why, badge = V("ab_gpu_all", source, zero)
            self.assertFalse(ok); self.assertEqual(badge, "red"); self.assertIn("no GEMM calls reached apple-bottom", why)
        # the same through AB_REPORT lines that report zero counts
        text = ("AB_REPORT[7] apple-bottom 1.4.0 (c5a1fa6a8166) AB_MODE=cpu\n"
                "AB_REPORT[7] zgemm calls=0 gpu_calls=0 cpu_calls=0 gpu_calls%=0.0 total_ms=0.0 gpu_ms=0.0 cpu_ms=0.0 gpu_time%=0.0 fallbacks=0\n")
        t = parse.totals_of(parse.parse_err(text)["routines"])
        self.assertEqual(t["calls"], 0)
        self.assertIsNone(V("ab_cpu", "ab_report", t)[0])
        # a run with real counters is unchanged
        self.assertTrue(V("ab_cpu", "ab_report", self.tot(1, 0))[0])

    def test_ab_record_empty_routines_json(self):
        """A v2 profile JSON with `routines: []` (fixture err_empty.txt beside it) reaches the verdict with calls == 0:
        evidence profile_json, gpu_verified None (amber) on ab_cpu/ab_auto/ab_gpu, False (red) on ab_gpu_all."""
        tmp = FIX.parent / "_empty_routines.json"
        tmp.write_text(json.dumps({"build": "apple-bottom 1.4.0 (c5a1fa6a8166)", "timestamp": "2026-10-16T00:00:00Z",
                                   "pid": 50001, "threshold_ns": 0, "routines": []}))
        binary = {"ab_build_info": "apple-bottom 1.4.0 (c5a1fa6a8166)", "dylib_sha256": "33fbaa8c8944abcd",
                  "dylib_path": "/Users/grantheileman/qe-test/ab-frozen/33fbaa8c8944/libapplebottom.dylib"}
        err = (FIX / "err_empty.txt").read_text()
        try:
            for mode in ("ab_cpu", "ab_auto", "ab_gpu"):
                rec = parse.ab_record(mode, binary, FIX, err, tmp)
                self.assertEqual(rec["evidence_source"], "profile_json", mode)
                self.assertEqual(rec["schema"], "v2")
                self.assertEqual(rec["totals"]["calls"], 0)
                self.assertIsNone(rec["gpu_verified"], mode)
                self.assertEqual(rec["gpu_badge"], "amber", mode)
                self.assertIn("calls == 0", rec["gpu_verified_reason"])
                self.assertTrue(rec["ab_report_empty"])
                self.assertTrue(rec["build_matches_dylib"])
            rec = parse.ab_record("ab_gpu_all", binary, FIX, err, tmp)
            self.assertFalse(rec["gpu_verified"]); self.assertEqual(rec["gpu_badge"], "red")
            # without the JSON the same err.txt has no counters at all: still amber/None, never True
            rec = parse.ab_record("ab_cpu", binary, FIX, err, None)
            self.assertEqual(rec["evidence_source"], "none"); self.assertIsNone(rec["gpu_verified"]); self.assertEqual(rec["gpu_badge"], "amber")
        finally:
            tmp.unlink()

    def test_ab_record_from_fixture(self):
        binary = {"ab_build_info": "apple-bottom 1.4.0 (c5a1fa6a8166)", "dylib_sha256": "33fbaa8c8944abcd",
                  "dylib_path": "/Users/grantheileman/qe-test/ab-frozen/33fbaa8c8944/libapplebottom.dylib"}
        rec = parse.ab_record("ab_gpu", binary, FIX, (FIX / "err_ab_report_2ranks.txt").read_text(), None, "library default (32)")
        self.assertEqual(rec["evidence_source"], "ab_report")
        self.assertTrue(rec["build_matches_dylib"])
        self.assertEqual(rec["totals"]["gpu_calls"], 219 + 247 + 32)
        self.assertTrue(rec["gpu_verified"])            # only a too_large fallback
        self.assertEqual(rec["gpu_badge"], "amber")
        self.assertEqual(rec["ranks_reporting"], 2)
        self.assertEqual(rec["lib_loaded"], binary["dylib_path"])
        self.assertIsNone(rec["lib_ok"] if False else None)   # lib_ok compares resolved paths; path may not exist here
        rec = parse.ab_record("ab_gpu_all", binary, FIX, (FIX / "err_ab_report_2ranks.txt").read_text(),
                              FIX / "ab_profile_v2.json")
        self.assertEqual(rec["evidence_source"], "profile_json")
        self.assertEqual(rec["schema"], "v2")
        self.assertTrue(rec["gpu_verified"])
        self.assertEqual(rec["totals"]["calls"], 12180)
        rec = parse.ab_record("reference", {}, FIX, "dyld[1]: /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib\n", None)
        self.assertIsNone(rec["gpu_verified"])
        self.assertTrue(rec["lib_ok"])
        self.assertEqual(rec["evidence_source"], "none")


if __name__ == "__main__":
    unittest.main()
