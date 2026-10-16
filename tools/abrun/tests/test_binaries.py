"""Read-only assertions against the real inventory (each skipped when its path is absent)."""

from __future__ import annotations

import unittest
from pathlib import Path

from abrun import AB_FROZEN_ROOT, binaries, compare

INV = None


def inv():
    global INV
    if INV is None:
        INV = binaries.inventory(refresh=False, include_private=True)
    return INV


def by_id(bid):
    for b in inv()["binaries"] + inv()["other_solvers"]:
        if b.id == bid:
            return b
    return None


class TestReference(unittest.TestCase):
    def test_reference(self):
        b = by_id("ref")
        if b is None:
            self.skipTest("/usr/local/bin/pw.x absent")
        self.assertEqual(b.ab_linkage, "none")
        self.assertEqual(b.ab_symbols, {"U": 0, "T": 0})
        self.assertEqual(b.exlib_ab_symbols, 0)
        self.assertIn("/opt/homebrew/opt/open-mpi/lib/libmpi", b.links["libmpi"])
        self.assertEqual(b.cls, "reference")
        self.assertTrue(b.citable)
        self.assertEqual(b.qe_version, "7.4.1")
        self.assertEqual(b.studio_label, "qe (PATH, OpenBLAS, no apple-bottom)")
        self.assertTrue(b.label.startswith("reference | /usr/local/bin/pw.x | QE 7.4.1 | BLAS: OpenBLAS for everything"))


class TestFrozen(unittest.TestCase):
    def frozen(self):
        return [b for b in inv()["binaries"] if b.group == "frozen" and not b.private]

    def test_each_frozen(self):
        fz = self.frozen()
        if not fz:
            self.skipTest("no ~/qe-test/ab-frozen builds")
        for b in fz:
            with self.subTest(b.id):
                self.assertTrue(b.links["applebottom"])
                self.assertEqual(b.ab_linkage, "dynamic")
                self.assertTrue(b.dylib_path, "dylib must resolve")
                self.assertEqual(Path(b.dylib_path).parent, Path(b.path).parent, "dylib inside its own directory")
                self.assertTrue(b.sha_verified, "SHA256SUMS must match")
                self.assertTrue(b.launchable)
                self.assertFalse(b.live_library)
                self.assertEqual(b.exlib_ab_symbols, 2)
                self.assertIsNotNone(b.provenance)

    def test_override_table(self):
        expect = {"33fbaa8c8944": ("citable", True, True, False, "v2"), "b8cd26bdaee1": ("citable", True, False, False, "v2"),
                  "3ae144a9cadd": ("experiments", False, True, True, "v2"), "7bbb86408ee5": ("pre-1.4.0", False, False, False, "v1"),
                  "e56d1b46c753": ("parity-only", False, False, False, "v1"), "72cef8fb565a": ("experimental", False, True, True, "v2"),
                  "5fdc1e839797": ("experimental", False, True, True, "v2")}
        for h, (cls, citable, omp1, trace, schema) in expect.items():
            b = by_id(h)
            if b is None:
                continue
            with self.subTest(h):
                self.assertEqual(b.cls, cls)
                self.assertEqual(b.citable, citable)
                self.assertEqual(b.omp1_only, omp1)
                self.assertEqual("AB_TRACE" in b.features, trace)
                self.assertEqual(b.profile_schema, schema)
                if h in ("e56d1b46c753", "7bbb86408ee5"):
                    self.assertIsNone(b.ab_build_info)
                    self.assertNotIn("AB_REPORT", b.features)
                else:
                    self.assertIsNotNone(b.ab_build_info)
                    self.assertIn("AB_REPORT", b.features)
        b = by_id("33fbaa8c8944")
        if b:
            self.assertEqual(b.ab_build_info, "apple-bottom 1.4.0 (c5a1fa6a8166)")
            self.assertEqual(b.dylib_path, str(AB_FROZEN_ROOT / "33fbaa8c8944/libapplebottom.dylib"))
            self.assertTrue(b.sha256.startswith("2f0f029a"))
            self.assertNotIn("AB_TEST_FAIL_STAGE", b.features)

    def test_private_hidden(self):
        hidden = inv()["hidden_private"]
        default = binaries.inventory(refresh=False, include_private=False)
        ids = {b.id for b in default["binaries"]}
        for p in hidden:
            self.assertNotIn(Path(p).name, {Path(b.path).name for b in default["binaries"]})
        self.assertFalse(any(":" in i and i.split(":")[0] in {"33fbaa8c8944", "3ae144a9cadd"} for i in ids))
        if (AB_FROZEN_ROOT / "33fbaa8c8944/pw.x-xp").exists():
            self.assertIn(str(AB_FROZEN_ROOT / "33fbaa8c8944/pw.x-xp"), hidden)
        if (AB_FROZEN_ROOT / "3ae144a9cadd/pw.x-async").exists():
            self.assertIn(str(AB_FROZEN_ROOT / "3ae144a9cadd/pw.x-async"), hidden)
        self.assertFalse(any(b.group == "dev" for b in default["binaries"]), "dev builds hidden by default")
        # never offered: not in binaries[] even with include_private (--all / ?all=1); their records sit in private[]
        self.assertFalse(any(b.private for b in inv()["binaries"]))
        self.assertEqual(sorted(b.path for b in inv()["private"]), sorted(hidden))
        for b in inv()["private"]:
            with self.subTest(b.id):
                self.assertIn(":pw.x-", b.id)
                self.assertFalse(b.launchable); self.assertEqual(b.cls, "private"); self.assertFalse(b.citable)
                self.assertTrue(b.label.startswith("private "))
                # labelled from its own strings only: nothing inherited from the sibling pw.x's PROVENANCE.txt
                self.assertIsNone(b.provenance)
                self.assertNotIn("-O3", b.label); self.assertNotIn("OMP=1", b.qe_build); self.assertFalse(b.omp1_only)
                self.assertIn("provenance not catalogued", b.label)
                got = binaries.get(b.id)
                self.assertIsNotNone(got, "get() still names a private build so validate() can refuse it by name")
                self.assertEqual(got.launch_reason, "private research build")

    def test_binary_short_distinct_per_freeze(self):
        fz = self.frozen()
        if not fz:
            self.skipTest("no ~/qe-test/ab-frozen builds")
        shorts = {b.id: compare.binary_short({"binary": b.to_dict()}) for b in fz}
        self.assertEqual(len(set(shorts.values())), len(shorts), shorts)
        for bid, s in shorts.items():
            self.assertTrue(s.startswith(f"frozen {bid[:6]} "), s)
        if "33fbaa8c8944" in shorts:
            self.assertEqual(shorts["33fbaa8c8944"], "frozen 33fbaa 1.4.0 c5a1fa6")
        if "5fdc1e839797" in shorts and "72cef8fb565a" in shorts:
            self.assertTrue(shorts["5fdc1e839797"].endswith("9059433-dirty")); self.assertNotEqual(shorts["5fdc1e839797"], shorts["72cef8fb565a"])


class TestOthers(unittest.TestCase):
    def test_static(self):
        b = by_id("qe741-static")
        if b is None:
            self.skipTest("q-e-qe-7.4.1/bin/pw.x absent")
        self.assertEqual(b.ab_linkage, "static")
        self.assertGreater(b.ab_symbols["T"], 0)
        self.assertFalse(b.citable)
        self.assertEqual(b.studio_label, "qe-7.4.1 (+apple-bottom static)")

    def test_dev_live(self):
        b = by_id("builds:mpi-gpu-ab")
        if b is None:
            self.skipTest("builds/mpi-gpu-ab absent")
        self.assertEqual(b.group, "dev")
        self.assertTrue(b.live_library)
        self.assertFalse(b.citable)
        self.assertIn("LIVE library", b.label)

    def test_yambo(self):
        a = by_id("yambo:dp-mpi")
        if a is not None:
            self.assertEqual(a.ab_linkage, "none")
            self.assertFalse(a.launchable)
        for bid in ("yambo:dp-mpi-gpu-ab", "yambo:bin"):
            b = by_id(bid)
            if b is not None:
                self.assertEqual(b.ab_linkage, "dynamic")
                self.assertTrue(b.live_library)
                self.assertIn("LIVE", b.label)
                self.assertFalse(b.launchable)

    def test_labels_unique(self):
        labels = [b.label for b in inv()["binaries"]]
        self.assertEqual(len(labels), len(set(labels)))

    def test_verify_at_launch(self):
        b = by_id("33fbaa8c8944")
        if b is None:
            self.skipTest("33fbaa8c8944 absent")
        self.assertEqual(binaries.verify_at_launch(b), [])


if __name__ == "__main__":
    unittest.main()
