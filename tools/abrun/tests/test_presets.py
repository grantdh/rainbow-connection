"""presets.py: the &CONTROL rewrite (one-line, comma- and blank-separated namelists, wfcdir, prefix, comments,
unparseable groups) and the pseudopotential-name gate (an absolute or '../' ATOMIC_SPECIES name must never
reach prepare()'s unlink/symlink).  Pure: no run, no solver; the scratch UPF is a temp file."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from abrun import presets

ONE_LINE = ("&CONTROL calculation='scf', outdir='/Users/x/Dev/Claude/apple-bottom/build', pseudo_dir='/tmp/pp', "
            "prefix='pwscf', wfcdir='/Volumes/x' /\n&SYSTEM ibrav=2, nat=2, ntyp=1 /\n&ELECTRONS conv_thr=1d-8 /\n"
            "ATOMIC_SPECIES\n Si 28.086 Si.upf\n")


def control_entries(text: str) -> dict:
    _s, _e, entries = presets.parse_control_group(text)
    return {k.lower(): v for k, v in entries}


class TestRewrite(unittest.TestCase):
    def test_one_line_namelist(self):
        out = presets.rewrite_input(ONE_LINE)
        ent = control_entries(out)
        self.assertEqual(ent["outdir"], "'./tmp'")
        self.assertEqual(ent["pseudo_dir"], "'./pseudo'")
        self.assertEqual(ent["prefix"], "'pwscf'")
        self.assertEqual(ent["calculation"], "'scf'")
        self.assertNotIn("wfcdir", ent)
        self.assertNotIn("apple-bottom", out)
        self.assertNotIn("/Volumes", out)
        self.assertNotIn("/tmp/pp", out)
        self.assertIn("&SYSTEM ibrav=2, nat=2, ntyp=1 /", out, "only the CONTROL group is touched")
        self.assertIn("outdir = './tmp'", out)                       # one entry per line: the line-anchored parsers see it
        meta = presets.parse_input_meta(out)
        self.assertEqual((meta["prefix"], meta["calculation"], meta["pseudos"]), ("pwscf", "scf", ["Si.upf"]))

    def test_blank_separated_entries(self):
        out = presets.rewrite_input("&CONTROL calculation='scf' outdir='/x/y' prefix = 'a' /\n&SYSTEM ibrav=2 /\n")
        ent = control_entries(out)
        self.assertEqual(ent["outdir"], "'./tmp'")
        self.assertEqual(ent["calculation"], "'scf'")
        self.assertEqual(ent["prefix"], "'a'")
        self.assertNotIn("/x/y", out)

    def test_comments_and_quotes(self):
        src = ("&CONTROL\n  calculation = 'scf', ! a comment, with a comma and outdir='fake'\n"
               "  title = 'it''s, quoted /slash/'\n  outdir = \"/else/where\" ! trailing\n/\n&SYSTEM ibrav=2 /\n")
        out = presets.rewrite_input(src)
        ent = control_entries(out)
        self.assertEqual(ent["outdir"], "'./tmp'")
        self.assertEqual(ent["title"], "'it''s, quoted /slash/'")
        self.assertNotIn("/else/where", out)
        self.assertNotIn("a comment", out)

    def test_injected_when_absent(self):
        out = presets.rewrite_input("&CONTROL\n  calculation = 'scf'\n/\n&SYSTEM ibrav=2 /\n")
        ent = control_entries(out)
        self.assertEqual((ent["outdir"], ent["pseudo_dir"]), ("'./tmp'", "'./pseudo'"))
        self.assertTrue(out.index("outdir") < out.index("/\n&SYSTEM"), "inside the group, before its terminator")

    def test_template_unchanged(self):
        self.assertEqual(presets.rewrite_input(presets.SI2_TEMPLATE), presets.SI2_TEMPLATE)

    def test_prefix_with_path_refused(self):
        for p in ("'../../../x'", "'a/b'", "'/abs'", "''", '"../x"'):      # quoted: an unquoted '/' is the terminator, for QE too
            with self.assertRaises(presets.PresetError, msg=p):
                presets.rewrite_input(f"&CONTROL prefix={p} /\n&SYSTEM ibrav=2 /\n")
        self.assertIn("prefix = 'si64'", presets.rewrite_input("&CONTROL prefix='si64' /\n&SYSTEM ibrav=2 /\n"))

    def test_unparseable_is_an_error(self):
        for bad in ("&SYSTEM ibrav=2 /\n",                                   # no CONTROL group
                    "&CONTROL calculation='scf'\n&SYSTEM ibrav=2 /\n",       # no terminator before the next group
                    "&CONTROL calculation='scf\n/\n",                        # unterminated quote
                    "&CONTROL calculation /\n",                              # entry without '='
                    "&CONTROL 3x = 1 /\n"):                                  # not a variable name
            with self.assertRaises(presets.PresetError, msg=bad):
                presets.rewrite_input(bad)

    def test_placeholder_survives(self):
        with self.assertRaises(presets.PresetError):
            presets.rewrite_input("&CONTROL calculation='scf' /\nATOMIC_SPECIES\n Si 1 PSEUDO_DIR/Si.upf\n")


class TestPseudoNames(unittest.TestCase):
    def test_bare_names_only(self):
        self.assertEqual(presets.check_pseudo_name("Si.pbe-n-rrkjus_psl.1.0.0.UPF"), "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
        for bad in ("/Users/x/pseudos/Si.upf", "../../../../pseudos/Si.upf", ".", "..", "a/b.upf", "", " Si.upf"):
            with self.assertRaises(presets.PresetError, msg=bad):
                presets.check_pseudo_name(bad)

    def test_prepare_refuses_a_path_and_leaves_the_file_alone(self):
        tmp = Path(tempfile.mkdtemp(prefix="abrun-presets-"))
        try:
            upf = tmp / "pp" / "X.upf"
            upf.parent.mkdir()
            upf.write_text('<UPF version="2.0.1">\n<PP_HEADER element="Si"/>\n</UPF>\n' * 50)
            size = upf.stat().st_size
            for name in (str(upf), "../pp/X.upf"):
                inp = tmp / "evil.in"
                inp.write_text(f"&CONTROL\n calculation='scf'\n/\n&SYSTEM ibrav=2 /\nATOMIC_SPECIES\n Si 28.086 {name}\n")
                cwd = tmp / "run"
                with self.assertRaises(presets.PresetError, msg=name) as cm:
                    presets.prepare("custom", cwd, str(inp), write=True)
                self.assertIn("not a bare file name", str(cm.exception))
                self.assertTrue(upf.is_file() and not upf.is_symlink())
                self.assertEqual(upf.stat().st_size, size)
                self.assertFalse((cwd / "pseudo").exists(), "nothing was written")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
