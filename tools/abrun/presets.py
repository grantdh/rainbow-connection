"""Preset inputs and pseudopotentials.

Every run gets a fresh scratch cwd with input.in (outdir -> './tmp', pseudo_dir -> './pseudo') and the
UPFs symlinked into cwd/pseudo/ after a sha256 pin check and exlib.check_upf.  The si64 presets live in
another session's directory (MUPPET_SI64_INPUTS overrides) with a /private/tmp pseudo_dir, so the text is
rewritten and a leftover placeholder is an error, never a silent run.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from . import HOME, REPO_ROOT, exlib

SI64_INPUTS = Path(os.environ.get("MUPPET_SI64_INPUTS", str(HOME / "qe-test/xp/inputs"))).expanduser()
PSEUDO_SEARCH = [SI64_INPUTS, REPO_ROOT / "shared/pseudo", HOME / "pseudos", exlib.PSEUDO_SI.parent]

# Pinned sha256 of every pseudopotential a preset may name (shasum -a 256, 2026-10-15).
UPF_SHA256 = {
    "Si.upf": "4fb30411d7d95fe5c171720cb049d8b20752a72f46380ecfb25021e1d696bad0",
    "Si.pbe-n-kjpaw_psl.1.0.0.UPF": "8f630aadb55097e3b456a4a7c49b0aabafb5c4b8ef586ff15fe247f98361d39e",
    "Si.pbe-n-rrkjus_psl.1.0.0.UPF": "669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a",
}

# The SI_SCF template of examples/si-thin-film/run.py (calc scf, no nbnd, 4x4x4 shifted, conv_thr 1e-8).
SI2_TEMPLATE = """&CONTROL
  calculation = 'scf'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.20
  nat = 2
  ntyp = 1
  ecutwfc = 30.0
  force_symmorphic = .true.
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 1 1 1
"""

CAMPAIGN_SOURCE = ("apple-bottom benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/tables.md "
                   "(apple-bottom 1.4.0, 33fbaa8c8944, conv_thr 1e-6; observations, 2026-10-15)")

PRESETS: dict[str, dict] = {
    "si2": {
        "id": "si2",
        "label": "si2: 2-atom Si, US pseudo, ecut 30, 4x4x4 shifted (~5 s)",
        "source": "examples/si-thin-film/run.py SI_SCF template",
        "template": SI2_TEMPLATE,
        "input_sha256": None,
        "expected_E_Ry": None,
        "expected_source": None,
        "wall_obs": {},
        "wall_source": None,
    },
    "C": {
        "id": "C",
        "label": "C: Si64, 160 bands, Gamma (campaign observations: 68 s ab_cpu / 78 s ab_gpu at 1 rank)",
        "source": str(SI64_INPUTS / "C_si64_160b_gamma.in"),
        "input_sha256": "f8223baae5f72bf315e81d8c6476041ec5d5ce53dfc91746e9cc974d08ca97c5",
        "expected_E_Ry": -2989.94060060,
        "expected_source": CAMPAIGN_SOURCE + ", 1 rank",
        "wall_obs": {"ab_cpu@np1": 68.1, "ab_auto@np1": 68.7, "ab_gpu@np1": 77.7, "ab_cpu@np8": 13.9, "ab_gpu@np8": 16.2},
        "wall_source": CAMPAIGN_SOURCE,
    },
    "B": {
        "id": "B",
        "label": "B: Si64, 160 bands, 2x2x2 k (campaign observations: 188 s ab_cpu / 264 s ab_gpu at 1 rank)",
        "source": str(SI64_INPUTS / "B_si64_160b_k222.in"),
        "input_sha256": "ba99ea070d134afd5343377a133c71be8dee9b58ad035e3d586485cab5a9a9ff",
        "expected_E_Ry": -2990.44093535,
        "expected_source": CAMPAIGN_SOURCE + ", 1 rank",
        "wall_obs": {"ab_cpu@np1": 188.3, "ab_auto@np1": 187.2, "ab_gpu@np1": 264.0, "ab_cpu@np8nk4": 53.6, "ab_gpu@np8nk4": 65.1},
        "wall_source": CAMPAIGN_SOURCE,
    },
    "A": {
        "id": "A",
        "label": "A: Si64, 500 bands, 2x2x2 k (campaign observations: 306 s ab_cpu / 397 s ab_gpu at 1 rank)",
        "source": str(SI64_INPUTS / "A_si64_500b_k222.in"),
        "input_sha256": "dd191315cce8a14ef2e4dfa5cc5cde1a4058c48b5d16bc482ee9a669567b7eea",
        "expected_E_Ry": -2990.44276149,
        "expected_source": CAMPAIGN_SOURCE + ", 1 rank",
        "wall_obs": {"ab_cpu@np1": 306.4, "ab_auto@np1": 302.6, "ab_gpu@np1": 396.9, "ab_cpu@np8nk4": 97.1, "ab_gpu@np8nk4": 102.4},
        "wall_source": CAMPAIGN_SOURCE,
    },
    "custom": {
        "id": "custom",
        "label": "Custom .in (absolute path)",
        "source": None,
        "input_sha256": None,
        "expected_E_Ry": None,
        "expected_source": None,
        "wall_obs": {},
        "wall_source": None,
    },
}


class PresetError(ValueError):
    """A preset cannot be prepared (missing file, placeholder left, pin mismatch); message names the file."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- input text

def parse_input_meta(text: str) -> dict:
    """nat, nbnd, ntyp, ecutwfc, calculation, prefix, electron_maxstep, k-points line, pseudo names."""
    def num(key, cast=float):
        m = re.search(rf"^\s*{key}\s*=\s*([-\d.deD+]+)", text, re.M | re.I)
        if not m:
            return None
        try:
            return cast(m.group(1).replace("d", "e").replace("D", "e"))
        except ValueError:
            return None

    def quoted(key):
        m = re.search(rf"^\s*{key}\s*=\s*'([^']*)'", text, re.M | re.I)
        return m.group(1) if m else None

    kp = None
    m = re.search(r"^\s*K_POINTS\s*(?:\{?\s*(\w+)\s*\}?)?\s*$\n?(.*)$", text, re.M | re.I)
    if m:
        kind = (m.group(1) or "").lower()
        kp = "gamma" if kind == "gamma" else (" ".join(m.group(2).split()) if kind in ("automatic", "") else kind)
    return {
        "calculation": quoted("calculation"), "prefix": quoted("prefix"),
        "nat": num("nat", int), "ntyp": num("ntyp", int), "nbnd": num("nbnd", int),
        "ecutwfc": num("ecutwfc"), "electron_maxstep": num("electron_maxstep", int) or 100,
        "kpoints": kp, "pseudos": pseudo_names(text),
    }


def pseudo_names(text: str) -> list[str]:
    names = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*ATOMIC_SPECIES", line, re.I):
            for row in lines[i + 1:]:
                if not row.strip() or re.match(r"^\s*[A-Z_]{5,}", row) and row.split()[0].isupper():
                    break
                parts = row.split()
                if len(parts) >= 3:
                    names.append(parts[2])
            break
    return names


def check_pseudo_name(name: str) -> str:
    """A UPF name in ATOMIC_SPECIES must be a bare file name: '/x.upf' or '../../x.upf' would make prepare()
    unlink and replace a file outside the run's pseudo/ directory (the real pseudopotential)."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or os.sep in name or name != name.strip():
        raise PresetError(f"pseudopotential name {name!r} is not a bare file name (no path separators, not '.'/'..')")
    return name


# --- the &CONTROL namelist --------------------------------------------------------------------------------

_RE_NL_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^()]*\))?(?:%[A-Za-z_][A-Za-z0-9_]*)*")


_RE_NEXT_ENTRY = re.compile(r"\s+(" + _RE_NL_KEY.pattern + r")\s*=")


def _nl_tokens(body: str) -> list[str]:
    """Split a namelist body into entry strings on top-level commas and newlines, and on blanks that precede
    the next `key =` (Fortran accepts blanks as value separators: `calculation='scf' outdir='x'`); quotes are
    respected and '!' comments outside quotes dropped.  Raises PresetError on an unterminated quote."""
    entries, cur, i, n = [], [], 0, len(body)
    while i < n:
        c = body[i]
        if c in "'\"":
            j = body.find(c, i + 1)
            while j != -1 and j + 1 < n and body[j + 1] == c:        # doubled quote inside the string
                j = body.find(c, j + 2)
            if j == -1:
                raise PresetError("&CONTROL: unterminated quoted string")
            cur.append(body[i:j + 1]); i = j + 1
        elif c == "!":
            j = body.find("\n", i)
            i = n if j == -1 else j
        elif c in ",\n":
            entries.append("".join(cur)); cur = []; i += 1
        elif c in " \t" and "=" in "".join(cur) and _RE_NEXT_ENTRY.match(body, i):
            entries.append("".join(cur)); cur = []; i += 1          # blank-separated `key = value key2 = value2`
        else:
            cur.append(c); i += 1
    entries.append("".join(cur))
    return [e.strip() for e in entries if e.strip()]


def parse_control_group(text: str) -> tuple[int, int, list[tuple[str, str]]]:
    """(start, end, [(key, value_text)...]) of the &CONTROL group: `start` is the offset of '&', `end` the
    offset just past the terminating '/'.  Raises PresetError when there is no &CONTROL, no terminator, an
    entry without '=', or a key that is not a namelist variable."""
    m = re.search(r"^[ \t]*&CONTROL\b", text, re.M | re.I)
    if not m:
        raise PresetError("&CONTROL group not found in the input")
    start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
    i, n = m.end(), len(text)
    end = None
    while i < n:                                                    # find '/' outside quotes and comments
        c = text[i]
        if c in "'\"":
            j = text.find(c, i + 1)
            while j != -1 and j + 1 < n and text[j + 1] == c:
                j = text.find(c, j + 2)
            if j == -1:
                raise PresetError("&CONTROL: unterminated quoted string")
            i = j + 1
        elif c == "!":
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif c == "/":
            end = i + 1
            break
        elif c == "&" and re.match(r"&(END|[A-Za-z]+)", text[i:]):
            raise PresetError("&CONTROL: another namelist starts before the terminating '/'")
        else:
            i += 1
    if end is None:
        raise PresetError("&CONTROL: no terminating '/'")
    entries = []
    for ent in _nl_tokens(text[m.end():end - 1]):
        if "=" not in ent:
            raise PresetError(f"&CONTROL: cannot parse entry {ent!r}")
        key, val = ent.split("=", 1)
        key, val = key.strip(), val.strip()
        if not _RE_NL_KEY.fullmatch(key):
            raise PresetError(f"&CONTROL: cannot parse key {key!r}")
        entries.append((key, val))
    return start, end, entries


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1].replace(v[0] * 2, v[0])
    return v


def rewrite_input(src_text: str, cwd: Path | None = None) -> str:
    """Rewrite the &CONTROL group: outdir -> './tmp', pseudo_dir -> './pseudo' (set or injected inside the
    group, one entry per line), wfcdir removed (it would send wavefunctions outside the run directory), every
    other entry kept in order; prefix kept but refused when it contains '/' or '..'.  Comma-separated and
    one-line namelists are parsed like QE does, so the original paths can never survive after the '/'.

    Raises PresetError when the group cannot be parsed unambiguously or a placeholder ('OUTDIR',
    'PSEUDO_DIR') or a /private/tmp path survives."""
    start, end, entries = parse_control_group(src_text)
    out: list[tuple[str, str]] = []
    seen = set()
    for key, val in entries:
        k = key.lower()
        if k == "wfcdir":
            continue
        if k == "outdir":
            val = "'./tmp'"
        elif k == "pseudo_dir":
            val = "'./pseudo'"
        elif k == "prefix":
            p = _unquote(val)
            if "/" in p or os.sep in p or ".." in p or not p:
                raise PresetError(f"prefix {p!r} must be a bare name (no '/' or '..')")
        seen.add(k)
        out.append((key, val))
    if "outdir" not in seen:
        out.insert(0, ("outdir", "'./tmp'"))
    if "pseudo_dir" not in seen:
        out.insert(min(1, len(out)), ("pseudo_dir", "'./pseudo'"))
    group = "&CONTROL\n" + "".join(f"  {k} = {v}\n" for k, v in out) + "/"
    text = src_text[:start] + group + src_text[end:]
    for bad in ("OUTDIR", "PSEUDO_DIR", "/private/tmp"):
        if bad in text:
            raise PresetError(f"rewritten input still contains the placeholder {bad!r}")
    if not text.endswith("\n"):
        text += "\n"
    return text


# --------------------------------------------------------------------------- pseudopotentials

def pseudo_search_dirs(src: Path | None) -> list[Path]:
    dirs = [SI64_INPUTS]
    if src is not None:
        dirs.append(Path(src).resolve().parent)
    dirs += [REPO_ROOT / "shared/pseudo", HOME / "pseudos", exlib.PSEUDO_SI.parent]
    seen, out = set(), []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def resolve_upfs(names: list[str], src: Path | None = None) -> list[dict]:
    """[{file, sha256, path, pinned}] for every UPF, searching the preset dirs; PresetError names a missing one."""
    out = []
    for name in names:
        check_pseudo_name(name)
        found = None
        for d in pseudo_search_dirs(src):
            cand = d / name
            if cand.is_file():
                found = cand
                break
        if found is None:
            raise PresetError(f"pseudopotential {name} not found in {[str(d) for d in pseudo_search_dirs(src)]}")
        exlib.check_upf(found)
        sha = sha256_file(found)
        pinned = UPF_SHA256.get(name)
        if pinned and sha != pinned:
            raise PresetError(f"{found}: sha256 {sha} != pinned {pinned}")
        out.append({"file": name, "sha256": sha, "path": str(found), "pinned": pinned is not None})
    return out


# --------------------------------------------------------------------------- preparing a run directory

def source_text(preset_id: str, input_path: str | None = None) -> tuple[str, Path | None]:
    """(source input text, source path or None for the si2 template)."""
    if preset_id == "custom":
        if not input_path:
            raise PresetError("custom preset needs input_path (an absolute .in path)")
        p = Path(input_path)
        if not p.is_absolute():
            raise PresetError(f"input_path must be absolute: {input_path}")
        if not p.is_file():
            raise PresetError(f"input file not found: {p}")
        return p.read_text(errors="replace"), p
    if preset_id not in PRESETS:
        raise PresetError(f"unknown preset {preset_id!r}; known: {list(PRESETS)}")
    pr = PRESETS[preset_id]
    if pr.get("template"):
        return pr["template"], None
    p = Path(pr["source"])
    if not p.is_file():
        raise PresetError(f"preset {preset_id} source missing: {p} (set MUPPET_SI64_INPUTS)")
    text = p.read_text(errors="replace")
    if pr["input_sha256"] and sha256_text(text) != pr["input_sha256"]:
        raise PresetError(f"preset {preset_id} source {p} changed: sha256 {sha256_text(text)} != pinned {pr['input_sha256']}")
    return text, p


def describe(preset_id: str) -> dict:
    """The /api/presets card: availability, pins, meta parsed from the (source) text, expected E, walls."""
    pr = dict(PRESETS[preset_id])
    pr.pop("template", None)
    card = {**pr, "available": True, "reason": None, "nat": None, "nbnd": None, "kpoints": None, "pseudos": []}
    if preset_id == "custom":
        return card
    try:
        text, src = source_text(preset_id)
        meta = parse_input_meta(text)
        card.update({"nat": meta["nat"], "nbnd": meta["nbnd"], "kpoints": meta["kpoints"]})
        card["pseudos"] = resolve_upfs(meta["pseudos"], src)
    except (PresetError, ValueError) as exc:
        card["available"], card["reason"] = False, str(exc)
    return card


def list_presets() -> list[dict]:
    return [describe(pid) for pid in PRESETS]


def prepare(preset_id: str, cwd: Path, input_path: str | None = None, write: bool = True) -> dict:
    """Write cwd/input.in and cwd/pseudo/<UPF> symlinks; return the prepared-input record.

    Keys: preset, source, source_sha256, input_sha256 (of the rewritten text the run sees), input_text,
    pseudos [{file, sha256, path, pinned}], meta {nat, nbnd, kpoints, electron_maxstep, ...},
    expected_E_Ry, expected_source, wall_obs, wall_source."""
    text, src = source_text(preset_id, input_path)
    rewritten = rewrite_input(text, cwd)
    meta = parse_input_meta(rewritten)
    pseudos = resolve_upfs(meta["pseudos"], src)
    pr = PRESETS[preset_id]
    rec = {
        "preset": preset_id, "source": str(src) if src else pr["source"], "source_sha256": sha256_text(text),
        "input_sha256": sha256_text(rewritten), "input_text": rewritten, "pseudos": pseudos, "meta": meta,
        "expected_E_Ry": pr["expected_E_Ry"], "expected_source": pr["expected_source"],
        "wall_obs": pr["wall_obs"], "wall_source": pr["wall_source"],
    }
    if write:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "input.in").write_text(rewritten)
        pdir = cwd / "pseudo"
        pdir.mkdir(exist_ok=True)
        pdir_resolved = pdir.resolve()
        for ps in pseudos:
            link = pdir / check_pseudo_name(ps["file"])
            # belt and braces: only ever unlink/symlink a direct child of this run's pseudo/ directory
            if link.parent.resolve() != pdir_resolved or link.name != ps["file"]:
                raise PresetError(f"refusing to replace {link}: not inside {pdir}")
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(ps["path"], link)
    return rec
