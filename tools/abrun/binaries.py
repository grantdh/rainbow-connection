"""Binary discovery and labelling (read-only: shasum, otool, nm, strings).

Catalog entries are `Binary` records.  The catalog is cached in tools/abrun/binaries.json
(ABRUN_BINARIES_CACHE overrides) keyed by (path, mtime, size); `inventory(refresh=True)` rescans.
Nothing here writes outside this package directory, and nothing is ever executed from the binaries.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import (AB_FROZEN_ROOT, HOME, LIVE_LIBRARY_DIRS, PKG_DIR, QE_BUILDS_ROOT, exlib)

CACHE_PATH = Path(os.environ.get("ABRUN_BINARIES_CACHE", str(PKG_DIR / "binaries.json"))).expanduser()
CACHE_VERSION = 4                                   # 4: private builds labelled from their own strings

STUDIO_LABELS = {                                   # muppet-studio's own label strings, reused verbatim
    "/usr/local/bin/pw.x": "qe (PATH, OpenBLAS, no apple-bottom)",
    str(HOME / "qe-test/q-e-qe-7.4.1/bin/pw.x"): "qe-7.4.1 (+apple-bottom static)",
}

# Advisory class per known freeze (hash -> (class, reason)).  The raw PROVENANCE.txt is always shown too.
CLASS_OVERRIDES = {
    "33fbaa8c8944": ("citable", "override table: 1.4.0 release c5a1fa6, clean worktree, -O3 pw.x"),
    "b8cd26bdaee1": ("citable", "override table: 1.4.0 candidate 1dbdb52, -O0 -g pw.x: slow, VAL-1 build"),
    "3ae144a9cadd": ("experiments", "override table: release/v1.5.0 local 317bb91, numerics identical to 1.4.0, "
                                    "adds AB_TRACE; fine for parity and trace runs, not for timed/cited numbers"),
    "7bbb86408ee5": ("pre-1.4.0", "override table: eaad87e; no AB_REPORT/ab_build_info; auto uses fixed FLOP "
                                  "thresholds; old JSON schema"),
    "e56d1b46c753": ("parity-only", "override table: dirty tree, commit unknown; NOT for any performance number"),
    "72cef8fb565a": ("experimental", "override table: dirty/spike; not for citing"),
    "5fdc1e839797": ("experimental", "override table: dirty/spike; not for citing"),
}
O3_PW_SHA_PREFIX = "2f0f029a"                       # the shared -O3 pw.x of 33fb/3ae1/72ce/5fdc: OMP=1 only
SOLVER_NAMES = ("ph.x", "yambo", "p2y", "ypp")


@dataclass
class Binary:
    id: str
    group: str                                      # reference | frozen | static | dev | other
    path: str
    sha256: str | None = None
    sha_verified: bool | None = None
    qe_version: str | None = None
    qe_build: str = ""
    links: dict = field(default_factory=dict)       # applebottom, openblas, accelerate, fftw3_omp, libmpi
    ab_symbols: dict = field(default_factory=lambda: {"U": 0, "T": 0})
    exlib_ab_symbols: int = 0                       # exlib.apple_bottom_symbols() for continuity with the examples
    ab_linkage: str = "none"                        # dynamic | static | none
    rpaths: list = field(default_factory=list)
    dylib_path: str | None = None
    dylib_sha256: str | None = None
    ab_build_info: str | None = None
    features: list = field(default_factory=list)
    profile_schema: str = "none"                    # v2 | v1 | none
    live_library: bool = False
    omp1_only: bool = False
    cls: str = "unclassified"                       # serialised as "class"
    citable: bool = False
    class_reason: str = ""
    provenance: str | None = None
    sha256sums: dict = field(default_factory=dict)
    blas: str = ""
    studio_label: str | None = None
    label: str = ""
    launchable: bool = True
    launch_reason: str | None = None
    private: bool = False
    solver: str = "pw.x"
    mtime: float = 0.0
    size: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Binary":
        d = dict(d)
        d["cls"] = d.pop("class", d.pop("cls", "unclassified"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_apple_bottom(self) -> bool:
        return self.ab_linkage != "none"


# --------------------------------------------------------------------------- probes (read-only)

def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def sha256_of(path: Path | str) -> str | None:
    out = _run(["shasum", "-a", "256", str(path)]).split()
    return out[0] if out else None


def otool_links(path: Path) -> dict:
    out = _run(["otool", "-L", str(path)])
    libmpi = None
    for line in out.splitlines()[1:]:
        s = line.strip().split(" (")[0]
        if s.endswith("libmpi.40.dylib") or "/libmpi." in s:
            libmpi = s
    return {
        "applebottom": "libapplebottom" in out,
        "openblas": "libopenblas" in out,
        "accelerate": "Accelerate.framework" in out,
        "fftw3_omp": "libfftw3_omp" in out,
        "libmpi": libmpi,
    }


def otool_rpaths(path: Path) -> list[str]:
    out = _run(["otool", "-l", str(path)])
    rpaths, grab = [], 0
    for line in out.splitlines():
        if "LC_RPATH" in line:
            grab = 3
            continue
        if grab:
            grab -= 1
            m = re.search(r"\bpath (\S+)", line)
            if m:
                rpaths.append(m.group(1))
                grab = 0
    return rpaths


def nm_ab_symbols(path: Path) -> dict:
    out = _run(["nm", str(path)])
    return {"U": sum(1 for l in out.splitlines() if " U _ab_" in l),
            "T": sum(1 for l in out.splitlines() if " T _ab_" in l)}


def strings_of(path: Path) -> list[str]:
    return _run(["strings", "-a", str(path)]).splitlines()


def qe_version_from_strings(lines: list[str]) -> str | None:
    """QE stores version_number as a padded literal ('7.4.1 ', '7.5   '); no 'PWSCF v.' literal survives."""
    for s in lines:
        m = re.match(r"^\s*([6-9]\.\d(?:\.\d)?)\s*$", s)
        if m:
            return m.group(1)
    return None


def resolve_dylib(path: Path, rpaths: list[str]) -> str | None:
    """First existing <rpath>/libapplebottom.dylib with @executable_path/@loader_path -> the binary's directory."""
    here = str(path.resolve().parent)
    for rp in rpaths:
        cand = Path(rp.replace("@executable_path", here).replace("@loader_path", here)) / "libapplebottom.dylib"
        if cand.exists():
            return str(cand)
    return None


def is_live_library(dylib_path: str | None) -> bool:
    if not dylib_path:
        return False
    real = Path(dylib_path).resolve()
    for d in LIVE_LIBRARY_DIRS:
        try:
            if real.is_relative_to(d.resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def dylib_facts(dylib_path: str | None) -> dict:
    if not dylib_path or not Path(dylib_path).exists():
        return {"dylib_sha256": None, "ab_build_info": None, "features": [], "profile_schema": "none"}
    lines = strings_of(Path(dylib_path))
    build = next((s.strip() for s in lines if re.match(r"^apple-bottom [0-9]", s)), None)
    feats = sorted({m for s in lines for m in re.findall(r"\bAB_[A-Z0-9_]+", s)})
    schema = "v2" if any("gpu_calls" in s for s in lines) else ("v1" if any("gpu_dispatch_count" in s for s in lines) else "none")
    return {"dylib_sha256": sha256_of(dylib_path), "ab_build_info": build, "features": feats, "profile_schema": schema}


def read_sha256sums(d: Path) -> dict:
    sums = {}
    f = d / "SHA256SUMS"
    if f.exists():
        for line in f.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sums[parts[-1].lstrip("*")] = parts[0]
    return sums


# --------------------------------------------------------------------------- classification

def classify(hash_id: str | None, provenance: str | None, group: str) -> tuple[str, bool, str]:
    """(class, citable, reason). Advisory; the raw PROVENANCE.txt is always shown alongside."""
    if group == "reference":
        return "reference", True, "apple-bottom-free reference build (nm: 0 _ab_ symbols)"
    if group == "static":
        return "static", False, "QE 7.4.1 relinked with a static apple-bottom; never citable"
    if group == "dev":
        return "dev", False, "build tree under ~/qe-test/builds: rebuilt by other sessions, never citable"
    if hash_id in CLASS_OVERRIDES:
        c, why = CLASS_OVERRIDES[hash_id]
        return c, c == "citable", why
    text = provenance or ""
    if re.search(r"not for citing|SPIKE|Not a release build|-dirty", text, re.I):
        return "experimental", False, "PROVENANCE: not for citing / spike / dirty"
    if re.search(r"NOT suitable for: any performance number|commit is therefore NOT known", text, re.I):
        return "parity-only", False, "PROVENANCE: not suitable for performance numbers"
    if re.search(r"suitable for: citable|release/v1\.4\.0", text) and "dirty" not in text:
        return "citable", True, "PROVENANCE: citable / release build, clean"
    return "unclassified", False, "unknown freeze: no override and no recognised PROVENANCE wording"


def _opt_summary(prov: str | None, links: dict, name_hint: str = "") -> str:
    if prov and re.search(r"-O3", prov):
        return "-O3"
    if prov and re.search(r"-O0", prov):
        return "-O0"
    if "o3" in name_hint.lower():
        return "-O3"
    if links.get("fftw3_omp"):
        return "-O3 -fopenmp"
    return ""


def make_label(b: Binary) -> str:
    qe = b.qe_build or f"QE {b.qe_version or '?'}"
    if b.ab_linkage == "none":
        blas = "BLAS: OpenBLAS for everything"
        ab = f"apple-bottom: none (nm {b.ab_symbols.get('U', 0)} _ab_ symbols)"
        if b.group == "reference":
            head = f"reference | {b.path}"
        else:
            head = f"{b.group} {b.id}"
    elif b.private:
        # pw.x-xp / pw.x-async: experimentally patched QE sharing a freeze's dylib; the directory's PROVENANCE.txt
        # describes the sibling pw.x, so nothing from it is claimed here.
        build = re.sub(r"^apple-bottom ", "", b.ab_build_info) if b.ab_build_info else "(no ab_build_info)"
        blas = f"GEMMs via apple-bottom {build} (the sibling freeze's dylib)"
        ab = "private research build: provenance not catalogued (the directory's PROVENANCE.txt describes the sibling pw.x)"
        head = f"private {b.id}"
    elif b.ab_linkage == "static":
        blas = "GEMMs via static apple-bottom (cpu->Accelerate/AMX, gpu->Metal); rest OpenBLAS"
        ab = "apple-bottom: static relink"
        head = f"{b.group} {b.id}"
    else:
        build = re.sub(r"^apple-bottom ", "", b.ab_build_info) if b.ab_build_info else "(no ab_build_info: pre-1.4.0)"
        blas = f"GEMMs via apple-bottom {build}: cpu->Accelerate/AMX, gpu->Metal; rest OpenBLAS"
        ab = ""
        head = f"{b.group} {b.id}"
    tail = b.cls
    if b.omp1_only:
        tail += ", OMP=1 only"
    if b.live_library:
        tail += ", LIVE library"
    if b.group == "dev":
        tail += ", never cite"
    parts = [head, qe, blas] + ([ab] if ab else []) + [tail]
    return " | ".join(parts)


# --------------------------------------------------------------------------- probing one binary

def probe(path: Path, bid: str, group: str, hash_id: str | None = None, solver: str = "pw.x",
          private: bool = False) -> Binary:
    path = Path(path)
    st = path.stat()
    links = otool_links(path)
    rpaths = otool_rpaths(path)
    syms = nm_ab_symbols(path)
    lines = strings_of(path)
    prov_file = path.parent / "PROVENANCE.txt"
    # A private pw.x-* shares a freeze directory whose PROVENANCE.txt describes the sibling pw.x, not this binary:
    # its QE build, OMP rule and class come from its own strings and sha only.
    prov = prov_file.read_text(errors="replace") if prov_file.exists() and not private else None
    sums = read_sha256sums(path.parent) if group == "frozen" else {}
    sha = sha256_of(path)
    linkage = "static" if syms["T"] > 0 else ("dynamic" if syms["U"] > 0 or links["applebottom"] else "none")
    dylib = resolve_dylib(path, rpaths) if linkage == "dynamic" else None
    dfacts = dylib_facts(dylib)
    live = is_live_library(dylib)
    qe_version = qe_version_from_strings(lines)
    cls, citable, reason = classify(hash_id, prov, group)
    omp1 = bool(prov and re.search(r"OMP(_NUM_THREADS)?\s*=\s*1 only", prov, re.I)) or bool(sha and sha.startswith(O3_PW_SHA_PREFIX))
    # QE 7.5 has no release build on this machine: every 7.5 pw.x here is the develop checkout (5b7f8e1bc).
    develop = bool(prov and re.search(r"QE develop", prov)) or (qe_version == "7.5" and group in ("frozen", "dev"))
    opt = "" if private else _opt_summary(prov, links, path.parent.parent.name if group == "dev" else "")
    qe_build = f"QE {'develop ' if develop else ''}{qe_version or '?'}"
    if opt:
        qe_build += f" {opt}"
    if omp1:
        qe_build += " (OMP=1 only)"
    sha_verified = None
    launchable, launch_reason = True, None
    if group == "frozen":
        want_pw, want_dl = sums.get(path.name), sums.get("libapplebottom.dylib")
        sha_verified = bool(want_pw and want_pw == sha and want_dl and want_dl == dfacts["dylib_sha256"])
        if not sha_verified:
            launchable, launch_reason = False, "SHA256SUMS mismatch"
    if live:
        citable = False
    if solver != "pw.x":
        launchable, launch_reason = False, "informational: not a launchable pw.x"
    if private:
        launchable, launch_reason = False, "private research build"
        cls, citable, reason = "private", False, "private research build (pw.x-xp / pw.x-async): never offered, never cited"
    if linkage == "none":
        blas = "OpenBLAS for all BLAS/LAPACK; no apple-bottom"
    elif linkage == "static":
        blas = "dgemm/zgemm via static apple-bottom (cpu: Accelerate/AMX, gpu: Metal DD); other BLAS/LAPACK: OpenBLAS"
    else:
        blas = "dgemm/zgemm via apple-bottom (cpu: Accelerate/AMX, gpu: Metal DD); other BLAS/LAPACK: OpenBLAS"
    b = Binary(id=bid, group=group, path=str(path), sha256=sha, sha_verified=sha_verified, qe_version=qe_version,
               qe_build=qe_build, links=links, ab_symbols=syms, exlib_ab_symbols=exlib.apple_bottom_symbols(path),
               ab_linkage=linkage, rpaths=rpaths, dylib_path=dylib, dylib_sha256=dfacts["dylib_sha256"],
               ab_build_info=dfacts["ab_build_info"], features=dfacts["features"], profile_schema=dfacts["profile_schema"],
               live_library=live, omp1_only=omp1, cls=cls, citable=citable, class_reason=reason, provenance=prov,
               sha256sums=sums, blas=blas, studio_label=STUDIO_LABELS.get(str(path)), launchable=launchable,
               launch_reason=launch_reason, private=private, solver=solver, mtime=st.st_mtime, size=st.st_size)
    if solver != "pw.x" and linkage == "dynamic":
        b.label = (f"other {bid} | {solver} | apple-bottom-linked to {'the LIVE build' if live else 'a frozen build'}; "
                   f"not apple-bottom-free; AB_MODE applies | {dylib or 'dylib unresolved'}")
    elif solver != "pw.x":
        b.label = f"other {bid} | {solver} | apple-bottom-free (OpenBLAS; nm {syms['U']} _ab_ symbols)"
    else:
        b.label = make_label(b)
    return b


# --------------------------------------------------------------------------- candidates

def candidates() -> list[dict]:
    """Every binary the catalog knows about, with how to classify it (paths may be absent)."""
    out = [{"path": exlib.QE_REF, "id": "ref", "group": "reference"}]
    if AB_FROZEN_ROOT.is_dir():
        for d in sorted(AB_FROZEN_ROOT.iterdir()):
            if (d / "pw.x").exists():
                out.append({"path": d / "pw.x", "id": d.name, "group": "frozen", "hash": d.name})
            for p in sorted(d.glob("pw.x-*")):
                out.append({"path": p, "id": f"{d.name}:{p.name}", "group": "frozen", "hash": d.name, "private": True})
    out.append({"path": exlib.QE_BIN / "pw.x", "id": "qe741-static", "group": "static"})
    if QE_BUILDS_ROOT.is_dir():
        for d in sorted(QE_BUILDS_ROOT.iterdir()):
            if (d / "bin" / "pw.x").exists():
                out.append({"path": d / "bin" / "pw.x", "id": f"builds:{d.name}", "group": "dev"})
    others = [
        (Path("/usr/local/bin/ph.x"), "ref:ph.x", "ph.x"),
        (QE_BUILDS_ROOT / "mpi-gpu-ab-o3/bin/ph.x", "builds:mpi-gpu-ab-o3:ph.x", "ph.x"),
        (HOME / "yambo-build/builds/dp-mpi/yambo", "yambo:dp-mpi", "yambo"),
        (HOME / "yambo-build/builds/dp-mpi-gpu-ab/yambo", "yambo:dp-mpi-gpu-ab", "yambo"),
        (exlib.YAMBO_BIN / "yambo", "yambo:bin", "yambo"),
        (exlib.YAMBO_BIN / "p2y", "p2y:bin", "p2y"),
        (exlib.YAMBO_BIN / "ypp", "ypp:bin", "ypp"),
    ]
    for p, bid, solver in others:
        out.append({"path": p, "id": bid, "group": "other", "solver": solver})
    return out


# --------------------------------------------------------------------------- cache + inventory

_FAKES: dict[str, Binary] = {}


def register_fake(binary: Binary) -> None:
    """Tests register a fake pw.x (tests/fake_pw.py) under its own id; never persisted to the cache."""
    _FAKES[binary.id] = binary


def _load_cache() -> dict:
    try:
        d = json.loads(CACHE_PATH.read_text())
        if d.get("cache_version") == CACHE_VERSION:
            return d
    except (OSError, ValueError):
        pass
    return {"cache_version": CACHE_VERSION, "entries": {}}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1))
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def inventory(refresh: bool = False, include_private: bool = False, include_dev: bool | None = None) -> dict:
    """{generated, binaries[], other_solvers[], private[], hidden_private[], hidden_dev[]}.

    include_private=True (the ?all=1 of the API) lists dev builds among binaries.  The private research
    builds (pw.x-xp / pw.x-async) are never in binaries[]: their paths are in hidden_private and their
    records in private[] (so `get()` can name them when refusing).  Dev builds are otherwise hidden
    (hidden_dev names them)."""
    if include_dev is None:
        include_dev = include_private
    cache = {"cache_version": CACHE_VERSION, "entries": {}} if refresh else _load_cache()
    entries = cache["entries"]
    bins, others, private, hidden_private, hidden_dev = [], [], [], [], []
    dirty = False
    for c in candidates():
        p = Path(c["path"])
        if not p.exists():
            continue
        st = p.stat()
        key = str(p)
        cached = entries.get(key)
        if cached and cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size:
            b = Binary.from_dict(cached["binary"])
        else:
            b = probe(p, c["id"], c["group"], c.get("hash"), c.get("solver", "pw.x"), c.get("private", False))
            entries[key] = {"mtime": st.st_mtime, "size": st.st_size, "binary": b.to_dict()}
            dirty = True
        if c["group"] == "other":
            others.append(b)
        elif b.private:
            hidden_private.append(b.path)
            private.append(b)
        elif b.group == "dev" and not include_dev:
            hidden_dev.append(b.id)
        else:
            bins.append(b)
    if dirty:
        _save_cache(cache)
    bins.extend(_FAKES.values())
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "binaries": bins, "other_solvers": others,
            "private": private, "hidden_private": hidden_private, "hidden_dev": hidden_dev}


def get(binary_id: str, refresh: bool = False) -> Binary | None:
    """Look a launchable-or-not pw.x up by id (dev and private ids included; 'ref' is the reference)."""
    if binary_id in _FAKES:
        return _FAKES[binary_id]
    inv = inventory(refresh=refresh, include_private=True, include_dev=True)
    for b in inv["binaries"] + inv["private"]:
        if b.id == binary_id:
            return b
    return None


def verify_at_launch(binary: Binary) -> list[str]:
    """Re-read SHA256SUMS and recompute the two hashes right before a run; returns violations."""
    v = []
    p = Path(binary.path)
    if not p.exists():
        return [f"binary missing: {p}"]
    if binary.id in _FAKES:                      # tests/fake_pw.py: no SHA256SUMS, nothing to re-verify
        return v
    if binary.group == "frozen":
        sums = read_sha256sums(p.parent)
        want_pw = sums.get(p.name)
        have_pw = sha256_of(p)
        if not want_pw or want_pw != have_pw:
            v.append(f"SHA256SUMS mismatch for {p.name}: want {want_pw}, have {have_pw}")
        if binary.dylib_path:
            want_dl, have_dl = sums.get("libapplebottom.dylib"), sha256_of(binary.dylib_path)
            if not want_dl or want_dl != have_dl:
                v.append(f"SHA256SUMS mismatch for libapplebottom.dylib: want {want_dl}, have {have_dl}")
    elif binary.sha256:
        have = sha256_of(p)
        if have != binary.sha256:
            v.append(f"{p} changed since it was catalogued (sha256 {have} != {binary.sha256}); refresh the inventory")
    return v
