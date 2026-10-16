"""Comparison of run records: energies against a reference with the honest tolerance class, GEMM/GPU
columns, wall comparability rules, and the markdown table the examples embed in RESULTS.md.

Pair classes (per row, field `pair`):
  same_binary          same pw.x sha256 AND same libapplebottom sha256: the only honest BLAS A/B (one frozen
                       binary, only AB_MODE differs), 1e-8 Ry.
  same_pw_other_dylib  same pw.x, different libapplebottom (four freezes share the -O3 pw.x 2f0f029a): a
                       library A/B at 1e-8 Ry, labelled as such and never under the same-binary heading.
  cross_build          different pw.x (reference 7.4.1 OpenBLAS vs an apple-bottom build): 1e-5 Ry, 'not a
                       BLAS A/B'; the note names both builds from the records' binary metadata.
No speedup is ever computed: wall_ratio_vs_ref is null unless both runs are bench-grade on the same pw.x.
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics

SAME_BUILD_TOL = 1e-8
CROSS_BUILD_TOL = 1e-5
# Fallback only: used when a record carries no binary metadata (qe_build / qe_version / ab_linkage).
CROSS_BUILD_NOTE_FALLBACK = "different pw.x sha256 (build metadata missing on a record): 1e-5 Ry, not a BLAS A/B"
FORCE_TOL = {True: 1e-7, False: 1e-5}            # Ry/bohr, same pw.x / cross-build
STRESS_TOL = {True: 0.01, False: 0.1}             # kbar
MD_HEADER = "| run | binary | mode | E_tot (Ry) | dE vs ref | tol | iters | QE wall (s), observation | grade | GEMM calls | GPU calls | fallbacks |"
MD_MODES_LEGEND = ("modes: reference = apple-bottom-free pw.x (OpenBLAS for all BLAS/LAPACK); ab_cpu = AB_MODE=cpu (dgemm/zgemm "
                   "on Accelerate/AMX via libapplebottom, rest OpenBLAS); ab_auto = AB_MODE=auto (calibrated table); ab_gpu = "
                   "AB_MODE=gpu at the library floor; ab_gpu_all = AB_MODE=gpu AB_MIN_GPU_DIM=0. tol: same pw.x+dylib = the "
                   "one-binary BLAS A/B, 1e-8 Ry; same pw.x, other dylib = library A/B, 1e-8 Ry; cross-build = different QE "
                   "build, not a BLAS A/B, 1e-5 Ry. Walls are observations; no ratio unless both runs are bench-grade on one pw.x.")
RE_AB_BUILD = re.compile(r"^apple-bottom\s+(\S+)\s*\(([0-9a-f]+)(-dirty)?\)")


def ab_version_commit(build_info: str | None) -> tuple[str | None, str | None]:
    """'apple-bottom 1.4.0 (c5a1fa6a8166)' -> ('1.4.0', 'c5a1fa6'); '... (9059433fbe51-dirty)' -> ('1.4.0', '9059433-dirty')."""
    if not build_info:
        return None, None
    m = RE_AB_BUILD.match(build_info.strip())
    if not m:
        ver = build_info.replace("apple-bottom ", "").split(" ")[0]
        return ver or None, None
    return m.group(1), m.group(2)[:7] + (m.group(3) or "")


def _short(sha: str | None, n: int = 6) -> str:
    return (sha or "?")[:n]


def binary_short(rec: dict) -> str:
    """Table cell: group, freeze hash, apple-bottom version + commit (so two 1.4.0 dylibs read differently)."""
    b = rec.get("binary") or {}
    g, bid = b.get("group", "?"), b.get("id", "?")
    ver, commit = ab_version_commit(b.get("ab_build_info"))
    lib = " ".join(x for x in (ver or ("pre-1.4.0" if b.get("ab_linkage") == "dynamic" else None), commit) if x)
    if g == "reference":
        return f"reference {b.get('qe_version') or ''}".strip()
    if g == "frozen":
        return f"frozen {bid[:6]} {lib or 'pre-1.4.0'}"
    if g == "dev":
        return f"dev {bid.split(':', 1)[-1]} {lib}".strip() + (" LIVE" if b.get("live_library") else "")
    if g == "static":
        return f"static {b.get('qe_version') or ''}".strip()
    return f"{g} {bid}"


def _blas_short(b: dict) -> str:
    link = b.get("ab_linkage")
    if link == "none":
        return "OpenBLAS, no apple-bottom"
    if link == "static":
        return "static apple-bottom GEMMs + OpenBLAS"
    ver, commit = ab_version_commit(b.get("ab_build_info"))
    lib = " ".join(x for x in (ver or "pre-1.4.0", commit) if x)
    return f"apple-bottom {lib} GEMMs + OpenBLAS"


def _has_build_meta(b: dict) -> bool:
    return bool(b.get("qe_build") or b.get("qe_version")) and b.get("ab_linkage") is not None


def build_desc(rec: dict) -> str | None:
    """One-line description of a record's binary from its catalogued metadata (None when absent)."""
    b = rec.get("binary") or {}
    if not _has_build_meta(b):
        return None
    qe = b.get("qe_build") or f"QE {b.get('qe_version')}"
    parts = [qe, b.get("blas") or _blas_short(b), f"pw.x {_short(b.get('sha256'), 8)}"]
    if b.get("ab_linkage") == "dynamic":
        parts.append(f"libapplebottom {_short(b.get('dylib_sha256'), 8)} ({b.get('ab_build_info') or 'no ab_build_info: pre-1.4.0'})")
    return "; ".join(parts)


def _get(rec: dict, *keys, default=None):
    cur = rec
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def pair_kind(rec: dict, ref: dict) -> str:
    """same_binary | same_pw_other_dylib | cross_build (a missing pw.x sha256 on either side is cross_build)."""
    pw, pw_ref = _get(rec, "binary", "sha256"), _get(ref, "binary", "sha256")
    if not pw or pw != pw_ref:
        return "cross_build"
    if _get(rec, "binary", "dylib_sha256") == _get(ref, "binary", "dylib_sha256"):
        return "same_binary"
    return "same_pw_other_dylib"


def tolerance_note(rec: dict, ref: dict, kind: str) -> str:
    """The sentence behind the tolerance, built from both records' binary metadata."""
    b, br = rec.get("binary") or {}, ref.get("binary") or {}
    pw = _short(b.get("sha256"))
    if kind == "same_binary":
        if b.get("dylib_sha256"):
            return f"same pw.x {pw} and libapplebottom {_short(b.get('dylib_sha256'))}: BLAS A/B (only AB_MODE differs), 1e-8 Ry"
        return f"same pw.x {pw}: BLAS A/B (only AB_MODE differs), 1e-8 Ry"
    if kind == "same_pw_other_dylib":
        return (f"same pw.x {pw}, different libapplebottom {_short(br.get('dylib_sha256'))} vs {_short(b.get('dylib_sha256'))} "
                f"({br.get('ab_build_info') or 'pre-1.4.0'} vs {b.get('ab_build_info') or 'pre-1.4.0'}): library A/B, "
                f"not the one-binary mode A/B, 1e-8 Ry")
    if not (_has_build_meta(b) and _has_build_meta(br)):
        return CROSS_BUILD_NOTE_FALLBACK
    qe, qe_ref = b.get("qe_build") or f"QE {b.get('qe_version')}", br.get("qe_build") or f"QE {br.get('qe_version')}"
    return (f"different pw.x ({_short(br.get('sha256'))} vs {pw}): {qe_ref} ({_blas_short(br)}) vs {qe} ({_blas_short(b)}): "
            f"1e-5 Ry, not a BLAS A/B")


def select_reference(runs: list[dict], ref_id: str | None = None, pins: dict | None = None) -> str | None:
    """Explicit ref, else the pinned run for the input sha256, else the newest reference-mode run with the
    same input, else the newest run."""
    if not runs:
        return None
    ids = {r["id"] for r in runs}
    if ref_id and ref_id in ids:
        return ref_id
    newest = max(runs, key=lambda r: r.get("created") or "")
    sha = newest.get("input_sha256")
    if pins and sha in pins and pins[sha] in ids:
        return pins[sha]
    refs = [r for r in runs if _get(r, "request", "mode") == "reference" and r.get("input_sha256") == sha]
    if refs:
        return max(refs, key=lambda r: r.get("created") or "")["id"]
    return newest["id"]


def _max_abs_diff(a, b):
    if a is None or b is None:
        return None
    try:
        return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    except (TypeError, ValueError):
        return None


def wall_note(rec: dict, ref: dict) -> tuple[bool, str]:
    """(wall_comparable, note): both bench-grade, equal ranks/nk/omp and launcher, same pw.x."""
    for r, who in ((rec, "this run"), (ref, "reference")):
        grade = _get(r, "etiquette", "timing_grade", default="untimed")
        if grade != "bench":
            why = _get(r, "etiquette", "timing_grade_reason", default="")
            return False, f"untimed ({who}: {why})" if why else f"untimed ({who})"
    if _get(rec, "etiquette", "contention") or _get(ref, "etiquette", "contention"):
        return False, "contended"
    for k in ("ranks", "nk", "omp"):
        if _get(rec, "request", k) != _get(ref, "request", k):
            return False, f"{k} differ"
    if _get(rec, "plan", "launcher") != _get(ref, "plan", "launcher"):
        return False, "launcher differs"
    kind = pair_kind(rec, ref)
    if kind == "cross_build":
        return False, "cross-build: not a BLAS A/B"
    if kind == "same_pw_other_dylib":
        return True, "both bench-grade; same pw.x, different libapplebottom (library A/B, not the one-binary mode A/B)"
    return True, "both bench-grade"


def row_for(rec: dict, ref: dict, energy_tol: float | None = None) -> dict:
    same_input = rec.get("input_sha256") == ref.get("input_sha256")
    kind = pair_kind(rec, ref)
    same_sha = kind != "cross_build"                                   # same pw.x sha256 (either dylib)
    e, e_ref = _get(rec, "result", "total_energy_Ry"), _get(ref, "result", "total_energy_Ry")
    tol = energy_tol if energy_tol is not None else (SAME_BUILD_TOL if same_sha else CROSS_BUILD_TOL)
    dE = (e - e_ref) if (same_input and e is not None and e_ref is not None) else None
    it, it_ref = _get(rec, "result", "scf_iterations"), _get(ref, "result", "scf_iterations")
    w, w_ref = _get(rec, "result", "wall_qe_s"), _get(ref, "result", "wall_qe_s")
    comparable, note = wall_note(rec, ref)
    quotient = (w / w_ref) if (w and w_ref) else None
    totals = _get(rec, "ab", "totals") or {}
    calls, gpu = totals.get("calls"), totals.get("gpu_calls")
    dF = _max_abs_diff(_get(rec, "result", "forces"), _get(ref, "result", "forces")) if same_input else None
    dS = _max_abs_diff(_get(rec, "result", "stress_kbar"), _get(ref, "result", "stress_kbar")) if same_input else None
    return {
        "id": rec["id"], "is_ref": rec["id"] == ref["id"], "binary_short": binary_short(rec), "binary_desc": build_desc(rec),
        "binary_id": _get(rec, "binary", "id"), "mode": _get(rec, "request", "mode"), "preset": rec.get("preset"),
        "pw_sha_short": _short(_get(rec, "binary", "sha256"), 8), "dylib_sha_short": _short(_get(rec, "binary", "dylib_sha256"), 8) if _get(rec, "binary", "dylib_sha256") else None,
        "same_input": same_input, "total_energy_Ry": e, "dE_vs_ref_Ry": dE,
        "pair": kind, "same_pw_sha": same_sha, "same_dylib_sha": kind == "same_binary", "same_binary": kind == "same_binary",
        "tolerance_Ry": tol, "within_tolerance": (abs(dE) <= tol) if dE is not None else None,
        "tolerance_note": tolerance_note(rec, ref, kind),
        "scf_iterations": it, "iters_delta": (it - it_ref) if (it is not None and it_ref is not None) else None,
        "wall_qe_s": w, "wall_ratio_vs_ref": quotient if comparable else None,
        "wall_quotient_untimed": None if comparable else quotient,
        "wall_comparable": comparable, "wall_note": note,
        "timing_grade": _get(rec, "etiquette", "timing_grade", default="untimed"),
        "gemm_calls": calls, "gpu_calls": gpu, "gpu_pct": (100.0 * gpu / calls) if calls else None,
        "fallbacks": totals.get("fallbacks"), "fallback_breakdown": totals.get("fallback_breakdown"),
        "gpu_verified": _get(rec, "ab", "gpu_verified"), "evidence_source": _get(rec, "ab", "evidence_source"),
        "lib_build": _get(rec, "ab", "build"), "lib_ok": _get(rec, "ab", "lib_ok"),
        "max_dF_Ry_bohr": dF, "max_dstress_kbar": dS,
        "dF_within": (dF <= FORCE_TOL[same_sha]) if dF is not None else None,
        "dstress_within": (dS <= STRESS_TOL[same_sha]) if dS is not None else None,
        "ranks": _get(rec, "request", "ranks"), "nk": _get(rec, "request", "nk"), "omp": _get(rec, "request", "omp"),
    }


def groups_of(runs: list[dict]) -> list[dict]:
    g: dict[str, list] = {}
    for r in runs:
        key = "|".join(str(x) for x in (r.get("preset"), _get(r, "binary", "id"), _get(r, "request", "mode"),
                                        _get(r, "request", "ranks"), _get(r, "request", "nk"), _get(r, "request", "omp")))
        g.setdefault(key, []).append(r)
    out = []
    for key, rs in g.items():
        walls = [w for w in (_get(r, "result", "wall_qe_s") for r in rs) if w is not None]
        es = [e for e in (_get(r, "result", "total_energy_Ry") for r in rs) if e is not None]
        out.append({"key": key, "n": len(rs), "ids": [r["id"] for r in rs],
                    "wall_median": statistics.median(walls) if walls else None,
                    "wall_range": [min(walls), max(walls)] if walls else None,
                    "energy_spread_Ry": (max(es) - min(es)) if es else None})
    return out


def _fmt(x, nd=8):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _fmt_sci(x):
    return "-" if x is None else f"{x:.1e}"


PAIR_TAG = {"same_binary": "same pw.x+dylib", "same_pw_other_dylib": "same pw.x, other dylib", "cross_build": "cross-build"}


def markdown_legend(rows: list[dict]) -> str:
    """Two lines after the table: each distinct binary with its catalogued build, then the fixed mode/tol key."""
    seen: dict[str, str | None] = {}
    for r in rows:
        seen.setdefault(r["binary_short"], r.get("binary_desc"))
    bins = "; ".join(f"{k} = {v or 'no build metadata on the record'}" for k, v in seen.items())
    return f"binaries: {bins}\n{MD_MODES_LEGEND}"


def to_markdown(rows: list[dict], ref_id: str) -> str:
    with_fs = any(r["max_dF_Ry_bohr"] is not None or r["max_dstress_kbar"] is not None for r in rows)
    header = MD_HEADER + (" max dF (Ry/bohr) | max dsigma (kbar) |" if with_fs else "")
    sep = "|" + "---|" * (header.count("|") - 1)
    lines = [header, sep]
    for r in rows:
        if r["is_ref"]:
            de, tol = "ref", "-"
        elif not r["same_input"]:
            de, tol = "different input", "-"
        else:
            flag = "" if r["within_tolerance"] is None else (" ok" if r["within_tolerance"] else " FAIL")
            de, tol = _fmt_sci(r["dE_vs_ref_Ry"]) + flag, _fmt_sci(r["tolerance_Ry"]) + f" ({PAIR_TAG.get(r['pair'], r['pair'])})"
        if r["mode"] == "reference":
            gemm, gpu, fb = "n/a", "n/a", "n/a"
        elif r["gemm_calls"] is None:
            gemm, gpu, fb = "no counters", "unverifiable", "-"
        else:
            gemm, gpu, fb = str(r["gemm_calls"]), f"{r['gpu_calls']}/{r['gemm_calls']}", str(r["fallbacks"])
        cells = [r["id"], r["binary_short"], r["mode"], _fmt(r["total_energy_Ry"]), de, tol, _fmt(r["scf_iterations"]),
                 _fmt(r["wall_qe_s"], 2), r["timing_grade"], gemm, gpu, fb]
        if with_fs:
            cells += [_fmt_sci(r["max_dF_Ry_bohr"]) if not r["is_ref"] else "ref", _fmt(r["max_dstress_kbar"], 3) if not r["is_ref"] else "ref"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n" + markdown_legend(rows)


def compare(runs: list[dict], ref_id: str | None = None, energy_tol: float | None = None, pins: dict | None = None) -> dict:
    """{ref, rows[], groups[], sections, markdown}: the reference row first, then same-binary rows, then
    same-pw.x/other-dylib rows, then cross-build rows.  sections.same_binary holds only the one-binary A/B
    rows; sections.cross_build holds every other row (each row's `pair` says which kind it is)."""
    if not runs:
        return {"ref": None, "rows": [], "groups": [], "markdown": "", "sections": {"same_binary": [], "cross_build": []}}
    ref_id = select_reference(runs, ref_id, pins)
    ref = next(r for r in runs if r["id"] == ref_id)
    rows = [row_for(r, ref, energy_tol) for r in runs]
    ref_row = [r for r in rows if r["is_ref"]]
    same = [r for r in rows if not r["is_ref"] and r["pair"] == "same_binary"]
    other_dylib = [r for r in rows if not r["is_ref"] and r["pair"] == "same_pw_other_dylib"]
    cross = [r for r in rows if not r["is_ref"] and r["pair"] == "cross_build"]
    ordered = ref_row + same + other_dylib + cross
    return {"ref": ref_id, "rows": ordered, "groups": groups_of(runs), "markdown": to_markdown(ordered, ref_id),
            "sections": {"same_binary": [r["id"] for r in same],
                         "cross_build": [r["id"] for r in other_dylib + cross]}}


def export(result: dict, fmt: str = "md") -> str:
    if fmt == "json":
        return json.dumps(result, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        cols = ["id", "binary_short", "mode", "total_energy_Ry", "dE_vs_ref_Ry", "tolerance_Ry", "within_tolerance", "pair",
                "scf_iterations", "wall_qe_s", "timing_grade", "wall_comparable", "gemm_calls", "gpu_calls", "fallbacks",
                "gpu_verified", "lib_build", "max_dF_Ry_bohr", "max_dstress_kbar"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in result["rows"]:
            w.writerow(r)
        return buf.getvalue()
    return result["markdown"]
