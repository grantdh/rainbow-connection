"""Standalone checks for lib/fdtd/case.py and the conformance cases (prints PASS/FAIL, exit 1 on any FAIL).

1. The case files reproduce the problems examples/metal-fdtd-crosscheck actually ran: dt, engine grid and
   conductor size are compared with the metadata.txt each engine recorded (skipped when results/ is absent;
   it is gitignored).
2. The conductor in the case files equals the cell set the Swift program's own marking loop produces,
   re-implemented here line by line (LoopFDTD/dipole_s11.swift, "Create Dipole Geometry").
3. Validation refuses malformed cases, and the case hash is stable and content-sensitive.

Run: python3 lib/fdtd/test_case.py
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import testlib  # noqa: E402
from case import Case, CaseError, PEC_CELL_CONVENTIONS  # noqa: E402

CASES, XCHECK = testlib.CASES, testlib.XCHECK
C_SWIFT = 3.0e8                      # c0 in LoopFDTD and MEEPhistopheles dipole_s11.swift

checks = testlib.exlib.Checks()


def check(name: str, ok: bool | None, detail: str) -> None:
    """ok None = reported, not gated."""
    testlib.check(checks, name, ok, detail)


def swift_cells(gap_fixed: bool) -> np.ndarray:
    """The cells LoopFDTD/dipole_s11.swift marks PEC, with Swift's integer arithmetic (its own indices)."""
    nx, ny, nz, dx = 64, 64, 128, np.float32(0.025)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    length = int(np.float32(np.float32(3e8) / np.float32(300e6)) / 2 / dx)    # Int(dipoleLength / dx)
    radius = max(1, int(np.float32(0.002) / dx))
    gap = max(1, int(np.float32(0.01) / dx))
    g = (gap + 1) // 2 if gap_fixed else gap // 2                            # the GAP_FIX edit, or as shipped
    out = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dxg, dyg, dzg = i - cx, j - cy, k - cz
                on_wire = math.sqrt(float(dxg * dxg + dyg * dyg)) <= float(radius)
                upper = on_wire and g <= dzg <= length // 2
                lower = on_wire and -length // 2 <= dzg <= -g
                if upper or lower:
                    out.append((i, j, k))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def case_cells(case: Case) -> np.ndarray:
    """Cells (Swift indices) whose three E samples are PEC in the case, under the e_minus_half layout."""
    shift = PEC_CELL_CONVENTIONS["e_minus_half"]
    sets = [{tuple(r) for r in (case.pec_samples[c] - np.asarray(shift[c]))} for c in ("Ex", "Ey", "Ez")]
    return np.array(sorted(sets[0] & sets[1] & sets[2]), dtype=np.int64).reshape(-1, 3)


def metadata(engine_variant: str) -> dict:
    p = XCHECK / engine_variant / "metadata.txt"
    if not p.exists():
        return {}
    out = {}
    for line in re.split(r"\n|(?<=Ohm)(?=engine)", p.read_text()):
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip().lower()] = v.strip()
    return out


def main() -> int:
    cases = {v: Case.load(CASES / f) for v, f in
             (("empty", "c1-empty-pec-box.json"), ("fixed", "c4-gap-dipole.json"),
              ("shipped", "c4-gap-dipole-closed-gap.json"))}

    # 2. conductor = the Swift program's own marking loop
    for variant, fixed in (("fixed", True), ("shipped", False)):
        want, got = swift_cells(fixed), case_cells(cases[variant])
        same = len(want) == len(got) and np.array_equal(np.unique(want, axis=0), got)
        check(f"{cases[variant].raw['name']}: conductor equals the Swift marking loop", same,
              f"case {len(got)} cells, Swift loop {len(want)} cells")
    gap = np.array([[32, 32, 63]])                                           # spec Ez under the feed (Swift Ez(32,32,64))
    in_pec = {v: bool((cases[v].pec_samples["Ez"] == gap).all(axis=1).any()) for v in cases}
    check("feed-gap Ez sample is free in c4-gap-dipole", not in_pec["fixed"], f"PEC={in_pec['fixed']}")
    check("feed-gap Ez sample is PEC in c4-gap-dipole-closed-gap", in_pec["shipped"], f"PEC={in_pec['shipped']}")
    check("no source sample on the conductor in c4-gap-dipole", cases["fixed"].sources_on_pec() == 0,
          f"{cases['fixed'].sources_on_pec()} overlapping samples")
    check("closed-gap case drives 5 conductor samples (the rod's cross-section)", cases["shipped"].sources_on_pec() == 5,
          f"{cases['shipped'].sources_on_pec()} overlapping samples; PEC wins in the spec, the Swift loop lets the "
          "drive through")
    src = cases["fixed"].raw["sources"][0]["indices"]
    check("source plane is the feed plane", src["lo"][2] == src["hi"][2] == 63, f"source k = {src['lo'][2]}")

    # 1. against what the cross-check recorded
    if not XCHECK.exists():
        check("cross-check metadata", None, f"{XCHECK} absent (gitignored); run the example to enable these checks")
    for variant, case in cases.items():
        for eng in ("loopfdtd", "meephistopheles"):
            md = metadata(f"{eng}_{variant}")
            if not md:
                continue
            grid = [int(x) for x in md["grid"].split("x")]
            check(f"{eng} {variant}: engine grid = case n + 1 (Swift guards one extra plane)",
                  grid == [n + 1 for n in case.n], f"recorded {grid}, case {case.n}")
            if "time step" in md:
                rec = float(md["time step"].split()[0])
                rel = abs(case.dt(C_SWIFT) - rec) / rec
                check(f"{eng} {variant}: dt at the engine's c = 3e8", rel < 1e-5,
                      f"case {case.dt(C_SWIFT):.6e} s, recorded {rec:.6e} s (rel {rel:.1e})")
            if "pec_cells" in md:
                check(f"{eng} {variant}: conductor size", case.pec_count("e_minus_half") == int(md["pec_cells"]),
                      f"case {case.pec_count('e_minus_half')} cells, recorded {md['pec_cells']}")
        md = metadata(f"meep_{variant}")
        if md:
            rec = float(md["dt_s"])
            check(f"meep {variant}: dt at c = 299792458", abs(case.dt() - rec) / rec < 1e-12,
                  f"case {case.dt():.15e} s, recorded {rec:.15e} s")
            grid = [int(x) for x in md["grid"].split("x")]
            check(f"meep {variant}: box solved by the MEEP mirror vs the case box", None,
                  f"MEEP {grid} cells = {[round(n * case.dx, 4) for n in grid]} m; case {case.n} = "
                  f"{[round(b, 4) for b in case.box_m]} m (the mirror solves N dx, the Swift programs (N-1) dx)")

    # 2-D TM cases (MEEPhistopheles' feed plane k = 64): its kernel layout equals the spec's 2-D layout, so its
    # (i, j) are spec indices. The cross-section of the Swift marking loop at k = 64 is the closed-gap conductor.
    c2 = {v: Case.load(CASES / f) for v, f in (("empty", "c2d-empty-pec-box.json"), ("cross", "c2d-pec-cross.json"))}
    plane = {(int(i), int(j)) for i, j, k in swift_cells(False) if k == 64}
    got = {tuple(int(x) for x in r) for r in c2["cross"].pec_samples["Ez"]}
    check("c2d-pec-cross: conductor equals the Swift loop's k = 64 plane (closed gap)", got == plane,
          f"case {len(got)} cells, plane {len(plane)}")
    fixed_plane = {(int(i), int(j)) for i, j, k in swift_cells(True) if k == 64}
    check("gap-fixed dipole leaves no conductor in the feed plane (so 'fixed' = c2d-empty)", not fixed_plane,
          f"{len(fixed_plane)} cells at k = 64")
    check("c2d-pec-cross drives 5 conductor samples", c2["cross"].sources_on_pec() == 5,
          f"{c2['cross'].sources_on_pec()} overlapping")
    for v in ("empty", "fixed", "shipped"):
        md = metadata(f"meephistopheles_{v}")
        if md:
            grid = [int(x) for x in md["grid"].split("x")]
            check(f"meephistopheles {v}: plane grid = 2-D case n + 1", grid[:2] == [n + 1 for n in c2["empty"].n],
                  f"recorded {grid}, case {c2['empty'].n}")
    raw2 = c2["empty"].raw
    for label, mutate in {
        "2-D: Ex source": lambda r: r["sources"][0].__setitem__("component", "Ex"),
        "2-D: courant above 1/sqrt(2)": lambda r: r["time"].__setitem__("courant", 0.72),
        "2-D: 3-wide indices": lambda r: r["probes"][0].__setitem__("indices", [[30, 30, 1]]),
        "2-D: z faces": lambda r: r["boundaries"].__setitem__("z-", "pec"),
    }.items():
        r = copy.deepcopy(raw2); mutate(r)
        try:
            Case(r); check(f"refuses: {label}", False, "accepted")
        except CaseError as e:
            check(f"refuses: {label}", True, str(e)[:90])

    # 3. validation and hashing
    raw = cases["fixed"].raw
    check("hash is stable on reload", Case.load(CASES / "c4-gap-dipole.json").sha256 == cases["fixed"].sha256,
          cases["fixed"].sha256[:16])
    bumped = copy.deepcopy(raw); bumped["time"]["steps"] += 1
    check("hash changes with content", Case(bumped).sha256 != cases["fixed"].sha256, "steps + 1")
    reordered = json.loads(json.dumps(raw, sort_keys=True))
    check("hash ignores key order", Case(reordered).sha256 == cases["fixed"].sha256, "sorted keys")
    bad_cases = {
        "courant above 1/sqrt(3)": lambda r: r["time"].__setitem__("courant", 0.6),
        "Ez index past the grid": lambda r: r["probes"][0]["indices"].__setitem__("hi", [34, 34, 127]),
        "unknown pec_cells convention": lambda r: r["regions"][0].__setitem__("convention", "cell_centre"),
        "reserved probe kind": lambda r: r["probes"][0].__setitem__("kind", "dft_plane"),
        "missing boundary face": lambda r: r["boundaries"].pop("z+"),
        "duplicate probe name": lambda r: r["probes"][1].__setitem__("name", "gap_ez"),
    }
    for label, mutate in bad_cases.items():
        r = copy.deepcopy(raw); mutate(r)
        try:
            Case(r); check(f"refuses: {label}", False, "accepted")
        except CaseError as e:
            check(f"refuses: {label}", True, str(e)[:90])

    return testlib.finish(checks)


if __name__ == "__main__":
    sys.exit(main())
