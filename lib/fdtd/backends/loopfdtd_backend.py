"""LoopFDTD backend for MUPPET FDTD cases, interim (code-generation) mode of docs/FDTD_BACKEND_SPEC.md.

    python3 lib/fdtd/backends/loopfdtd_backend.py CASE.json RUN_DIR [--guard-fix] [--engine-c] [--swift-source-order]

Writes RUN_DIR/record.json and RUN_DIR/probes.npz. Exit 3 means the case needs a feature this backend refuses.

What runs: the two Metal kernels (updateE, updateH) of ~/Dev/Claude/LoopFDTD/dipole_s11.swift, extracted verbatim at
build time (nothing is copied into this repo or written into LoopFDTD), driven by metal_host.swift in place of the
program's own host loop (metal_interim.py). The record lists every deviation from the shipped program:
  - always:               "host": the case-driven host replaces the program's host code (kernels verbatim)
  - --guard-fix:          the per-component boundary guard (loopfdtd_patches.guard_fix), so the far-wall normal
                          E is updated; without it the kernel is as shipped
  - --engine-c:           the program's float32 constants (c0 = 3e8, mu0 = 4e-7 pi) and float32 waveform
                          arithmetic, to reproduce the program's recorded runs; by default the host uses the
                          case's c, so dt and Cb/Db match every other engine
  - --swift-source-order: also drive source samples that sit on the conductor (the program's behaviour); by
                          default PEC wins, as the spec says

Index maps (from the kernels' stencils; lib/fdtd/test_case.py checks the conductor map against the program's
own marking loop). LoopFDTD runs n + 1 cells per axis and holds tangential E at nodes 0 and n, so it solves the
case box of n dx. Spec -> LoopFDTD index: Ex +x, Ey +y, Ez +z; Hx +y+z, Hy +x+z, Hz +x+y (one cell each). The E
shifts are the negated e_minus_half convention of case.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from case import Case, CaseError, E_COMPONENTS, H_COMPONENTS, PEC_CELL_CONVENTIONS  # noqa: E402
from common import check_capabilities, refuse  # noqa: E402
from loopfdtd_patches import GUARD_FIX_LABEL, guard_fix  # noqa: E402
import metal_interim as mi  # noqa: E402

LOOP = Path.home() / "Dev/Claude/LoopFDTD"
PROGRAM = LOOP / "dipole_s11.swift"
FIELDS = E_COMPONENTS + H_COMPONENTS
SHIFT = {**{c: tuple(-x for x in s) for c, s in PEC_CELL_CONVENTIONS["e_minus_half"].items()},
         "Hx": (0, 1, 1), "Hy": (1, 0, 1), "Hz": (1, 1, 0)}


def engine_c_coefficients(case: Case):
    """The program's float32 arithmetic, operation for operation (dipole_s11.swift lines 22-42, 263-264)."""
    f = np.float32
    c0, mu0 = f(3e8), f(4e-7) * f(np.pi)
    eps0 = f(1) / (mu0 * c0 * c0)
    dx = f(case.dx)
    dt = dx / (c0 * np.sqrt(f(3.0))) * f(0.99)
    return f(dt / (eps0 * dx)), f(dt / (mu0 * dx)), float(dt), 3e8, mi.wave_f32(dt)


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) != 2 or flags - {"--guard-fix", "--engine-c", "--swift-source-order"}:
        print(__doc__)
        return 2
    case_path, run_dir = Path(pos[0]), Path(pos[1])
    try:
        case = Case.load(case_path)
    except CaseError as e:
        print(f"INVALID CASE: {e}", file=sys.stderr)
        return 2
    check_capabilities(case, "LoopFDTD (dipole_s11 kernels)", dims={3}, precisions={"fp32"})
    # The kernel zeroes all three E components of a masked cell: the conductor must be whole e_minus_half cells.
    cells = [{tuple(x) for x in case.pec_samples[c] + np.asarray(SHIFT[c])} for c in E_COMPONENTS]
    if not (cells[0] == cells[1] == cells[2]):
        refuse("conductor is not a union of whole LoopFDTD cells (pec_cells, convention e_minus_half)")
    if "--engine-c" in flags and abs(case.courant - 0.99 / math.sqrt(3)) > 1e-12:
        refuse("--engine-c reproduces the program's dt = 0.99 dx / (c0 sqrt 3); the case's Courant differs")

    N = [k + 1 for k in case.n]
    patches = []
    if "--guard-fix" in flags:
        patches.append(GUARD_FIX_LABEL)
    if "--engine-c" in flags:
        patches.append("engine_c: the program's float32 constants (c0 = 3e8) and float32 waveform arithmetic")
    if "--swift-source-order" in flags:
        patches.append("swift_source_order: sources also drive conductor samples")
    engine = mi.Engine(
        name="loopfdtd", repo=LOOP, program=PROGRAM, fields=FIELDS, n_e=3, grid=N,
        to_flat=lambda comp, idx: (lambda li: li[:, 0] + li[:, 1] * N[0] + li[:, 2] * N[0] * N[1])(
            np.asarray(idx, dtype=np.int64) + np.asarray(SHIFT[comp])),
        patches=patches, patch=guard_fix if "--guard-fix" in flags else None)
    coeffs = engine_c_coefficients(case) if "--engine-c" in flags else mi.case_coefficients(case)
    run_dir.mkdir(parents=True, exist_ok=True)
    mi.run(case, case_path, run_dir, engine, coeffs, drive_pec="--swift-source-order" in flags)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
