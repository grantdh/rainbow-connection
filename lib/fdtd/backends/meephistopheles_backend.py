"""MEEPhistopheles backend for 2-D TM MUPPET FDTD cases, interim (code-generation) mode of docs/FDTD_BACKEND_SPEC.md.

    python3 lib/fdtd/backends/meephistopheles_backend.py CASE.json RUN_DIR [--engine-c] [--swift-source-order]

Writes RUN_DIR/record.json and RUN_DIR/probes.npz. Exit 3 means the case needs a feature this backend refuses.

What runs: the updateE/updateH kernels of ~/Dev/Claude/MEEPhistopheles/src/metal/dipole_s11.swift, extracted
verbatim at build time, driven by metal_host.swift in place of the program's host loop (metal_interim.py).

Why only 2-D: the kernels evolve Ez, Hx and Hy with no z-derivative, so every z-plane is an independent 2-D TM
problem. The program's 64 x 64 x 128 runs are 128 such planes, of which only the driven feed plane (k = 64) is not
zero. The host runs one plane: Nz = 3 (the kernels skip k = 0 and Nz - 1) with the case in plane k = 1.
Index map: the kernels put Ez(i,j) at (i, j), Hx(i,j) at (i, j+1/2), Hy(i,j) at (i+1/2, j): the spec's 2-D layout,
so spec and engine (i, j) are the same. The kernels hold Ez at nodes 0 and N-1, so N = n + 1 solves the case box.

Recorded deviations (record.engine.patches):
  - always:               the case-driven host replaces the program's host (kernels verbatim), one plane
  - --engine-c:           the program's float32 constants (c0 = 3e8 in dt, eps0 = 8.854e-12, mu0 = 4 pi 1e-7) and
                          float32 waveform, to reproduce its recorded runs; by default the case's c and exact eps0
  - --swift-source-order: also drive source samples on the conductor (the program's behaviour)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from case import Case, CaseError, TM_E, TM_H  # noqa: E402
from common import check_capabilities, refuse  # noqa: E402
import metal_interim as mi  # noqa: E402

REPO_DIR = Path.home() / "Dev/Claude/MEEPhistopheles"
PROGRAM = REPO_DIR / "src/metal/dipole_s11.swift"


def engine_c_coefficients(case: Case):
    """The program's float32 arithmetic, operation for operation (dipole_s11.swift lines 22-41 and the Cb/Db lines).
    Its eps0 is the literal 8.854e-12, so its wave speed is 1/sqrt(mu0 eps0) = 2.997956e8, not the 3e8 in dt."""
    f = np.float32
    c0, mu0, eps0 = f(3e8), f(4.0) * f(np.pi) * f(1e-7), f(8.854e-12)
    dx = f(case.dx)
    dt = dx / (c0 * np.sqrt(f(3.0))) * f(0.99)
    c_eff = float(1 / np.sqrt(float(mu0) * float(eps0)))
    return f(dt / (eps0 * dx)), f(dt / (mu0 * dx)), float(dt), c_eff, mi.wave_f32(dt)


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) != 2 or flags - {"--engine-c", "--swift-source-order"}:
        print(__doc__)
        return 2
    case_path, run_dir = Path(pos[0]), Path(pos[1])
    try:
        case = Case.load(case_path)
    except CaseError as e:
        print(f"INVALID CASE: {e}", file=sys.stderr)
        return 2
    check_capabilities(case, "MEEPhistopheles (dipole_s11 kernels, 2-D TM)", dims={2}, precisions={"fp32"})
    if "--engine-c" in flags and abs(case.courant - 0.99 / np.sqrt(3)) > 1e-12:
        refuse("--engine-c reproduces the program's dt = 0.99 dx / (c0 sqrt 3); the case's Courant differs")

    N = [case.n[0] + 1, case.n[1] + 1, 3]
    plane = N[0] * N[1]                                    # data plane k = 1

    def to_flat(comp, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return idx[:, 0] + idx[:, 1] * N[0] + plane

    def other_planes(E):                                   # every plane but k = 1 must stay exactly zero
        return {"max_abs_E_outside_data_plane": float(np.abs(np.delete(E[0], np.arange(plane, 2 * plane))).max())}

    patches = ["one plane, Nz = 3 (the kernels have no z-derivative)"]
    if "--engine-c" in flags:
        patches.append("engine_c: the program's float32 constants (c0 = 3e8, eps0 = 8.854e-12) and float32 waveform")
    if "--swift-source-order" in flags:
        patches.append("swift_source_order: sources also drive conductor samples")
    engine = mi.Engine(name="meephistopheles", repo=REPO_DIR, program=PROGRAM, fields=TM_E + TM_H, n_e=1, grid=N,
                       to_flat=to_flat, patches=patches, extra_checks=other_planes)
    coeffs = engine_c_coefficients(case) if "--engine-c" in flags else mi.case_coefficients(case)
    run_dir.mkdir(parents=True, exist_ok=True)
    mi.run(case, case_path, run_dir, engine, coeffs, drive_pec="--swift-source-order" in flags)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
