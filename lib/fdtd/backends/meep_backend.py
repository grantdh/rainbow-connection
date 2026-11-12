"""MEEP backend for MUPPET FDTD cases (docs/FDTD_BACKEND_SPEC.md). Run with the meep_arm interpreter:

    $MUPPET_MEEP_PYTHON lib/fdtd/backends/meep_backend.py CASE.json RUN_DIR

Writes RUN_DIR/record.json and RUN_DIR/probes.npz. Exit 3 means the case needs a feature this backend refuses
(nothing is run); any other non-zero exit is a failure.

How a case maps onto MEEP (each point was checked on a small grid; see lib/fdtd/test_meep_backend.py):
- Length unit a = dx, so resolution = 1, every Yee sample sits on an integer or half-integer coordinate, and a
  point source carries no 1/dV factor. MEEP's staggering equals the spec's (Ez at (i, j, k+1/2), and so on).
- The MEEP cell is centred on the origin with no boundary layers. Its grid lines up with the metallic cell
  surface only when the cell count along an axis is even: with an odd count the nodes sit half a cell off the
  walls, and a point source spreads over two samples. (geometry_center does not help; with it, sources and
  readback disagree by one cell.) So each odd axis gets one extra MEEP cell, and a metal slab covers spec
  x >= n - 1/4. That zeroes the tangential E on the plane x = n and leaves the normal E at n - 1/2 free: a PEC wall
  at exactly n dx. Spec position p sits at MEEP p - m/2, where m is the MEEP cell count.
- pec_samples: one mp.metal block of side 1/2 centred on each sample. With eps_averaging off, MEEP samples
  epsilon at each component's own position, so the block zeroes that one E sample and no other.
- soft_e: MEEP has no field setter, but its D update adds -dt J(t_n + dt/2) at a point source. So a current
  J(tau) = -A s(n dt) / dt_meep on the exact sample reproduces "E += A s(n dt) after the E update of
  iteration n" to rounding. The run checks this at iteration 1, where E at the source must equal A s(dt).
- Probes are read with get_field_point at the exact sample position, where MEEP's interpolation weight is 1.
  E is stored in V/m (unchanged); H is converted from MEEP's units (H_meep = Z0 H) to A/m.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case import Case, CaseError, position  # noqa: E402
from common import (MU0, check_capabilities, e_probe_peak, probe_entries, refuse, sha256,  # noqa: E402
                    write_record)

import meep as mp  # noqa: E402


def waveform(w: dict):
    """The drive s(t) in float64 scalar arithmetic (math, not numpy: kept as validated)."""
    f0, t0, width = w["f0_Hz"], w["t0_s"], w["width_s"]
    return lambda t: math.sin(2 * math.pi * f0 * t) * math.exp(-((t - t0) / width) ** 2)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    case_path, run_dir = Path(argv[0]), Path(argv[1])
    try:
        case = Case.load(case_path)
    except CaseError as e:
        print(f"INVALID CASE: {e}", file=sys.stderr)
        return 2
    precision = "fp32" if mp.is_single_precision() else "fp64"
    check_capabilities(case, f"MEEP {mp.__version__} ({precision}; meep-single is the fp32 build)",
                       dims={2, 3}, precisions={precision})
    run_dir.mkdir(parents=True, exist_ok=True)
    mp.verbosity(0)

    n, dim, ecomps = case.n, case.dim, case.e_comps
    dt_phys = case.dt()
    dt_m = case.courant                                   # a = dx, c = 1: dt = Courant dx / c
    m = [k + (k % 2) for k in n]                          # MEEP cell counts: even on every axis
    shift = np.array(m, dtype=float) / 2
    vec = lambda p: mp.Vector3(*[float(x) for x in np.asarray(p, dtype=float) - shift])
    pad = (lambda v: v) if dim == 3 else (lambda v: list(v) + [mp.inf])   # 2-D blocks extend through z

    geometry = []
    for a in range(dim):
        if m[a] != n[a]:                                  # far wall of an odd axis: slab from n - 1/4 to the cell edge
            lo, hi = n[a] - 0.25, m[a] + 0.5
            size = [m[b] + 1.0 for b in range(dim)]; size[a] = hi - lo
            centre = [m[b] / 2 for b in range(dim)]; centre[a] = (lo + hi) / 2
            geometry.append(mp.Block(size=mp.Vector3(*pad(size)), center=vec(centre), material=mp.metal))
    for comp in ecomps:
        for p in position(comp, case.pec_samples[comp]):
            geometry.append(mp.Block(size=mp.Vector3(*pad([0.5] * dim)), center=vec(p), material=mp.metal))

    sources, src_checks = [], []
    for s, idx in zip(case.raw.get("sources", []), case.source_samples):
        comp = getattr(mp, s["component"])
        amp, s_of_t = float(s["amplitude_V_per_m"]), waveform(s["waveform"])

        # MEEP evaluates the current at tau = (n + 1/2) dt_m for the D update of step n (and at (n + 1) dt_m for
        # bookkeeping that does not touch E in vacuum); floor() maps both to the iteration they belong to.
        def j_of_tau(tau, amp=amp, s_of_t=s_of_t):
            it = math.floor(tau / dt_m + 1e-6)
            return -amp * s_of_t(it * dt_phys) / dt_m

        for p in position(s["component"], idx):
            sources.append(mp.Source(mp.CustomSource(src_func=j_of_tau), component=comp, center=vec(p)))
            src_checks.append((comp, vec(p), amp * s_of_t(1 * dt_phys)))

    sim = mp.Simulation(cell_size=mp.Vector3(*[float(k) for k in m]), resolution=1, dimensions=dim,
                        boundary_layers=[], Courant=case.courant, eps_averaging=False,
                        geometry=geometry, sources=sources)
    sim.init_sim()

    z0 = MU0 * case.c
    probes = []                                           # per probe: MEEP component, points, stride, scale, rows
    for p in case.raw.get("probes", []):
        comp = p["component"]
        probes.append({"name": p["name"], "mp": getattr(mp, comp), "every": int(p.get("every", 1)),
                       "pts": [vec(x) for x in position(comp, case.probe_samples[p["name"]])],
                       "scale": 1.0 if comp in ecomps else 1.0 / z0, "rows": []})

    source_check = None
    tic = time.time()
    for it in range(case.steps):
        sim.fields.step()
        if sim.fields.t != it + 1:
            raise RuntimeError(f"MEEP advanced {sim.fields.t} steps after iteration {it}")
        if it == 1 and src_checks:
            comp, pt, want = src_checks[0]
            got = sim.get_field_point(comp, pt).real
            source_check = {"iteration": 1, "E_got": got, "E_want": want,
                            "rel_err": abs(got - want) / abs(want) if want else None}
        for pr in probes:
            if it % pr["every"] == 0:
                pr["rows"].append([sim.get_field_point(pr["mp"], q).real * pr["scale"] for q in pr["pts"]])
    wall = time.time() - tic

    # Read-back at the end: every PEC sample, and tangential E on a 7 x 7 lattice of each wall (all of them would
    # be ~10^5 reads in 3-D).
    read = lambda comp, idx: max((abs(sim.get_field_point(getattr(mp, comp), vec(p)).real)
                                  for p in position(comp, idx)), default=0.0)
    wall_max = max(read(cm, idx) for a in range(dim) for plane in (0, n[a])
                   for cm, idx in case.wall_samples(a, plane, lattice=7).items())
    pec_max = max(read(cm, case.pec_samples[cm]) for cm in ecomps)

    arrays = {pr["name"]: np.asarray(pr["rows"], dtype=np.float64) for pr in probes}
    np.savez_compressed(run_dir / "probes.npz", **arrays)
    so = Path(mp.__file__).parent / "_meep.so"
    write_record(
        run_dir, case, case_path,
        engine={"name": "meep", "version": mp.__version__, "python": sys.executable,
                "binary": str(so), "binary_sha256": sha256(so.read_bytes()) if so.exists() else None,
                "patches": [], "precision": precision,
                "mapping": "a = dx, resolution 1, centred even-count cell, metal cell boundary plus a metal "
                           "slab beyond n on odd axes; pec_samples as 1/2-cell mp.metal blocks, eps_averaging "
                           "off; soft_e as an equivalent point current"},
        realised={"dt_s": dt_phys, "c_m_per_s": case.c, "grid": n, "meep_cell": m},
        checks={"source_equivalence": source_check, "pec_max_abs_E": pec_max, "wall_max_abs_E": wall_max,
                "probe_e_max_abs_at_end": e_probe_peak(case, arrays)},
        wall_s=wall, throughput_note="includes per-step Python probe reads",
        probes=probe_entries(case, {pr["name"]: pr["every"] for pr in probes},
                             {k: list(v.shape) for k, v in arrays.items()}, dt_phys),
        time_labels="row n is read after iteration n (H then E update, then soft sources at t = n dt), "
                    "labelled t = n dt as in the Swift programs; H samples lead E by dt/2 in leapfrog time")
    print(f"MEEP case {case.raw['name']}: {case.steps} steps in {wall:.1f} s; pec max |E| {pec_max:.2e}; "
          f"source check {source_check}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
