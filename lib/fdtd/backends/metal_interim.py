"""Interim-mode driver shared by the Metal backends (docs/FDTD_BACKEND_SPEC.md section 2).

A backend describes its engine (the Swift program whose `metalCode` kernels to run, the field buffers in kernel
order, the engine grid, the spec -> engine index map, its float32 constants) and calls `run`. This module extracts
the kernels verbatim, builds them with metal_host.swift (cached by a hash of kernel and host), writes the host's
inputs, runs it, and turns its outputs into probes.npz and record.json, with the read-back checks.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from common import MU0, e_probe_peak, git_state, probe_entries, sha256, write_record
from case import Case

HOST = Path(__file__).resolve().parent / "metal_host.swift"
BUILD_ROOT = Path(__file__).resolve().parents[1] / "runs" / "metal" / "bin"
SWIFTC = ["swiftc", "-O", "-framework", "Metal", "-framework", "Foundation"]


@dataclass
class Engine:
    name: str
    repo: Path
    program: Path
    fields: tuple[str, ...]                               # buffer order of the kernels, E components first
    n_e: int
    grid: list[int]                                       # engine grid (Nx, Ny, Nz)
    to_flat: Callable[[str, np.ndarray], np.ndarray]      # spec indices of a component -> engine flat indices
    patches: list[str] = field(default_factory=list)
    patch: Callable[[str], tuple[str, int]] | None = None  # source edit before extracting: text -> (text, count)
    extra_checks: Callable[[np.ndarray], dict] | None = None   # gets the final E buffers (n_e x cells)


def extract_kernel(engine: Engine) -> tuple[str, bytes]:
    data = engine.program.read_bytes()
    text = data.decode()
    if engine.patch:
        text, n = engine.patch(text)
        if n != 1:
            raise RuntimeError(f"patch matched {n} times in {engine.program} (expected 1): the program changed")
    m = re.search(r'let metalCode = """\n(.*?)\n"""', text, re.S)
    if not m:
        raise RuntimeError(f"no metalCode block in {engine.program}")
    return m.group(1), data


def build_host(kernel: str) -> tuple[Path, str, bytes]:
    """Build (once per kernel + host hash) and return (executable, build key, host source bytes).

    Safe when several backends build the same key at once (parallel test runs): each compiles in its own
    temporary directory and moves the finished binary into place with an atomic rename, so a reader sees either
    no binary or a complete one, and the builds that lose the race only replace it with identical bytes."""
    host = HOST.read_bytes()
    key = sha256(kernel.encode() + b"\0" + host)[:16]
    bdir = BUILD_ROOT / key
    exe = bdir / "host"
    if not exe.exists():
        bdir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=BUILD_ROOT))
        try:
            (tmp / "kernel.swift").write_text('let metalCode = """\n' + kernel + '\n"""\n')
            (tmp / "main.swift").write_bytes(host)       # swiftc allows top-level code only in main.swift
            r = subprocess.run(SWIFTC + ["kernel.swift", "main.swift", "-o", "host"], cwd=tmp, capture_output=True,
                               text=True)
            if r.returncode != 0:
                raise RuntimeError(f"swiftc failed:\n{r.stderr[-2000:]}")
            for name in ("kernel.swift", "main.swift", "host"):     # the sources first, the binary last
                os.replace(tmp / name, bdir / name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return exe, key, host


def wave_f32(dt: np.float32):
    """The programs' float32 drive: t = Float(step) * dt, sin(omega t) * exp(-((t - t0) / spread)^2) * amplitude."""
    f = np.float32

    def wave(s: dict, n: np.ndarray) -> np.ndarray:
        w = s["waveform"]
        t = n.astype(f) * dt
        g = np.exp(-(((t - f(w["t0_s"])) / f(w["width_s"])) ** 2)).astype(f)
        return (np.sin(f(2) * f(np.pi) * f(w["f0_Hz"]) * t) * g * f(s["amplitude_V_per_m"])).astype(f)
    return wave


def wave_case(dt: float):
    """The case's drive in float64, rounded to float32 once."""
    def wave(s: dict, n: np.ndarray) -> np.ndarray:
        w = s["waveform"]
        t = n * dt
        v = s["amplitude_V_per_m"] * np.sin(2 * np.pi * w["f0_Hz"] * t) * np.exp(-((t - w["t0_s"]) / w["width_s"]) ** 2)
        return v.astype(np.float32)
    return wave


def case_coefficients(case: Case):
    """(Cb, Db, dt, c, wave) from the case's c and exact SI constants, rounded to float32 once."""
    dt, c = case.dt(), case.c
    eps0 = 1.0 / (MU0 * c * c)
    return np.float32(dt / (eps0 * case.dx)), np.float32(dt / (MU0 * case.dx)), dt, c, wave_case(dt)


def run(case: Case, case_path: Path, run_dir: Path, engine: Engine, coeffs, *, drive_pec: bool) -> dict:
    Cb, Db, dt, c, wave = coeffs
    io = run_dir / "io"
    io.mkdir(parents=True, exist_ok=True)
    kernel, program_bytes = extract_kernel(engine)
    exe, key, host_bytes = build_host(kernel)
    cells = int(np.prod(engine.grid))
    e0 = case.e_comps[0]

    pec = np.zeros(cells, dtype=np.uint8)
    for comp in case.e_comps:                           # a masked cell zeroes all its E: any component marks it
        if len(case.pec_samples[comp]):
            pec[engine.to_flat(comp, case.pec_samples[comp])] = 1
    pec.tofile(io / "pec.u8")

    steps = np.arange(case.steps)
    pec_sets = {c_: {tuple(r) for r in v} for c_, v in case.pec_samples.items()}
    src_rows, amps, skipped = [], [], 0
    for w_i, (s, idx) in enumerate(zip(case.raw.get("sources", []), case.source_samples)):
        amps.append(wave(s, steps))
        comp = s["component"]
        for sp, fi in zip(idx, engine.to_flat(comp, idx)):
            if tuple(sp) in pec_sets[comp] and not drive_pec:
                skipped += 1                              # PEC wins (spec section 3.5)
                continue
            src_rows.append((engine.fields.index(comp), int(fi), w_i))
    np.asarray(src_rows, dtype=np.int32).reshape(-1, 3).tofile(io / "src_idx.i32")
    (np.stack(amps) if amps else np.zeros((0, case.steps), np.float32)).astype(np.float32).tofile(io / "src_amp.f32")

    probe_cols, spans = [], {}
    for p in case.raw.get("probes", []):
        comp, name = p["component"], p["name"]
        start = len(probe_cols)
        probe_cols += [(engine.fields.index(comp), int(x)) for x in engine.to_flat(comp, case.probe_samples[name])]
        spans[name] = (start, len(probe_cols), int(p.get("every", 1)))
    np.asarray(probe_cols, dtype=np.int32).reshape(-1, 2).tofile(io / "probe_idx.i32")
    (io / "config.json").write_text(json.dumps({
        "N": engine.grid, "steps": case.steps, "n_fields": len(engine.fields), "n_e": engine.n_e,
        "Cb_bits": int(np.float32(Cb).view(np.uint32)), "Db_bits": int(np.float32(Db).view(np.uint32))}))

    tic = time.time()
    r = subprocess.run([str(exe), str(io)], capture_output=True, text=True)
    wall_total = time.time() - tic
    if r.returncode != 0:
        raise RuntimeError(f"host failed:\n{r.stdout[-1000:]}\n{r.stderr[-2000:]}")
    timing = json.loads((io / "timing.json").read_text())
    raw = np.fromfile(io / "probes.f32", dtype=np.float32).reshape(case.steps, max(len(probe_cols), 1))
    arrays = {name: raw[::every, a:b].astype(np.float64) for name, (a, b, every) in spans.items()}
    np.savez_compressed(run_dir / "probes.npz", **arrays)

    # Read-back from the final fields: the conductor and every tangential sample of every wall.
    E = np.fromfile(io / "final_E.f32", dtype=np.float32).reshape(engine.n_e, cells)
    (io / "final_E.f32").unlink()
    val = lambda comp, idx: E[engine.fields.index(comp)][engine.to_flat(comp, idx)]
    pec_max = max((float(np.abs(val(cm, case.pec_samples[cm])).max()) for cm in case.e_comps
                   if len(case.pec_samples[cm])), default=0.0)
    wall_max = max(float(np.abs(val(cm, idx)).max()) for a in range(case.dim) for plane in (0, case.n[a])
                   for cm, idx in case.wall_samples(a, plane).items())

    source_check = None                                   # E at the first driven, probed sample after iteration 1
    if src_rows and case.steps > 1:
        for name, (a, b, every) in spans.items():
            col = next((k for k, pc in enumerate(probe_cols[a:b]) if pc == src_rows[0][:2]), None)
            if every == 1 and col is not None:
                got, want = float(arrays[name][1, col]), float(amps[src_rows[0][2]][1])
                source_check = {"iteration": 1, "E_got": got, "E_want": want,
                                "rel_err": abs(got - want) / abs(want) if want else None}
                break

    checks = {"source_equivalence": source_check, "pec_max_abs_E": pec_max, "wall_max_abs_E": wall_max,
              **(engine.extra_checks(E) if engine.extra_checks else {}),
              "probe_e_max_abs_at_end": e_probe_peak(case, arrays)}
    rec = write_record(
        run_dir, case, case_path,
        engine={"name": engine.name, "repo": str(engine.repo), **git_state(engine.repo, engine.program),
                "program": str(engine.program), "program_sha256": sha256(program_bytes),
                "kernel_sha256": sha256(kernel.encode()), "host_sha256": sha256(host_bytes),
                "binary": str(exe), "binary_sha256": sha256(exe.read_bytes()), "build_key": key,
                "patches": ["host: lib/fdtd/backends/metal_host.swift replaces the program's host (kernels verbatim)"]
                           + engine.patches, "precision": "fp32", "device": timing.get("device")},
        realised={"dt_s": dt, "c_m_per_s": c, "grid": engine.grid, "Cb": float(Cb), "Db": float(Db),
                  "source_samples_skipped_on_pec": skipped},
        checks=checks, wall_s=timing["wall_s"], wall_total_s=wall_total,
        throughput_note="case cells x steps / host loop wall; includes the per-step CPU source and probe work, "
                        "as in the program",
        probes=probe_entries(case, {n: e for n, (_, _, e) in spans.items()},
                             {n: list(v.shape) for n, v in arrays.items()}, dt),
        time_labels="row n is read after iteration n (updateH, updateE, then soft sources at t = n dt), labelled t = n dt")
    print(f"{engine.name} case {case.raw['name']}: {case.steps} steps in {timing['wall_s']:.2f} s; "
          f"pec max |E| {pec_max:.1e}; walls {wall_max:.1e}; source check {source_check}")
    return rec
