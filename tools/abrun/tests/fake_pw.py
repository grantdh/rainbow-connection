#!/opt/homebrew/bin/python3
"""A fake pw.x for the tests: QE-shaped stdout and apple-bottom-shaped stderr, no solver.

Honours AB_MODE / AB_MIN_GPU_DIM / AB_PROFILE / AB_PROFILE_JSON / AB_REPORT / DYLD_PRINT_LIBRARIES exactly as
the library would, plus ABRUN_FAKE_SCHEMA=old|new (profile JSON v1/v2), ABRUN_FAKE_SLEEP (seconds; SIGINT
terminates it), ABRUN_FAKE_RC (exit status), ABRUN_FAKE_BUILD (the build string), ABRUN_FAKE_DYLIB (the path
the dyld line names), ABRUN_FAKE_IGNORE_SIGINT=1 / ABRUN_FAKE_IGNORE_SIGTERM=1 (a rank that survives the
signal, for the cancel-escalation tests) and ABRUN_FAKE_RANK (set by tests/fake_mpirun.py: only rank 0
prints the QE stdout, every rank prints its own AB_REPORT lines).  Importable: fake_binary() builds the
catalog entries the runner and server use ('ab', 'reference', or 'mpi' = 'ab' launched through
tests/fake_mpirun.py because links.libmpi is set).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BUILD = "apple-bottom 1.4.0 (c5a1fa6a8166)"
ENERGY = -22.83836927
WALL = "2m 1.49s"
CALLS = {"zgemm": 1000, "dgemm": 200}
ABOVE_FLOOR = {"zgemm": 300, "dgemm": 20}
FALLBACK_KEYS = ("no_device", "too_large", "gpu_failed", "other")
STAGE_KEYS = ("init", "alloc", "pack", "upload", "kernel", "download", "epilogue", "other")


def fake_dylib() -> str:
    return os.environ.get("ABRUN_FAKE_DYLIB", str(HERE / "libapplebottom.dylib"))


def fake_binary(kind: str = "ab"):
    """Binary records for the fake: kind 'ab' (id 'fake', dynamic apple-bottom, launched directly), 'reference'
    ('reference-fake'), or 'mpi' ('fake-mpi': 'ab' with links.libmpi set, so launch.plan() chooses the mpirun
    launcher; the tests point launch.MPIRUN at tests/fake_mpirun.py)."""
    from abrun.binaries import Binary  # noqa: WPS433  (import here so the script stays stdlib-only)
    path = str(HERE / "fake_pw.py")
    if kind == "reference":
        return Binary(id="reference-fake", group="reference", path=path, sha256="fake-ref", qe_version="7.4.1",
                      qe_build="QE 7.4.1 (fake)", links={"applebottom": False, "openblas": True, "accelerate": False,
                                                          "fftw3_omp": False, "libmpi": None},
                      ab_symbols={"U": 0, "T": 0}, ab_linkage="none", cls="reference", citable=True,
                      class_reason="fake reference", blas="fake OpenBLAS", label="fake reference | tests/fake_pw.py",
                      launchable=True, solver="pw.x")
    mpi = kind == "mpi"
    return Binary(id="fake-mpi" if mpi else "fake", group="frozen", path=path, sha256="fake-ab", sha_verified=None,
                  qe_version="7.5", qe_build="QE develop 7.5 (fake)",
                  links={"applebottom": True, "openblas": True, "accelerate": False, "fftw3_omp": False,
                         "libmpi": str(HERE / "fake_mpirun.py") if mpi else None},
                  ab_symbols={"U": 2, "T": 0}, ab_linkage="dynamic", rpaths=["@executable_path"], dylib_path=fake_dylib(),
                  dylib_sha256="fake-dylib-sha", ab_build_info=os.environ.get("ABRUN_FAKE_BUILD", DEFAULT_BUILD),
                  features=["AB_DISPATCH_TABLE", "AB_MIN_GPU_DIM", "AB_MODE", "AB_PROFILE", "AB_PROFILE_JSON",
                            "AB_PROFILE_THRESHOLD_NS", "AB_REPORT", "AB_TRACE"],
                  profile_schema="v2", omp1_only=True, cls="experiments", citable=False, class_reason="fake binary",
                  blas="fake apple-bottom", label=f"fake frozen{' (mpi)' if mpi else ''} | tests/fake_pw.py",
                  launchable=True, solver="pw.x")


def counts():
    mode = os.environ.get("AB_MODE", "")
    floor = int(os.environ.get("AB_MIN_GPU_DIM", "32") or 32)
    out = {}
    for name, n in CALLS.items():
        if mode == "gpu":
            gpu = n if floor == 0 else ABOVE_FLOOR[name]
        else:
            gpu = 0
        out[name] = (n, gpu, n - gpu)
    return out


def main() -> int:
    pid = os.getpid()
    env = os.environ
    mode = env.get("AB_MODE")
    build = env.get("ABRUN_FAKE_BUILD", DEFAULT_BUILD)
    if env.get("ABRUN_FAKE_IGNORE_SIGINT") == "1":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if env.get("ABRUN_FAKE_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    rank = int(env.get("ABRUN_FAKE_RANK", "0") or 0)
    nranks = max(1, int(env.get("ABRUN_FAKE_NRANKS", "1") or 1))
    out = print if rank == 0 else (lambda *a, **k: None)      # QE prints from the ionode only
    if env.get("DYLD_PRINT_LIBRARIES") == "1":
        if mode is not None:
            sys.stderr.write(f"dyld[{pid}]: {fake_dylib()}\n")
        sys.stderr.write(f"dyld[{pid}]: /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib\n")
    out("     Program PWSCF v.7.4.1 starts on 16Oct2026 at  0: 0: 0 ")
    out(f"     Parallel version (MPI), running on     {nranks} processors")
    out("     number of k points=     8")
    out("     Dense  grid:    12345 G-vectors     FFT dimensions: (  24,  24,  24)")
    out("          k = 0.0000 0.0000 0.0000 (   411 PWs)   bands (ev):")
    sys.stdout.flush()
    Path("tmp").mkdir(exist_ok=True)
    (Path("tmp") / "si.save").mkdir(exist_ok=True)
    energies = [-22.83762927, -22.83833114, ENERGY]
    accs = ["0.05202059", "0.00028242", "1.1E-09"]
    sleep_total = float(env.get("ABRUN_FAKE_SLEEP", "0") or 0)
    for i, (e, a) in enumerate(zip(energies, accs), 1):
        out(f"     iteration #{i:3d}     ecut=    30.00 Ry     beta= 0.70")
        out(f"     total energy              =     {e:.8f} Ry")
        out(f"     estimated scf accuracy    <       {a} Ry")
        sys.stdout.flush()
        if sleep_total > 0:
            t_end = time.time() + sleep_total / len(energies)
            while time.time() < t_end:
                time.sleep(0.1)
    out("     convergence has been achieved in   3 iterations")
    out("     highest occupied, lowest unoccupied level (ev):     6.2143    6.8449")
    out(f"!    total energy              =     {ENERGY:.8f} Ry")
    out(f"     PWSCF        :      1.23s CPU   {WALL} WALL")
    out("   JOB DONE.")
    sys.stdout.flush()
    rc = int(env.get("ABRUN_FAKE_RC", "0") or 0)
    if mode is None:
        return rc
    c = counts()
    if env.get("AB_PROFILE") == "1":
        for name, (n, gpu, cpu) in c.items():
            for k in range(gpu):
                sys.stderr.write(f"AB_PROFILE {name} M=256 N=8 K=256  init=0.000 alloc=0.010 pack=0.020 upload=0.030 "
                                 f"kernel=0.040 download=0.005 epilogue=0.010 other=0.005  total=0.120 ms\n")
        sys.stderr.write(f"=== {build} BLAS profile, pid {pid} ===\n")
        if sum(n for n, _, _ in c.values()) == 0:
            sys.stderr.write("No BLAS-ABI calls recorded.\n\n")
        jp = env.get("AB_PROFILE_JSON")
        if jp:
            old = env.get("ABRUN_FAKE_SCHEMA", "new") == "old"
            routines = []
            for name, (n, gpu, cpu) in c.items():
                r = {"name": name, "calls": n, "total_ns": n * 120000}
                if old:
                    r.update({"gpu_dispatch_count": gpu, "amx_fallback_count": cpu})
                else:
                    r.update({"gpu_calls": gpu, "cpu_calls": cpu, "gpu_ns": gpu * 120000, "cpu_ns": cpu * 10000,
                              "gpu_stage_ns": {k: gpu * 5000 for k in STAGE_KEYS},
                              "fallbacks": {k: 0 for k in FALLBACK_KEYS}})
                r.update({"max_m": 256, "max_n": 8, "max_k": 256, "avg_m": 256.0, "avg_n": 8.0, "avg_k": 256.0,
                          "dim_histogram": [n, 0, 0, 0, 0, 0, 0, 0], "ns_by_bin": [n * 120000, 0, 0, 0, 0, 0, 0, 0]})
                routines.append(r)
            doc = {"timestamp": "2026-10-16T00:00:00Z", "pid": pid, "threshold_ns": 0, "routines": routines}
            if not old:
                doc = {"build": build, **doc}
            Path(jp).write_text(json.dumps(doc, indent=1))
            sys.stderr.write(f"ab_prof: JSON written to {jp}\n")
    if env.get("AB_REPORT") == "1":
        sys.stderr.write(f"AB_REPORT[{pid}] {build} AB_MODE={mode}\n")
        for name, (n, gpu, cpu) in c.items():
            pct = 100.0 * gpu / n if n else 0.0
            sys.stderr.write(f"AB_REPORT[{pid}] {name} calls={n} gpu_calls={gpu} cpu_calls={cpu} gpu_calls%={pct:.1f} "
                             f"total_ms={n * 0.12:.3f} gpu_ms={gpu * 0.12:.3f} cpu_ms={cpu * 0.01:.3f} gpu_time%={pct:.1f} fallbacks=0\n")
            if gpu:
                sys.stderr.write(f"AB_REPORT[{pid}] {name} gpu_stages_ms init=0.1 alloc=0.2 pack=0.3 upload=0.4 kernel=0.5 "
                                 f"download=0.6 epilogue=0.7 other=0.8\n")
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
