"""MEEP 3D mirror of the Swift/Metal dipole_s11 programs (PEC-walled box, gap-fed dipole).

--empty removes the conductor: a point-ish soft source in a 1.6 x 1.6 x 3.2 m PEC cavity, whose TM_mnp
resonances are analytic (m, n odd and p even are excited from the centre).

Run with the meep_arm interpreter:  python meep_dipole_box.py out_dir [--as-shipped] [--empty] [--steps N]
Writes out_dir/timedomain.txt (time, voltage-proxy, current-proxy) and metadata.txt in the same
format as LoopFDTD/dipole_s11.swift, so analyze.py treats all engines alike.

The geometry reproduces the Swift cell-marking rule exactly (cell centres, not nominal radii):
  64 x 64 x 128 cells of dx = 2.5 cm, arms are the 5-cell "+" cross-section
  (sqrt(di^2 + dj^2) <= 1) at k = 64 +/- dz, dz in [1, 10] and [-10, -1] (one-cell gap at dz = 0).
  --as-shipped reproduces the integer-division bug (gapCells/2 == 0): dz in [0, 10] and [-10, 0],
  i.e. a closed 21-cell rod.
Source: soft Ez injection over the 5 x 5 cells around the gap, s(t) = sin(2 pi f0 t) exp(-((t-t0)/w)^2),
f0 = 300 MHz, t0 = 10/f0, w = 3/f0. dt = 0.99 dx / (c sqrt 3), 2000 steps. Metallic box walls.
Units: a = 1 m, c = 1.
"""

import sys
import time
from pathlib import Path

import numpy as np
import meep as mp

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
as_shipped = "--as-shipped" in sys.argv
empty = "--empty" in sys.argv
c0 = 299792458.0
f0 = 300e6 / c0                       # 1.0 in MEEP units
dx = 0.025
Nx, Ny, Nz = 64, 64, 128
nsteps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 2000
courant = 0.99 / np.sqrt(3.0)
dt = courant * dx
half = 10                             # dipoleLengthCells/2 with dipoleLengthCells = Int(0.5/0.025) = 20
dz_lo = 0 if as_shipped else 1
cell = mp.Vector3(Nx * dx, Ny * dx, Nz * dx)
off = 0.5 * dx                        # Swift cell (32, 32, 64) is centred half a cell from the box centre
cross = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
arm_len = (half - dz_lo + 1) * dx
arm_zc = ((dz_lo + half) / 2 + 0.5) * dx
geometry = []
for di, dj in ([] if empty else cross):
    for sign in (+1, -1):
        zc = sign * ((dz_lo + half) / 2) * dx + off
        geometry.append(mp.Block(size=mp.Vector3(dx, dx, arm_len), center=mp.Vector3(di * dx + off, dj * dx + off, zc),
                                 material=mp.metal))
spread_periods = float(sys.argv[sys.argv.index("--spread") + 1]) if "--spread" in sys.argv else 3.0   # envelope spread in periods (Swift: 3)
t0_src, spread = 10.0 / f0, spread_periods / f0


def src_func(t):
    return np.sin(2 * np.pi * f0 * t) * np.exp(-((t - t0_src) / spread) ** 2)


patch_center = mp.Vector3(off, off, off)
patch_size = mp.Vector3(5 * dx, 5 * dx, 0)
sources = [mp.Source(mp.CustomSource(src_func=src_func, end_time=nsteps * dt), component=mp.Ez,
                     center=patch_center, size=patch_size)]
sim = mp.Simulation(cell_size=cell, resolution=1 / dx, geometry=geometry, sources=sources,
                    boundary_layers=[], Courant=courant, eps_averaging=False)
sim.init_sim()
patch = mp.Volume(center=patch_center, size=patch_size)

times, volts, amps = [], [], []
tic = time.time()
for step in range(nsteps):
    # One Yee step per sample. sim.run(until=dt) is NOT suitable: it compares a float32-rounded clock and
    # advances one or two steps per call, which makes the time axis non-uniform and every FFT peak wrong.
    sim.fields.step()
    assert sim.fields.t == step + 1, f"MEEP advanced {sim.fields.t} steps after {step + 1} samples"
    ez = sim.get_array(component=mp.Ez, vol=patch)
    hx = sim.get_array(component=mp.Hx, vol=patch)
    hy = sim.get_array(component=mp.Hy, vol=patch)
    v = float(np.mean(ez))
    cur = float(np.mean((hx[1:, 1:] - hx[1:, :-1]) - (hy[1:, 1:] - hy[:-1, 1:]))) * dx
    times.append(sim.fields.time() / c0)        # double precision: fields.t * dt
    volts.append(v)
    amps.append(cur)
    if step % 400 == 0:
        print(f"  step {step}/{nsteps} V={v:.3e} I={cur:.3e}", flush=True)
elapsed = time.time() - tic
mcells = Nx * Ny * Nz * nsteps / elapsed / 1e6

with open(out / "timedomain.txt", "w") as fh:
    fh.write("# time(s) voltage(V) current(A)\n")
    for t, v, i in zip(times, volts, amps):
        fh.write(f"{t} {v} {i}\n")
(out / "metadata.txt").write_text(
    f"engine: MEEP {mp.__version__} (CPU, FP64={not mp.is_single_precision()}){' as-shipped geometry' if as_shipped else ''}{' empty box' if empty else ''}\n"
    f"grid: {Nx} x {Ny} x {Nz}\ncell_size_m: {dx}\ndt_s: {dt / c0}\nsteps: {nsteps}\nspread_periods: {spread_periods}\npec_blocks: {len(geometry)}\n"
    f"gap_cells: {0 if as_shipped else 1}\nelapsed_s: {elapsed:.3f}\nthroughput_Mcells_per_s: {mcells:.1f}\n"
    f"boundaries: PEC box (no PML), like the Swift programs\n")
print(f"MEEP dipole box done: {elapsed:.2f} s, {mcells:.0f} Mcells/s")
