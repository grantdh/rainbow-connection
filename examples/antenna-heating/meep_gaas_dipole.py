"""MEEP 3D: a centre-fed dipole made of a lossy Drude conductor (doped GaAs in the mid-infrared).

Units: 1 MEEP length unit = 1 um, c = 1, so f[MEEP] = f[THz] / 299.792458; Z0 = 1.
Geometry: two square rods (side w) along z, total length L, separated by a vacuum gap g at the centre.
The first `pad` of each arm next to the gap is a perfect conductor (the feed terminals) and the rest is
the Drude material; a z-directed current source fills the gap between the terminals (delta-gap feed).
The terminals keep the source's singular near field off the lossy material: the first version drove the
GaAs directly, and half of the computed heat then sat within 0.3 um of the source and grew with
resolution, so the radiation efficiency did not converge (0.073 at 10 px/um, 0.015 at 15).
Material: eps(f) = eps_inf - fp^2 / (f^2 + i f gamma) (MEEP DrudeSusceptibility, frequency 1, sigma = fp^2).

Recorded (MEEP DFT convention: flux = Re(E* x H) and absorbed density = omega Im(eps) |E|^2, no factor 1/2):
  - P_rad(f): net flux through a closed box around the antenna;
  - the complex power S(f) leaving a closed box around the gap whose end faces lie inside the terminals
    (sum of E* x H . n dA; the metal carries no flux), and the terminal current I(f) = Ampere loop of H
    around the upper terminal at two loop sizes; Z_in = S / |I|^2 and P_in = Re(S);
  - at nheat frequencies the absorbed power omega Im(E* . D) on MEEP's own Yee grid (yee_heat.py): its
    volume sum (x2 for the lower arm) and the heat per unit length q(z);
    near2far U(theta) in the phi = 0 plane.
The cell is a whole, even number of pixels per side, so the cell centre is a grid point. Mirror planes
x, y (phase +1) and z (phase -1) cut the work by eight.

Run with the meep_arm interpreter:
    python meep_gaas_dipole.py OUT_DIR --fp FP --gamma G --eps-inf E [--length 5] [--width 0.6] [--gap 0.2]
        [--pad 0.2] [--fmin-thz 15] [--fmax-thz 40] [--resolution 10] [--nfreq 51] [--nheat 11]
Writes OUT_DIR/antenna.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import meep as mp

from yee_heat import add_heat_region, slab_heat

THZ = 1 / 299.792458

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--fp", type=float, required=True, help="unscreened plasma frequency, MEEP units")
p.add_argument("--gamma", type=float, required=True, help="Drude damping as an ordinary frequency, MEEP units")
p.add_argument("--eps-inf", type=float, required=True)
p.add_argument("--length", type=float, default=5.0)
p.add_argument("--width", type=float, default=0.6)
p.add_argument("--gap", type=float, default=0.2)
p.add_argument("--pad", type=float, default=0.2, help="length of each perfect-conductor feed terminal (um)")
p.add_argument("--fmin-thz", type=float, default=15.0)
p.add_argument("--fmax-thz", type=float, default=40.0)
p.add_argument("--resolution", type=float, default=10.0)
p.add_argument("--nfreq", type=int, default=51)
p.add_argument("--nheat", type=int, default=11)
p.add_argument("--pml", type=float, default=5.0)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

fmin, fmax = a.fmin_thz * THZ, a.fmax_thz * THZ
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
px = 1 / a.resolution
arm = (a.length - a.gap) / 2
g0 = a.gap / 2 + a.pad                                          # where the GaAs starts
margin = 1.0
bx, bz = a.width / 2 + margin, a.length / 2 + margin            # flux / near2far box half-sizes
air = 1.0
nx, nz = (int(np.ceil((b + air + a.pml) / px)) for b in (bx, bz))
sx, sz = 2 * nx * px, 2 * nz * px
cell = mp.Vector3(sx, sx, sz)
drude = mp.Medium(epsilon=a.eps_inf, E_susceptibilities=[mp.DrudeSusceptibility(frequency=1.0, gamma=a.gamma, sigma=a.fp ** 2)])
geometry = []
for s in (+1, -1):
    geometry.append(mp.Block(size=mp.Vector3(a.width, a.width, a.pad), center=mp.Vector3(z=s * (a.gap / 2 + a.pad / 2)), material=mp.metal))
    geometry.append(mp.Block(size=mp.Vector3(a.width, a.width, arm - a.pad), center=mp.Vector3(z=s * (g0 + (arm - a.pad) / 2)),
                             material=drude))
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3(),
                     size=mp.Vector3(a.width, a.width, a.gap))]
sim = mp.Simulation(resolution=a.resolution, cell_size=cell, boundary_layers=[mp.PML(a.pml)], geometry=geometry,
                    sources=sources, symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)])

faces = [(mp.Vector3(x=+bx), mp.Vector3(0, 2 * bx, 2 * bz), +1), (mp.Vector3(x=-bx), mp.Vector3(0, 2 * bx, 2 * bz), -1),
         (mp.Vector3(y=+bx), mp.Vector3(2 * bx, 0, 2 * bz), +1), (mp.Vector3(y=-bx), mp.Vector3(2 * bx, 0, 2 * bz), -1),
         (mp.Vector3(z=+bz), mp.Vector3(2 * bx, 2 * bx, 0), +1), (mp.Vector3(z=-bz), mp.Vector3(2 * bx, 2 * bx, 0), -1)]
freqs = np.linspace(fmin, fmax, a.nfreq)
heat_freqs = np.linspace(fmin, fmax, a.nheat)
flux = [sim.add_flux(freqs, mp.FluxRegion(center=c, size=s)) for c, s, _ in faces]
# terminal current: Ampere loops around the upper terminal at mid-height, two and four cells clear of it
z_loop = a.gap / 2 + a.pad / 2
loops = {}
for n in (2, 4):
    sl = a.width / 2 + n * px
    loops[n] = [sim.add_dft_fields([mp.Hx], freqs, center=mp.Vector3(0, -sl, z_loop), size=mp.Vector3(2 * sl, 0, 0)),
                sim.add_dft_fields([mp.Hy], freqs, center=mp.Vector3(+sl, 0, z_loop), size=mp.Vector3(0, 2 * sl, 0)),
                sim.add_dft_fields([mp.Hx], freqs, center=mp.Vector3(0, +sl, z_loop), size=mp.Vector3(2 * sl, 0, 0)),
                sim.add_dft_fields([mp.Hy], freqs, center=mp.Vector3(-sl, 0, z_loop), size=mp.Vector3(0, 2 * sl, 0))]
# feed box: sides four cells clear of the terminals, ends at the terminals' mid-planes (inside the metal)
fx, fz = a.width / 2 + 4 * px, z_loop
COMPS = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
feed_faces = [(mp.Vector3(x=+fx), mp.Vector3(0, 2 * fx, 2 * fz), (1, 0, 0)), (mp.Vector3(x=-fx), mp.Vector3(0, 2 * fx, 2 * fz), (-1, 0, 0)),
              (mp.Vector3(y=+fx), mp.Vector3(2 * fx, 0, 2 * fz), (0, 1, 0)), (mp.Vector3(y=-fx), mp.Vector3(2 * fx, 0, 2 * fz), (0, -1, 0)),
              (mp.Vector3(z=+fz), mp.Vector3(2 * fx, 2 * fx, 0), (0, 0, 1)), (mp.Vector3(z=-fz), mp.Vector3(2 * fx, 2 * fx, 0), (0, 0, -1))]
feed_dft = [sim.add_dft_fields(COMPS, freqs, center=c, size=s) for c, s, _ in feed_faces]
# absorbed power: E and D on the Yee grid over the upper half of the antenna plus two vacuum cells
heat_dft, n_slab = add_heat_region(sim, heat_freqs, a.width / 2, a.length / 2, px)
n2f = sim.add_near2far(fcen, df, a.nheat, *[mp.Near2FarRegion(center=c, size=s, weight=w) for c, s, w in faces])

t0 = time.time()
sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-7))
wall = time.time() - t0

p_rad = sum(sign * np.array(mp.get_fluxes(fl)) for fl, (_, _, sign) in zip(flux, faces))


def line_integral(dft, comp):
    w = np.squeeze(np.asarray(sim.get_array_metadata(dft_cell=dft)[3]))
    return np.array([np.sum(w * np.squeeze(sim.get_dft_array(dft, comp, k))) for k in range(a.nfreq)])


def loop_current(m):
    return line_integral(m[0], mp.Hx) + line_integral(m[1], mp.Hy) - line_integral(m[2], mp.Hx) - line_integral(m[3], mp.Hy)


def box_power(dfts, box_faces):
    """Complex power out of a closed box: sum over faces of (E* x H) . n dA (MEEP DFT convention)."""
    s = np.zeros(a.nfreq, complex)
    for dft, (_, _, nrm) in zip(dfts, box_faces):
        w = np.squeeze(np.asarray(sim.get_array_metadata(dft_cell=dft)[3]))
        for k in range(a.nfreq):
            F = [np.squeeze(sim.get_dft_array(dft, c, k)) for c in COMPS]
            cr = (np.conj(F[1]) * F[5] - np.conj(F[2]) * F[4], np.conj(F[2]) * F[3] - np.conj(F[0]) * F[5],
                  np.conj(F[0]) * F[4] - np.conj(F[1]) * F[3])
            s[k] += np.sum(w * sum(nc * c for nc, c in zip(nrm, cr)))
    return s


current = loop_current(loops[2])
current_outer = loop_current(loops[4])
s_feed = box_power(feed_dft, feed_faces)
z_in = s_feed / abs(current) ** 2
p_in = s_feed.real

# heat per one-cell slab above the gap centre; the region starts in the vacuum gap, so no heat may lie below z = 0
z_mid = ((np.arange(n_slab) + 0.5) * px).tolist()
heat = []
for k, f in enumerate(heat_freqs):
    q_slab, below = slab_heat(sim, heat_dft, k, f, px, n_slab)
    total = q_slab.sum() + below
    if abs(below) > 1e-6 * abs(total):
        raise SystemExit(f"heat below the gap centre: {below:.3e} of {total:.3e}")
    heat.append({"f": float(f), "p_abs_both_arms": float(2 * total), "z": z_mid, "q_z": (q_slab[:n_slab] / px).tolist()})

theta = np.linspace(0, np.pi, 37)
R = 1000.0
ff = [np.array(sim.get_farfield(n2f, mp.Vector3(R * np.sin(t), 0, R * np.cos(t)))) for t in theta]
u = [[float(R ** 2 * np.real(np.cross(np.conj(f_[6 * k:6 * k + 3]), f_[6 * k + 3:6 * k + 6])) @ np.array([np.sin(t), 0, np.cos(t)]))
      for t, f_ in zip(theta, ff)] for k in range(a.nheat)]

json.dump({
    "length_um": a.length, "width_um": a.width, "gap_um": a.gap, "pad_um": a.pad, "gaas_start_um": g0, "resolution": a.resolution,
    "drude": {"fp": a.fp, "gamma": a.gamma, "eps_inf": a.eps_inf}, "cell": [sx, sx, sz],
    "voxels": (2 * nx) ** 2 * 2 * nz, "wall_s": round(wall, 1),
    "freq_thz": (freqs / THZ).tolist(), "p_rad": p_rad.tolist(),
    "p_in": p_in.tolist(), "z_re": z_in.real.tolist(), "z_im": z_in.imag.tolist(),
    "current_loop_ratio": (abs(current_outer) / abs(current)).tolist(),
    "heat": heat, "theta": theta.tolist(), "u_theta": u,
}, open(a.out / "antenna.json", "w"), indent=1)
print(f"GaAs dipole: res {a.resolution}, wall {wall:.0f} s")
