"""MEEP 3D: a doped-GaAs rod antenna in a mid-infrared plane wave (receive mode).

Units: 1 MEEP length unit = 1 um, c = 1, so f[MEEP] = f[THz] / 299.792458.
Geometry: a square rod (side w, length L) along z at the origin, Drude material
eps(f) = eps_inf - fp^2 / (f^2 + i f gamma) (MEEP DrudeSusceptibility, frequency 1, sigma = fp^2).
Illumination: an E_z-polarised plane wave travelling along +x (the polarisation that drives the rod's
length resonance), launched across the whole cell; mirror planes y (phase +1) and z (phase -1).

Two runs, as in ../mie-scattering/meep_mie.py: an empty cell gives the incident intensity, the incident
flux through a closed box around the rod (for the scattered-field subtraction) and that box's net flux
(the truncated plane wave's imbalance); the run with the rod then gives
  - sigma_sca(f): scattered-field flux out of the box / incident intensity (the re-radiated power);
  - sigma_abs(f): minus the total-field net flux into the box, less the empty-cell imbalance;
  - at every heat_every-th of the same frequencies, the absorbed power omega Im(E* . D) summed over MEEP's
    Yee grid in the upper half of the rod (x2 by symmetry; yee_heat.py) and the heat per unit length q(z).
    Poynting's theorem makes this equal to sigma_abs I_inc: two independent measurements.
The default band, 8-20 THz, brackets the 5 um rod's length resonance near 12 THz.
No feed: the transmitting version (meep_gaas_dipole.py) puts most of its heat where the current enters the
GaAs from the terminals, and that junction loss is not resolved at 10-15 px/um.
The source pulse is 1.5x wider than the monitored band, so the band edges are not on the noise floor.

Run with the meep_arm interpreter:
    python meep_gaas_rod.py OUT_DIR --fp FP --gamma G --eps-inf E [--length 5] [--width 0.6]
        [--fmin-thz 8] [--fmax-thz 20] [--resolution 10] [--nfreq 49] [--heat-every 4] [--pml-frac 0.3]
Writes OUT_DIR/rod.json.
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
p.add_argument("--fmin-thz", type=float, default=8.0)
p.add_argument("--fmax-thz", type=float, default=20.0)
p.add_argument("--resolution", type=float, default=10.0)
p.add_argument("--nfreq", type=int, default=49)
p.add_argument("--heat-every", type=int, default=4, help="heat map at every n-th flux frequency")
p.add_argument("--pml-frac", type=float, default=0.3, help="PML thickness / longest wavelength")
p.add_argument("--air", type=float, default=1.0, help="vacuum between the flux box and the PML (um)")
p.add_argument("--src-bw", type=float, default=1.5, help="source bandwidth / monitored band")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

fmin, fmax = a.fmin_thz * THZ, a.fmax_thz * THZ
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
freqs = np.linspace(fmin, fmax, a.nfreq)
heat_idx = np.arange(0, a.nfreq, a.heat_every)
heat_freqs = freqs[heat_idx]
px = 1 / a.resolution
dpml = a.pml_frac / fmin
margin = 0.5
bx, bz = a.width / 2 + margin, a.length / 2 + margin           # flux box half-sizes
nx, nz = (int(np.ceil((b + a.air + dpml) / px)) for b in (bx, bz))
sx, sz = 2 * nx * px, 2 * nz * px                               # whole, even pixel counts: the centre is a grid point
dpml = nx * px - bx - a.air                                     # absorb the rounding into the PML (x, y)
cell = mp.Vector3(sx, sx, sz)
drude = mp.Medium(epsilon=a.eps_inf, E_susceptibilities=[mp.DrudeSusceptibility(frequency=1.0, gamma=a.gamma, sigma=a.fp ** 2)])
# is_integrated=True is required for a plane-wave source that extends into the PML
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=a.src_bw * df, is_integrated=True), component=mp.Ez,
                     center=mp.Vector3(x=-sx / 2 + dpml), size=mp.Vector3(0, sx, sz))]
symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
faces = {  # name: (centre, size, outward sign)
    "x1": (mp.Vector3(x=-bx), mp.Vector3(0, 2 * bx, 2 * bz), -1), "x2": (mp.Vector3(x=+bx), mp.Vector3(0, 2 * bx, 2 * bz), +1),
    "y1": (mp.Vector3(y=-bx), mp.Vector3(2 * bx, 0, 2 * bz), -1), "y2": (mp.Vector3(y=+bx), mp.Vector3(2 * bx, 0, 2 * bz), +1),
    "z1": (mp.Vector3(z=-bz), mp.Vector3(2 * bx, 2 * bx, 0), -1), "z2": (mp.Vector3(z=+bz), mp.Vector3(2 * bx, 2 * bx, 0), +1),
}


def simulation(geometry, total_too=False):
    sim = mp.Simulation(resolution=a.resolution, cell_size=cell, boundary_layers=[mp.PML(dpml)], sources=sources,
                        k_point=mp.Vector3(), symmetries=symmetries, geometry=geometry)
    scat = {k: sim.add_flux(freqs, mp.FluxRegion(center=c, size=s)) for k, (c, s, _) in faces.items()}
    total = {k: sim.add_flux(freqs, mp.FluxRegion(center=c, size=s)) for k, (c, s, _) in faces.items()} if total_too else {}
    return sim, scat, total


def net(mons):
    return sum(sign * np.array(mp.get_fluxes(mons[k])) for k, (_, _, sign) in faces.items())


t0 = time.time()
sim0, scat0, _ = simulation([])
sim0.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-7))
incident = np.array(mp.get_fluxes(scat0["x1"])) / (2 * bx * 2 * bz)      # power per unit area through the x1 face
incident_data = {k: sim0.get_flux_data(m) for k, m in scat0.items()}
net_empty = net(scat0)        # in the empty cell the monitors see the total (= incident) field
t_empty = time.time() - t0
sim0.reset_meep()

t1 = time.time()
sim1, scat1, tot1 = simulation([mp.Block(size=mp.Vector3(a.width, a.width, a.length), center=mp.Vector3(), material=drude)],
                               total_too=True)
for k, m in scat1.items():
    sim1.load_minus_flux_data(m, incident_data[k])
# Yee-grid E and D over the upper half of the rod plus two vacuum cells
heat_dft, n_slab = add_heat_region(sim1, heat_freqs, a.width / 2, a.length / 2, px)
sim1.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-7))
scattered = net(scat1)
absorbed = -(net(tot1) - net_empty)
t_rod = time.time() - t1

# Heat per one-cell slab above z = 0. The rod continues through the mirror plane: samples below z = 0 belong to
# the mirrored half and the plane z = 0 is shared, which the slab weights already account for.
z_mid = ((np.arange(n_slab) + 0.5) * px).tolist()
heat = []
for k, (i, f) in enumerate(zip(heat_idx, heat_freqs)):
    q_slab, _ = slab_heat(sim1, heat_dft, k, f, px, n_slab)
    total = q_slab.sum()                                   # upper half of the rod
    if abs(q_slab[n_slab]) > 1e-6 * abs(total):
        raise SystemExit(f"heat above the region's top slab: {q_slab[n_slab]:.3e} of {total:.3e}")
    heat.append({"f": float(f), "p_abs_yee": float(2 * total), "p_abs_box": float(absorbed[i]), "incident": float(incident[i]),
                 "z": z_mid, "q_z": (q_slab[:n_slab] / px).tolist()})

json.dump({
    "length_um": a.length, "width_um": a.width, "resolution": a.resolution,
    "drude": {"fp": a.fp, "gamma": a.gamma, "eps_inf": a.eps_inf}, "cell": [sx, sx, sz], "pml": dpml,
    "voxels": (2 * nx) ** 2 * 2 * nz, "wall_s": {"empty": round(t_empty, 1), "rod": round(t_rod, 1)},
    "freq_thz": (freqs / THZ).tolist(), "incident": incident.tolist(),
    "sigma_sca_um2": (scattered / incident).tolist(), "sigma_abs_um2": (absorbed / incident).tolist(), "heat": heat,
}, open(a.out / "rod.json", "w"), indent=1)
print(f"GaAs rod: res {a.resolution}, wall {t_empty:.0f} + {t_rod:.0f} s")
