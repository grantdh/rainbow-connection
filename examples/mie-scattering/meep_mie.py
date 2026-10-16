"""MEEP 3D: scattering efficiency of a homogeneous sphere under plane-wave illumination.

Setup follows MEEP's documented Mie example (Python tutorials, "Mie Scattering of a Lossless Dielectric
Sphere"): an Ez plane wave travelling along +x through a cubic cell with PML on every face, a closed box
of six flux planes just outside the sphere, and two runs. The empty-cell run gives the incident
intensity and the incident fields on the box; the second run subtracts them (load_minus_flux_data), so
the box sees only the scattered field and its net outward flux is the scattered power.

    Q_sca(f) = P_sca(f) / (I_inc(f) * pi r^2),   x = 2 pi r f  (vacuum, r = 1, c = 1)

Two mirror planes (y even, z odd for Ez polarisation) cut the work by four. Both runs stop once every
DFT has converged (stop_when_dft_decayed), which also works for the slowly ringing high-index cases.

With --drude-fp the sphere is a Drude metal instead, eps(f) = eps_inf - fp^2 / (f^2 + i f gamma) (MEEP
DrudeSusceptibility with frequency 1 and sigma = fp^2; fp, gamma are ordinary frequencies in MEEP units).
A second set of box monitors then records the total field, whose net inward flux (minus the empty cell's
own net flux through the same box) is the absorbed power, so Q_abs is measured alongside Q_sca.

Run with the meep_arm interpreter:
    python meep_mie.py OUT_DIR [--resolution 20] [--index 2.0] [--nfreq 50] [--xmin 2] [--xmax 8]
                       [--drude-fp FP --drude-gamma G --drude-eps-inf E] [--pml-frac 0.5] [--air-frac 0.5]
Writes OUT_DIR/spectrum.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import meep as mp

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--resolution", type=float, default=20)
p.add_argument("--index", type=float, default=2.0)
p.add_argument("--nfreq", type=int, default=50)
p.add_argument("--xmin", type=float, default=2.0, help="smallest size parameter 2 pi r / lambda")
p.add_argument("--xmax", type=float, default=8.0)
p.add_argument("--drude-fp", type=float, default=None, help="unscreened plasma frequency (MEEP units)")
p.add_argument("--drude-gamma", type=float, default=0.0, help="damping rate as an ordinary frequency (MEEP units)")
p.add_argument("--drude-eps-inf", type=float, default=1.0)
p.add_argument("--pml-frac", type=float, default=0.5, help="PML thickness / longest wavelength")
p.add_argument("--air-frac", type=float, default=0.5, help="air gap between sphere and PML / longest wavelength")
p.add_argument("--src-bw", type=float, default=1.5, help="source bandwidth / monitored band")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

r = 1.0
fmin, fmax = a.xmin / (2 * np.pi * r), a.xmax / (2 * np.pi * r)
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
wvl_max = 1 / fmin
dpml = a.pml_frac * wvl_max
dair = a.air_frac * wvl_max
drude = a.drude_fp is not None
if drude:
    material = mp.Medium(epsilon=a.drude_eps_inf,
                         E_susceptibilities=[mp.DrudeSusceptibility(frequency=1.0, gamma=a.drude_gamma, sigma=a.drude_fp ** 2)])
else:
    material = mp.Medium(index=a.index)
s = 2 * (dpml + dair + r)
cell = mp.Vector3(s, s, s)
# is_integrated=True is required for a plane-wave source that extends into the PML. The pulse is wider
# than the monitored band: with fwidth = band, the band edges carried exp(-pi^2) ~ 5e-5 of the peak power
# and their spectra sat on the noise floor (Q_sca off by 20 % at x = 2 and 8 at every resolution)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=a.src_bw * df, is_integrated=True), center=mp.Vector3(-0.5 * s + dpml),
                     size=mp.Vector3(0, s, s), component=mp.Ez)]
symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
faces = {  # name: (centre, size, outward sign)
    "x1": (mp.Vector3(x=-r), mp.Vector3(0, 2 * r, 2 * r), -1), "x2": (mp.Vector3(x=+r), mp.Vector3(0, 2 * r, 2 * r), +1),
    "y1": (mp.Vector3(y=-r), mp.Vector3(2 * r, 0, 2 * r), -1), "y2": (mp.Vector3(y=+r), mp.Vector3(2 * r, 0, 2 * r), +1),
    "z1": (mp.Vector3(z=-r), mp.Vector3(2 * r, 2 * r, 0), -1), "z2": (mp.Vector3(z=+r), mp.Vector3(2 * r, 2 * r, 0), +1),
}


def simulation(geometry, total_too=False):
    sim = mp.Simulation(resolution=a.resolution, cell_size=cell, boundary_layers=[mp.PML(dpml)], sources=sources,
                        k_point=mp.Vector3(), symmetries=symmetries, geometry=geometry)
    mons = {k: sim.add_flux(fcen, df, a.nfreq, mp.FluxRegion(center=c, size=z)) for k, (c, z, _) in faces.items()}
    total = {k: sim.add_flux(fcen, df, a.nfreq, mp.FluxRegion(center=c, size=z)) for k, (c, z, _) in faces.items()} if total_too else {}
    return sim, mons, total


def net(mons):
    """Net flux out of the closed box."""
    return sum(sign * np.array(mp.get_fluxes(mons[k])) for k, (_, _, sign) in faces.items())


t0 = time.time()
sim0, mon0, _ = simulation([])
sim0.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6))
freqs = np.array(mp.get_flux_freqs(mon0["x1"]))
incident = np.array(mp.get_fluxes(mon0["x1"])) / (2 * r) ** 2       # power per unit area through the x1 face
incident_data = {k: sim0.get_flux_data(m) for k, m in mon0.items()}
# net flux of the incident field alone through the closed box (in the empty cell the monitors see the total
# field): ~0, subtracted so that any imbalance of the truncated plane wave is not counted as absorption
net_empty = net(mon0) if drude else None
t_empty = time.time() - t0
sim0.reset_meep()

t1 = time.time()
sim1, mon1, tot1 = simulation([mp.Sphere(radius=r, center=mp.Vector3(), material=material)], total_too=drude)
for k, m in mon1.items():
    sim1.load_minus_flux_data(m, incident_data[k])
sim1.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6))
scattered = net(mon1)
absorbed = -(net(tot1) - net_empty) if drude else None
t_sphere = time.time() - t1

qsca = scattered / (incident * np.pi * r ** 2)
if drude:
    eps = a.drude_eps_inf - a.drude_fp ** 2 / (freqs ** 2 + 1j * freqs * a.drude_gamma)
    n_max = float(np.max(abs(np.sqrt(eps))))
else:
    n_max = a.index
json.dump({
    "index": a.index if not drude else None,
    "drude": {"fp": a.drude_fp, "gamma": a.drude_gamma, "eps_inf": a.drude_eps_inf} if drude else None,
    "qabs": (absorbed / (incident * np.pi * r ** 2)).tolist() if drude else None,
    "radius": r, "resolution": a.resolution, "pixels_per_wavelength_in_sphere": a.resolution / (fmax * n_max),
    "cell_size": s, "voxels": int(round(s * a.resolution)) ** 3,
    "freq": freqs.tolist(), "x": (2 * np.pi * r * freqs).tolist(), "qsca": qsca.tolist(),
    "wall_s": {"empty": round(t_empty, 1), "sphere": round(t_sphere, 1)},
}, open(a.out / "spectrum.json", "w"), indent=1)
print(f"resolution {a.resolution}: {len(freqs)} frequencies, wall {t_empty:.0f} + {t_sphere:.0f} s")
