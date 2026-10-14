"""1D MEEP: millimetre-wave transmission through a thin conducting membrane.

Run with the meep_arm interpreter:  python meep_mmwave.py config.json spectrum.json
config.json: {"eps_r": static interband eps, "sigma_Sm": DC conductivity, "thickness_m",
              "f_min_Hz", "f_max_Hz", "points", "resolution_per_mm", "pml_mm", "pad_mm"}

Length unit a = 1 mm. MEEP's conductivity enters as eps(w) = eps_r (1 + i sigma_D / w) with w the
angular frequency in MEEP units (2 pi f_meep, f_meep = f a / c), so
    sigma_D = sigma / (eps0 eps_r) * a / c.
That unit choice is checked against the transfer-matrix result by the driver, not assumed.
"""

import json
import sys

import numpy as np
import meep as mp

cfg = json.load(open(sys.argv[1]))
out = sys.argv[2]
a_m = 1e-3
c0, eps0 = 299792458.0, 8.8541878128e-12
f2meep = a_m / c0                                  # Hz -> MEEP frequency units
fmin, fmax = cfg["f_min_Hz"] * f2meep, cfg["f_max_Hz"] * f2meep
fcen, df, nfreq = 0.5 * (fmin + fmax), fmax - fmin, cfg["points"]
sigma_D = cfg["sigma_Sm"] / (eps0 * cfg["eps_r"]) * a_m / c0
medium = mp.Medium(epsilon=cfg["eps_r"], D_conductivity=sigma_D)
d = cfg["thickness_m"] / a_m
dpml, pad = cfg["pml_mm"], cfg["pad_mm"]
sz = 2 * dpml + 2 * pad + d
src_z = -sz / 2 + dpml + 0.1 * pad
refl_z, tran_z = src_z + 0.2 * pad, sz / 2 - dpml - 0.1 * pad


def simulate(geometry, refl_data=None):
    sim = mp.Simulation(cell_size=mp.Vector3(0, 0, sz), resolution=cfg["resolution_per_mm"], dimensions=1,
                        boundary_layers=[mp.PML(dpml)], geometry=geometry,
                        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ex,
                                           center=mp.Vector3(0, 0, src_z))])
    refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    tran = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, tran_z)))
    if refl_data is not None:
        sim.load_minus_flux_data(refl, refl_data)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(10, mp.Ex, mp.Vector3(0, 0, tran_z), 1e-8))
    return sim, refl, tran


sim0, refl0, tran0 = simulate([])
incident = np.array(mp.get_fluxes(tran0))
freqs = np.array(mp.get_flux_freqs(tran0))
slab = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, d), center=mp.Vector3(), material=medium)]
sim1, refl1, tran1 = simulate(slab, sim0.get_flux_data(refl0))
R = -np.array(mp.get_fluxes(refl1)) / incident
T = np.array(mp.get_fluxes(tran1)) / incident
order = np.argsort(freqs)
json.dump({"freq_Hz": (freqs[order] / f2meep).tolist(), "T": T[order].tolist(), "R": R[order].tolist(),
           "SE_dB": (-10 * np.log10(np.maximum(T[order], 1e-30))).tolist(), "sigma_D_meep": sigma_D,
           "meep_version": mp.__version__, "resolution_per_mm": cfg["resolution_per_mm"]}, open(out, "w"))
print("MEEP done")
