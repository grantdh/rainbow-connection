"""1D MEEP transmission/reflection of a free-standing slab with a Lorentz-pole material.

Run with the meep_arm interpreter:  python meep_thin_film.py config.json spectrum.json
config.json: {"material": {"eps_inf", "poles": [{"omega_eV","gamma_eV","sigma"}]},
              "thickness_nm", "e_min_eV", "e_max_eV", "points", "resolution"}
Length unit a = 1 µm, so f_meep = E_eV / 1.23984198.
"""

import json
import sys

import numpy as np
import meep as mp

cfg = json.load(open(sys.argv[1]))
out = sys.argv[2]
hc = 1.23984198
mat = cfg["material"]
sus = [mp.LorentzianSusceptibility(frequency=p["omega_eV"] / hc, gamma=p["gamma_eV"] / hc, sigma=p["sigma"])
       for p in mat["poles"] if p["sigma"] > 1e-6]
medium = mp.Medium(epsilon=max(mat["eps_inf"], 1.0), E_susceptibilities=sus)
d = cfg["thickness_nm"] / 1000.0
dpml, pad = 1.0, 1.0
sz = 2 * dpml + 2 * pad + d
fmin, fmax = cfg["e_min_eV"] / hc, cfg["e_max_eV"] / hc
fcen, df, nfreq = (fmin + fmax) / 2, fmax - fmin, cfg["points"]
src_z = -sz / 2 + dpml + 0.2
refl_z, tran_z = src_z + 0.2, sz / 2 - dpml - 0.2


def simulate(geometry, refl_data=None):
    sim = mp.Simulation(cell_size=mp.Vector3(0, 0, sz), resolution=cfg["resolution"], dimensions=1,
                        boundary_layers=[mp.PML(dpml)], geometry=geometry,
                        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ex,
                                           center=mp.Vector3(0, 0, src_z))])
    refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    tran = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, tran_z)))
    if refl_data is not None:
        sim.load_minus_flux_data(refl, refl_data)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, tran_z), 1e-9))
    return sim, refl, tran


sim0, refl0, tran0 = simulate([])
incident = np.array(mp.get_fluxes(tran0))
freqs = np.array(mp.get_flux_freqs(tran0))
slab = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, d), center=mp.Vector3(), material=medium)]
sim1, refl1, tran1 = simulate(slab, sim0.get_flux_data(refl0))
R = -np.array(mp.get_fluxes(refl1)) / incident
T = np.array(mp.get_fluxes(tran1)) / incident
order = np.argsort(freqs)
json.dump({"energy_eV": (freqs * hc)[order].tolist(), "T": T[order].tolist(), "R": R[order].tolist(),
           "meep_version": mp.__version__, "resolution": cfg["resolution"], "single_precision": mp.is_single_precision()},
          open(out, "w"))
print("MEEP done")
