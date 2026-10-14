#!/usr/bin/env python3
"""
MEEP simulation using ab initio material from Yambo BSE.

This completes the pipeline:
  QE pw.x (ground state) → Yambo GW+BSE (optical properties) → MEEP (light propagation)

The material's ε(ω) was computed from first principles — no empirical parameters.
"""
import meep as mp
import os
import json
import numpy as np

print("Silicon-forge: QE → Yambo → MEEP photovoltaic workflow")
print("=" * 60)

# Load the fitted material from the converter
mat_file = os.path.join(os.path.dirname(__file__), 
    "../../../lib/converters/yambo_material.json")
# Fallback to direct path
import os
for path in [mat_file, 
             os.path.expanduser("~/Dev/Claude/rainbow-connection/lib/converters/yambo_material.json")]:
    if os.path.exists(path):
        mat_file = path
        break

with open(mat_file) as f:
    mat_data = json.load(f)

print(f"Material source: {mat_data['source_file']}")
print(f"Spectrum type: {mat_data['spectrum_type']}")
print(f"ε_∞ = {mat_data['eps_inf']:.4f}")
print(f"Poles: {len(mat_data['poles'])}")

# Build MEEP material from fitted parameters
susceptibilities = []
for p in mat_data['poles']:
    if p['sigma'] > 0.01:  # Skip dead poles
        hc = 1.23984198  # eV⋅μm
        susceptibilities.append(
            mp.LorentzianSusceptibility(
                frequency=p['omega_eV'] / hc,
                gamma=p['gamma_eV'] / hc,
                sigma=p['sigma']
            )
        )
        print(f"  Active pole: ω={p['omega_eV']:.3f} eV, "
              f"γ={p['gamma_eV']:.3f} eV, σ={p['sigma']:.4f}")

si_ab_initio = mp.Medium(
    epsilon=mat_data['eps_inf'],
    E_susceptibilities=susceptibilities,
)

# Simple thin-film absorption simulation
# A slab of ab initio Si illuminated by broadband light
resolution = 40  # pixels per μm
dpml = 1.0       # PML thickness

si_thickness = 0.5   # μm (500 nm Si film)
cell_z = si_thickness + 2 * dpml + 2.0  # air gaps + PML
cell = mp.Vector3(0, 0, cell_z)

# Si slab centered in the cell
geometry = [
    mp.Block(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(mp.inf, mp.inf, si_thickness),
        material=si_ab_initio,
    )
]

# Broadband source (covers the optical range)
fcen = 2.5 / 1.23984  # ~2.5 eV center frequency in MEEP units
df = 3.0 / 1.23984    # ~3 eV bandwidth (covers 1-4 eV)
sources = [
    mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0, -si_thickness/2 - 0.5),
    )
]

# Flux monitors: reflected and transmitted
pml_layers = [mp.PML(dpml)]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    dimensions=1,  # 1D for thin film (normal incidence)
)

# Monitor transmitted flux (past the slab)
nfreq = 100
trans_mon = sim.add_flux(
    fcen, df, nfreq,
    mp.FluxRegion(center=mp.Vector3(0, 0, si_thickness/2 + 0.3))
)

# Monitor reflected flux (before the slab)
refl_mon = sim.add_flux(
    fcen, df, nfreq,
    mp.FluxRegion(center=mp.Vector3(0, 0, -si_thickness/2 - 0.3))
)

print(f"\nSimulation parameters:")
print(f"  Si thickness: {si_thickness*1000:.0f} nm")
print(f"  Resolution: {resolution} pixels/μm")
print(f"  Frequency range: {(fcen-df/2)*1.23984:.1f} - {(fcen+df/2)*1.23984:.1f} eV")
print(f"  Cell size: {cell_z:.1f} μm")

# First run: empty cell (reference for normalization)
print(f"\nRunning reference (empty cell)...")
sim_ref = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    dimensions=1,
)
refl_ref = sim_ref.add_flux(
    fcen, df, nfreq,
    mp.FluxRegion(center=mp.Vector3(0, 0, -si_thickness/2 - 0.3))
)
trans_ref = sim_ref.add_flux(
    fcen, df, nfreq,
    mp.FluxRegion(center=mp.Vector3(0, 0, si_thickness/2 + 0.3))
)
sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6))
ref_flux = mp.get_fluxes(trans_ref)
ref_refl_data = sim_ref.get_flux_data(refl_ref)

# Second run: with Si slab
print(f"Running with ab initio Si slab...")
sim.load_minus_flux_data(refl_mon, ref_refl_data)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6))

# Extract results
freqs = mp.get_flux_freqs(trans_mon)
trans_flux = mp.get_fluxes(trans_mon)
refl_flux = mp.get_fluxes(refl_mon)

# Compute T, R, A
energies_eV = [f * 1.23984 for f in freqs]
T = [trans_flux[i] / ref_flux[i] if ref_flux[i] > 0 else 0 for i in range(nfreq)]
R = [-refl_flux[i] / ref_flux[i] if ref_flux[i] > 0 else 0 for i in range(nfreq)]
A = [1 - T[i] - R[i] for i in range(nfreq)]

print(f"\n{'='*60}")
print(f"Results: {si_thickness*1000:.0f} nm Si thin film (ab initio BSE material)")
print(f"{'='*60}")
print(f"  {'Energy (eV)':>12}  {'T':>8}  {'R':>8}  {'A':>8}")
print(f"  {'-'*42}")
for i in range(0, nfreq, nfreq//10):
    if 0.5 < energies_eV[i] < 5.0:
        print(f"  {energies_eV[i]:12.2f}  {T[i]:8.3f}  {R[i]:8.3f}  {A[i]:8.3f}")

# Save results
results = {
    'energy_eV': energies_eV,
    'transmittance': T,
    'reflectance': R,
    'absorptance': A,
    'si_thickness_um': si_thickness,
    'material_source': mat_data['source_file'],
}
outfile = os.path.expanduser(
    "~/Dev/Claude/rainbow-connection/workflows/si-photovoltaic/results/thin_film_tra.json")
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else x)
print(f"\nSaved: {outfile}")

print(f"\nThis simulation used NO empirical material parameters.")
print(f"ε(ω) was computed entirely from first principles:")
print(f"  QE pw.x    → DFT ground state")
print(f"  Yambo GW   → quasiparticle band gap correction")
print(f"  Yambo BSE  → excitonic optical absorption")
print(f"  Converter  → Lorentzian fit for MEEP")
print(f"  MEEP FDTD  → light propagation and absorption")
