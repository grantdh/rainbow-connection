# Data Format Converters

## yambo_eps_to_meep.py
Reads Yambo's ε(ω) output (o-*.eps_q1_diago_bse)
Fits to sum of Lorentzian oscillators: ε(ω) = ε_∞ + Σ σ_n ω_n / (ω_n² - ω² - iγ_n ω)
Outputs MEEP-compatible LorentzianSusceptibility parameters

## qe_structure_to_meep.py
Reads QE's output geometry (cell parameters, atomic positions)
Converts to MEEP's geometric primitives (Block, Cylinder, etc.)
Handles unit conversion (Bohr → μm)

## meep_fields_to_thermal.py (future)
Reads MEEP's absorbed power density
Converts to volumetric heat source for Elmer FEM
