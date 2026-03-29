# GaAs Membrane EMI Shielding Workflow

Computational reproduction of experimental results from:
Renteria, Heileman, et al., "Infrared-Transparent Semiconductor Membranes
for Electromagnetic Interference Shielding of Millimeter Waves"
(Advanced Materials, submitted 2024)

## Goal
Reproduce Figures 2, 4, 5, and 6 from first-principles computation,
validating the MUPPET pipeline against experimental data.

## Experimental Targets (from the paper)

### IR Transmittance (Fig. 2a)
- 200 nm GaAs: resonant peaks 750-2500 nm, T varies 40-100%
- 500 nm GaAs: more Fabry-Perot fringes, same T range
- 1000 nm GaAs: dense fringes, T_min ~ 40%

### mm-wave SE (Fig. 4b)
- 200 nm: SE ~ 12 dB at 75 GHz
- 500 nm: SE ~ 18 dB at 75 GHz
- 1000 nm: SE ~ 22-26 dB at 75 GHz

### Multilayer SE (Fig. 6b)
- 1 μm GaAs / 125 μm Kapton / 1 μm GaAs: SE ~ 38-40 dB

## Material Parameters (from paper)

### GaAs membranes
- Si doped, n = 1 × 10¹⁹ cm⁻³
- Relative permittivity: 13.1 (undoped, high-frequency)
- Measured DC conductivities (van der Pauw):
  - 200 nm: σ = 1.19 × 10⁵ S/m
  - 500 nm: σ = 8.81 × 10⁴ S/m
  - 1000 nm: σ = 8.01 × 10⁴ S/m

### Drude model for free carriers
- ε(ω) = ε_∞ - ω_p² / (ω² + iγω)
- ω_p² = n·e² / (ε₀·m*)
- For n = 10¹⁹ cm⁻³, m* = 0.063 m_e (GaAs):
  ω_p ≈ 2.3 × 10¹⁴ rad/s (~ 37 THz)
- γ = σ·m* / (n·e²·ε₀) from measured conductivity

### Kapton (polyamide)
- Dielectric constant: 3.4
- Thickness: 125 μm
- Loss tangent: ~0.003 at mm-wave frequencies

## Pipeline
```
01_run_qe_gaas.sh        # DFT ground state of GaAs (zincblende, 2 atoms)
02_convert_p2y.sh         # QE → Yambo
03_run_yambo_gw.sh        # GW band gap correction
04_run_yambo_bse.sh       # BSE ε(ω) for interband transitions
05_add_drude.py           # Add free-carrier Drude term to BSE ε(ω)
06_run_meep_ir.py         # MEEP: IR transmittance 750-2500 nm
07_run_meep_mmwave.py     # MEEP: mm-wave SE 65-85 GHz
08_run_meep_multilayer.py # MEEP: GaAs/Kapton/GaAs stack
09_compare_experiment.py  # Plot computed vs experimental
```

## Key Physics

The IR transmittance shows Fabry-Perot resonances: constructive/destructive
interference in the thin GaAs cavity. Peak positions depend on thickness and
refractive index. Free-carrier absorption (Drude) causes the baseline loss.

The mm-wave SE is dominated by reflection from the high-conductivity membrane.
At 75 GHz, skin depth in GaAs (σ ~ 10⁵ S/m) is ~5 μm — much larger than the
membrane thickness. So the shielding is NOT from absorption but from impedance
mismatch (reflection) and internal multiple reflections.

The multilayer GaAs/Kapton/GaAs enhances SE by adding more reflection interfaces.
The Kapton spacer acts as a quarter-wave (ish) impedance transformer at mm-wave.

## What This Validates

If MUPPET reproduces Figs. 2, 4, and 6 from first principles + measured σ:
- The QE → Yambo → MEEP pipeline works for III-V semiconductors (not just Si)
- The Drude + interband material model captures both IR and mm-wave regimes
- MEEP correctly handles deeply subwavelength films (200 nm at 4 mm wavelength)
- The multilayer transfer matrix is validated against full-wave FDTD
