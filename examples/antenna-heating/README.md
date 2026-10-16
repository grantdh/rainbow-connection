# Example 7: from the band structure of GaAs to the radiation and Joule heating of a mid-IR antenna

The first rung that strings the tools together: an ab initio material property feeds an antenna
simulation, which yields how much power is radiated, how much heats the antenna, where, and how hot it gets.

```
A  QE pw.x (PBE, NC ONCV, a = 5.6533 A)  GaAs SCF -> conduction band near Gamma along Gamma-X and Gamma-L
     -> m*_PBE and the PBE gap -> two-band Kane with the measured gap -> m*      (the pipeline's scissor step,
                                                                                 applied to the band curvature)
B  Drude eps(w) of n = 1e19 cm^-3 GaAs: plasma frequency from n and m*, damping from the measured DC
     conductivity of Grant's 1000 nm membrane (8.01e4 S/m), eps_inf = 10.86
C  MEEP 3D Drude sphere (r = 1 um, 22-40 THz) vs Mie theory with the same complex eps(w):
     absorption, scattering, Froehlich resonance -> the dispersive material in the FDTD code
D  transmitting: MEEP 3D gap-fed GaAs dipole (5 um, 0.6 um arms, 0.2 um perfect-conductor feed terminals),
     15-40 THz, at 10 and 15 px/um: Z_in, radiated power, absorbed power, radiation efficiency;
     P_in = P_rad + P_abs checks the absorbed power
E  receiving: the same 5 um GaAs rod (no gap) in an E-parallel plane wave, 8-20 THz around its length
     resonance, at 10 and 15 px/um: absorption and scattering (re-radiation) cross-sections, heat per
     unit length along the rod, absorbed power measured twice (box flux, Yee-grid E . D sum)
F  steady 1D conduction along the rod bridging two heat-sunk contacts, 1 kW/cm^2 at the absorption
     peak -> temperature rise T(z)
```

meep_gaas_dipole.py runs D, meep_gaas_rod.py runs E, and ../mie-scattering/meep_mie.py runs C.

## Why the mid-infrared

At the membranes' mm-wave frequencies (75 GHz), doped GaAs is DC-like (omega tau ~ 0.008). The effective
mass cancels out of the conductivity there, so an ab initio mass would be decorative. Below its screened
plasma frequency (29.8 THz at n = 1e19 cm^-3 with the Kane mass below) the same material is a lossy metal
whose permittivity depends directly on m*, and with it the antenna's resonance, efficiency and heat map.

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-sphere] [--skip-transmit] [--resolutions 10 15]
```

The receive stage at 15 px/um dominates the wall time (about 80 min on a shared M2 Max; the whole example
takes about two hours).

## What it checks

| stage | check | gate |
|---|---|---|
| F | 1D conduction integrator vs analytic uniform-source fin | 1e-3 |
| A | reference pw.x is apple-bottom-free | 0 `_ab_*` symbols |
| A | PBE direct gap at Gamma vs the same pseudos in workflows/gaas-emi-shield | 0.5135 +/- 0.10 eV |
| A | CBM mass isotropic, Gamma-X vs Gamma-L | 1 +/- 5 % |
| A | Kane-corrected m* at the 0 K gap vs experiment (0.067, Vurgaftman 2001) | +/- 20 % (two-band, no spin-orbit) |
| C | Drude sphere Q_abs and Q_sca vs Mie at 10 px/um | rel rms < 10 % each |
| C | Froehlich resonance position | < 3 % + one frequency bin |
| D | P_in = P_rad + P_abs | 1 +/- 5 % at 11 frequencies |
| D | radiation efficiency at resonance, 10 vs 15 px/um | +/- 10 % |
| E | absorbed power: Yee-grid E . D sum vs box flux | 1 +/- 5 % at the heat frequencies |
| E | peak sigma_abs, 10 vs 15 px/um | +/- 10 % |
| E | frequency of the sigma_abs peak, 10 vs 15 px/um | < 3 % + one frequency bin |

Gates A-D and F were fixed on 2026-10-14 before the first run. The E gates were fixed on 2026-10-15,
before its first 10 and 15 px/um runs. A 5 px/um plumbing scan from 5 to 40 THz had already located the
resonance and shown the two absorbed-power measurements within 1.1 %, so the first E gate is not blind.

## What went wrong, and what changed (2026-10-15)

- **Kane mass (A) fails and stays red.** m*_PBE = 0.0316 gives E_P = 15.6 eV, and with the 0 K gap the
  two-band Kane mass is 0.0888 against 0.067. A rigid gap correction that leaves E_P at its PBE value
  overshoots, and the two-band form ignores spin-orbit and remote bands. Stage B uses the 300 K-gap value
  (0.0837) as computed. It is not tuned toward experiment.
- **The Drude sphere (C) fails at 10 px/um.** Q_abs is off by 17.7 % rms and Q_sca by 33.6 %, against a
  10 % gate. The Froehlich resonance position passes. The error sits in the wings of the resonance (MEEP
  40-50 % high at 22 and 37 THz, within 7 % at 25-28 THz), and it falls as 1/resolution: 11.9 % / 20.3 %
  at 15 px/um and 8.4 % / 14.7 % at 20 px/um (first order; a 4/3 refinement gains 1.4x). The gate stays
  at 10 px/um as fixed, so the check stays red. That pattern is consistent with a staircased metal surface spreading the
  sphere's single dipolar resonance into its wings, while the material model is right where the field is
  resonant. A wider source pulse
  (which fixed the dielectric Mie example's band edges) does not change it (17.2 % / 32.9 %), and neither
  does moving the PML out. The rods in D and E are boxes aligned with the grid, so they are not staircased.
- **The transmitting dipole's absorbed power was wrong (D).** The first version integrated
  omega Im(eps) |E|^2 over a grid that MEEP had snapped out into the driven gap. Interpolating E across the
  GaAs surface, where the normal field jumps by |eps| ~ 30, over-counted the heat up to 4.8-fold
  ((P_rad + P_abs) / P_in = 4.8 at 15 THz). The absorbed power is now omega Im(E* . D), evaluated
  component by component on MEEP's own Yee grid, where E and D share a point; it is exactly zero in vacuum
  and metal. The energy balance now holds to 0.3 % at 15 px/um.
- **The transmitting dipole's efficiency does not converge (D), and stays red.** Driving the GaAs directly
  from the gap put half the heat within 0.3 um of the source. Perfect-conductor terminals moved the
  source's singular field off the lossy material, but much of the heat still sits where the current enters
  the GaAs: 14-69 % of it within 0.3 um of the terminals, across the band. That junction loss is not
  resolved at 10-15 px/um. The radiation resistance R_rad = eta R_in barely moves at 15 THz (33 vs 36 ohm)
  while the loss resistance halves (456 vs 213 ohm). At the 15 px/um radiation peak (29.5 THz), eta is
  0.099 at 10 px/um and 0.061 at 15. A scratch study at 7.5, 12.5 and 17.5 px/um, where every material
  face sits a quarter cell off both Yee lattices, rules out grid phase as the cause: eta keeps drifting
  (at 35 THz 0.45, 0.39, 0.35, 0.23 at 10, 12.5, 15, 17.5 px/um; at 15 THz between 0.07 and 0.20), while
  the energy balance holds to 0.2-1 % at every resolution. This is why the heat map and temperature come
  from the receiving stage E. There is no feed there, and the heat follows the induced current.

- **The receiving rod's registered convergence checks (10 vs 15 px/um) fail and stay red, but the rod does
  converge.** Scratch runs at 7.5, 12.5 and 20 px/um (2026-10-15/16) give, for the absorption peak:

  | px/um | rod faces relative to the Yee grid | peak (THz) | sigma_abs (um^2) | heat at 1 kW/cm^2 | wall |
  |---|---|---|---|---|---|
  | 7.5 | a quarter cell off both lattices | 13.02 | 32.5 | 326 uW, dT 15.8 K | 5 min |
  | 10 | on grid planes | 10.53 | 20.8 | 201 uW, dT 9.7 K | 16 min |
  | 12.5 | a quarter cell off both lattices | 11.49 | 25.2 | 244 uW, dT 11.8 K | 31 min |
  | 15 | on half-cell planes | 11.94 | 28.4 | 285 uW, dT 13.8 K | 72 min |
  | 20 | on grid planes | 11.44 | 24.2 | 236 uW, dT 11.5 K | 5.7 h |

  20 and 12.5 px/um agree to 0.4 % in frequency and 4 % in sigma_abs; the gated pair brackets them, 10 px/um
  low and 15 px/um high. So the grid phase of the rod's faces matters at these resolutions, presumably
  because where a face falls relative to the Yee samples shifts the effective edge of this thin, strongly
  negative-eps rod (not established here). An earlier reading of the 12.5 px/um point as
  refuting that (it falls between 10 and 15) and an extrapolation from 10/12.5/15 to 13-14 THz were wrong:
  the 20 px/um run settles it. The Poynting check holds to 0.04-0.26 % at every resolution, so the
  absorbed power is accounted for exactly on each grid; it is the resonance that moves with the grid. A
  sound convergence pair for this geometry is 12.5 and 20 px/um (about 6 h for stage E), or any two
  resolutions with the faces at the same grid phase.

## What the run gives (2026-10-15/16)

Best estimate (20 px/um, scratch run; the gated 15 px/um run in results/RESULTS.md overshoots it by 4 % in
frequency and 17 % in sigma_abs): the 5 um rod absorbs most strongly near 11.45 THz (lambda = 26 um, so
L / lambda = 0.19: the lossy-metal rod resonates far shorter than a perfect-conductor half-wave dipole
would). There sigma_abs is 24.2 um^2, 8x the rod's projected area, and the re-radiated (scattering)
cross-section is 2.1 um^2, so about 92 % of the power the rod intercepts becomes heat. Under 1 kW/cm^2 at
11 THz the rod absorbs 236 uW; the heat follows the induced current, peaking at 77 W/m at mid-length, and
the middle of a rod bridging two heat-sunk contacts warms by 11.5 K. The 12.5 px/um run agrees within 4 %.
The transmitting dipole, by contrast, deposits 14-69 % of its heat within 0.3 um of the feed terminals.

## Assumptions that are inputs, not results

Doping 1e19 cm^-3 and the DC conductivity are the membrane's measured values. eps_inf (10.86, Adachi) and
the thermal conductivity (46 W/m K for undoped GaAs at 300 K) are literature values. Heavy doping lowers
the thermal conductivity, so the temperature rise is a lower bound. The thermal stage is a free-standing
rod that sheds heat only into its two end contacts. It is an illustration of where the heat goes, not a
device temperature: a substrate would set that (Elmer is the planned next step).

Results: `results/RESULTS.md` after a run. Wall times there are observations on a shared machine.
