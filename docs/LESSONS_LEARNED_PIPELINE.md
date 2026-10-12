# Lessons Learned: Ab Initio Optical Pipeline on Apple Silicon
## QE → Yambo GW+BSE → MEEP FDTD — Zero Empirical Parameters

**Author:** Grant D. Heileman, UNM ECE  
**Date:** March 2026  
**Hardware:** Apple M2 Max (8P+4E cores, 30-core GPU, 96GB unified memory)

---

## Executive Summary

Complete methodology and results from running an end-to-end ab initio optical simulation pipeline on a MacBook. Computes thin-film T/R/A with zero empirical material parameters.

**Pipeline:** `QE SCF → QE NSCF → p2y → Yambo GW → Yambo BSE → yambo_eps_to_meep.py → MEEP FDTD`  
**Total wall time:** ~10 minutes (Yambo dominated). MEEP: 0.17 seconds.

---

## 1. Computational Parameters

### 1.1 DFT (QE 7.4.1)
- Cell: 2-atom FCC Si, a=10.20 bohr
- Functional: PBE, ecutwfc=30 Ry
- K-grid: 4×4×4, force_symmorphic=.true.
- Total energy: -22.83843768 Ry

### 1.2 GW (Yambo 5.3.0, DP)
- G₀W₀ + Plasmon-Pole Approximation
- Screening/self-energy bands: 1-20
- VBM correction: +0.281 eV, CBM: +0.765 eV
- Z ≈ 0.80

### 1.3 BSE (Yambo 5.3.0, DP)
- Screened exchange kernel, full diagonalization
- BSE bands: 3-6 (2 valence + 2 conduction)
- ε₁(0) = 8.22, ε₂ peak = 101.24 at 3.33 eV

### 1.4 Lorentzian Fit (yambo_eps_to_meep.py)
- 4 poles, differential_evolution optimizer
- ε∞ = 1.00 (lower bound, not itself an error: see §3.1)
- E₁ pole: 2.726 eV, σ=2.86
- E₂ pole: 3.431 eV, σ=8.18

### 1.5 FDTD (MEEP)
- 500 nm Si thin film, normal incidence
- 1D simulation, 40 px/μm, PML 1.0 μm
- Broadband source 1-4 eV

---

## 2. Results: 500 nm Si Thin Film

| Energy (eV) | λ (nm) | T | R | A |
|-------------|--------|-------|-------|-------|
| 1.00 | 1240 | 0.268 | 0.567 | 0.164 |
| 1.30 | 954 | 0.155 | 0.640 | 0.205 |
| 1.61 | 770 | 0.163 | 0.334 | **0.503** |
| 1.91 | 649 | 0.041 | 0.521 | 0.437 |
| 2.21 | 561 | 0.003 | 0.561 | 0.436 |
| 2.52 | 492 | 0.000 | 0.799 | 0.201 |
| 3.42 | 363 | 0.000 | 0.904 | 0.096 |

Peak absorption 50% at 1.6 eV (band edge). Zero transmittance above 2.2 eV.

---

## 3. Known Issues and Fixes

### 3.1 Static limit of the Lorentz fit (FIXED; was "eps_inf = 1.0, should be ~8-12")
The original diagnosis was wrong. The model gives ε(0) = ε∞ + Σσₙ, so sub-gap n and R depend on that sum, not on ε∞ alone. ε∞ ≈ 1 is fine: it means the 0-10 eV window holds nearly all the oscillator strength. The interim fix (pin ε∞ to [0.8, 1.2]×ε₁(0)) forces Σσ → 0 and was worse on every spectrum below. The original trigger: on a Si IP-RPA spectrum with ε₁(0) ≈ 22 it gave ε∞ = 17.7 and rms|Δε| ≈ 15.4. The "0.57 vs ~0.30" comparison also set a 500 nm film's R (Fabry-Pérot) against bulk Si. The analytic 500 nm film R(1 eV) from the causal o-gw_mpi ε is itself 0.57.

There were two real defects:
1. **No static constraint.** ε(0) landed wherever the ε₂ fit left it (GaAs IP: 44.0 vs 35.9). Fix: `fit_lorentzian` imposes ε∞ + Σσₙ = ε₁(0) exactly (σₙ = (ε₁(0) − ε∞)·wₙ/Σw), with ε∞ ∈ [1, ε₁(0)].
2. **Yambo `BSEmod="resonant"` writes a non-causal ε₁.** The header says `Green's Function: Resonant`, and Re ε is only the resonant half of the Hilbert transform of ε₂. Across 0.3-9.7 eV it matches resonant-only KK to rms 0.3-1.5, against 3.5-3.9 for full KK. So ε₁(0) − 1 is half its own KK sum: o-gw_mpi gives 7.92 vs 15.16. The IP outputs (`T-ordered`) are causal. Fix: `add_antiresonant()` restores Δε₁(ω) = (1/π)∫ε₂(ω')(ω'+ω)/((ω'+ω)²+η²)dω', and `main()` applies it when the header says Resonant.

Measured, 3 poles, 0-10 eV. Columns are ε∞ / Σσ / rms|Δε|. HEAD and pinned rms are against the file as fitted; "new" is against the causal ε₁ (identical to the file for IP):

| Spectrum | ε₁(0) file → causal | HEAD (ε∞ ∈ [1,20]) | pinned (interim) | new (sum rule) |
|---|---|---|---|---|
| GaAs IP, `gaas-emi-shield/yambo/o-gaas_ip.eps_q1_ip` | 35.93 | 1.26 / 42.74 / 8.27 | 28.75 / 29.28 / 27.09 | 1.34 / 34.60 / 5.31 |
| GaAs BSE, `~/Dev/yambo-build/gaas-bse/o-bse` | 10.36 → 19.79 | 1.00 / 15.67 / 5.62 | 8.28 / 14.35 / 11.11 | 1.73 / 18.06 / 4.71 |
| Si BSE, `~/yambo-build/si2-gw/o-gw_mpi` | 7.92 → 14.98 | 1.00 / 12.46 / 4.38 | 6.33 / 11.50 / 8.98 | 1.00 / 13.98 / 2.66 |
| Si BSE+scissor, `si2-gw/o-bse_scissors` | 7.77 → 14.53 | 1.00 / 11.96 / 3.95 | 6.21 / 10.95 / 8.36 | 1.21 / 13.32 / 2.47 |
| Si IP+scissor, `si2-gw/o-si_scissors` (IP) | 15.46 | 1.37 / 14.66 / 3.26 | 12.37 / 12.40 / 11.26 | 1.10 / 14.36 / 3.30 |

The sum rule costs nothing globally. Refit HEAD on the causal o-bse_scissors data and its rms is 2.44 against 2.47 for the new fit, while the ≤ 1 eV rms drops 0.46 → 0.12. Across all five spectra the new fit's ≤ 1 eV rms is 0.12-2.4, against 0.6-19.3 for HEAD. Its 500 nm film R(1 eV) is within 0.02 of the causal data's; HEAD/pinned miss by up to 0.5.

Still open: the causal Si ε₁(0) ≈ 15 vs experiment 11.7 (bulk R(0) 0.35 vs 0.30). That is the BSE convergence of §3.3, not the fitter.

### 3.2 No Indirect Gap
2-atom cell only has direct transitions. Si onset should be 1.12 eV not 1.6 eV.

### 3.3 Underconverged BSE
4×4×4 k-grid, 4 BSE bands is minimal. Need 8×8×8 and more bands.

### 3.4 GW Not Fed to BSE
Scissors correction needed to shift BSE peaks to correct positions (~+0.7 eV).

---

## 4. BSE Output Format
File: o-*.eps_q1_diago_bse — 5 columns: Energy(eV), Im[ε]_BSE, Re[ε]_BSE, Im[ε]_IP, Re[ε]_IP
With `BSEmod="resonant"` both Re columns are resonant-only (not KK-consistent with Im). The converter restores the anti-resonant term (§3.1).

## 5. Lorentzian Model
ε(ω) = ε∞ + Σₙ σₙωₙ²/(ωₙ² - ω² - iγₙω)
MEEP units: frequency = E(eV)/1.23984, gamma = Γ(eV)/1.23984

## 6. Reproducibility
All codes open source. Pseudopotential from QE library. Converter published with paper.
