# Example 7: shielding by a conducting slab against classical skin-effect theory

Exact solution: the characteristic-matrix transfer matrix (`../gaas-emi-membrane/emi_lib.py`) for a
slab with eps = eps_r + i sigma/(eps0 w). Theory it is tested against:

- Schelkunoff decomposition, exact for a single slab: SE = R + A + B with
  Zw = Z0/n, gamma = -i k0 n (so Re gamma > 0 in the e^{-iwt} convention),
  R = 20 log10 |(Zw+Z0)^2 / (4 Zw Z0)|, A = 20 log10 |e^{gamma d}|, B = 20 log10 |1 - rho^2 e^{-2 gamma d}|
- good-conductor limits: A = 8.686 d/delta with delta = sqrt(2/(w mu0 sigma)); R = 20 log10 (Z0 / 4|Zw|)
- thin-film limit (d << delta): SE = -20 log10 |2 / (2 + Z0 sigma d)|
- MEEP 1D with a DC-conductivity medium, sigma_D = sigma / (eps0 eps_r) * a/c, for d = 10 um (40 px at 4000 px/mm,
  slab centre offset by half a pixel so both faces fall on H sites; PML 2.5 mm; fields decayed to 1e-9)

Material: doped GaAs, sigma = 8.01e4 S/m, eps_r = 12.9, 75 GHz (delta = 6.49 um); d = 0.5-30 um spans
d/delta from 0.08 to 4.6.

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--skip-meep]   # analytic part < 1 s; MEEP ~2-3 min
```

## Gates

| check | criterion |
|---|---|
| SE_tmm = R + A + B | < 1e-6 dB at every thickness (identity) |
| A vs 8.686 d/delta | < 0.1 dB |
| R - R_cl == 40 log10\|1 + Zw/Z0\| | identity to 1e-9 dB (R_cl = 20 log(Z0/4\|Zw\|) is itself within 0.2 dB) |
| thin-film limit, d/delta < 0.2 | < 0.1 dB |
| B for d > 3 delta | < 0.1 dB |
| slope between 20 and 30 um | 8.686 +/- 0.05 dB per skin depth |
| thin-film formula at 30 um | must fail by > 5 dB (the regime boundary is real) |
| MEEP vs transfer matrix, 65-85 GHz | < 0.1 dB (slab edges placed on H sites: half-pixel centre offset) |

A note on how this example was built: the first version defined gamma with the sign of the other time
convention, so A came out negative and B diverged while the exact transfer matrix was right. The
identity gate caught it before any solver ran. That is what the gates are for.

## Results

All checks pass (2026-10-15, 176 s). Analytic part: identity holds to 7e-15 dB, |A - 8.686 d/delta|
<= 0.013 dB, R - R_cl = 0.0887 dB and equals 40 log10|1 + Zw/Z0| to 1e-9 dB, thin-film limit exact for
d/delta < 0.2, |B| < 0.02 dB beyond 3 delta, slope 8.695 dB/delta. MEEP 1D (10 um slab, 4000 px/mm,
slab edges on H sites): SE(75 GHz) = 44.64 dB vs 44.63 dB from the transfer matrix, max |dSE| = 0.006 dB
over 65-85 GHz. The first MEEP version of this example, with the slab edges on D sites, was 0.1-0.3 dB
off: a one-pixel ambiguity in the slab thickness, not a solver error.
