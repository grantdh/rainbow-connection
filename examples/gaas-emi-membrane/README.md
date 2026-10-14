# Example 2: GaAs membrane EMI shielding, from the band structure to a measured figure

Reproduces the physics chain behind Renteria, Heileman et al. (Adv. Mater. Technol. 9, 2401013, 2024)
and the "More Than Skin Deep" analysis with the ecosystem's own tools, then checks it against the
published measurements and against a full-wave FDTD run:

```
QE PAW GaAs SCF+NSCF (8x8x8, 20 bands)           on disk: workflows/gaas-emi-shield (March 2026)
  -> p2y SAVE                                      on disk
  -> yambo IP-RPA + 0.907 eV scissor (0.513 -> 1.42 eV)            run here (2 s)
  -> Lorentz fit with the static sum rule (lib/converters)          run here
  -> + Drude free carriers: n = 1e19 cm^-3, m* = 0.063 m_e (PBE),
     gamma from each membrane's measured van der Pauw conductivity  run here
  -> transfer matrix: SE 65-85 GHz (200/500/1000 nm), IR fringes,
     1 um GaAs / 125 um Kapton / 1 um GaAs                          run here
  -> MEEP 1D FDTD of the 1000 nm membrane at 65-85 GHz              run here (~80 s)
  -> comparison with the measured SE and the paper's thin-film formula
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py          # ~80 s; --skip-meep for ~5 s
```

## Results (2026-10-14, all checks pass; results/RESULTS.md)

| membrane | SE, transfer matrix @ 75 GHz | thin-film formula | paper | measured (65-85 GHz) |
|---|---|---|---|---|
| 200 nm | 14.78 dB | 14.78 | 14.78 | 13.5 (12-15) |
| 500 nm | 19.37 dB | 19.37 | 19.37 | 19.0 (17-21) |
| 1000 nm | 24.13 dB | 24.13 | 24.13 | 24.0 (22-26) |
| GaAs / Kapton / GaAs | 39.90 dB | - | - | 38-40 |

- MEEP 1D (4000 px/mm, DC-conductivity medium) reproduces the transfer matrix to 0.10 dB over
  65-85 GHz and gives 24.23 dB at 75 GHz for the 1000 nm membrane. This also pins MEEP's conductivity
  unit convention: `sigma_D = sigma / (eps0 eps_r) * a / c`.
- Yambo IP-RPA + scissor puts the E0 edge at 1.42 eV and shows E1 (2.75, 3.05 eV) and E2
  (4.19, 4.59, 5.11 eV) critical points where experiment has 2.9 and 4.7 eV.
- eps1(0) = 20.2 against 10.9 in experiment, so n = 4.5 against 3.4 in the IR: IP with a rigid scissor
  on this mesh inflates the Gamma-point edge (eps2 = 52 at E0 where experiment has ~3). Its
  Lorentzian-broadened tail also leaks absorption below the gap, so the fitted interband model is
  opaque where the membrane is transparent. The sub-gap IR curves therefore use the fit's static eps
  plus Drude (as the paper does with eps = 13.1): T = 0.18-0.99 with 1 / 3 / 6 Fabry-Perot fringes
  above 950 nm for 200 / 500 / 1000 nm, consistent with 2nd/lambda. BSE on a denser k-grid is the
  known fix for the index.

## What is first principles and what is measured

The band structure, the interband eps(w), the effective mass and the gap target come from the
calculations. The carrier density (nominal doping) and the DC conductivity of each membrane are
measurements, and they set the scattering rate. At 75 GHz the skin depth (5-6 um) is far larger than
the membrane, so SE is set by Z0 sigma d: the mm-wave test exercises the Drude + transfer-matrix + FDTD
chain, not the band structure. The IR window is where the ab initio eps(w) matters, and where its
present limitation (IP + scissor) shows.

## Files

- `run.py` driver, `emi_lib.py` (Drude, characteristic-matrix TMM, SE; same conventions as
  MoreThanSkinDeep/mtsd, to be merged into lib/), `meep_mmwave.py` (1D MEEP with a conductive slab)
- `results/` (gitignored): Yambo outputs, `material_interband.json`, MEEP spectrum, `summary.json`,
  `RESULTS.md`, `gaas_emi_membrane.png`
