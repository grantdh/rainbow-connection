# Example 4: plane wave on a dielectric sphere — MEEP 3D against the exact Mie series

The simplest 3D scattering problem with a closed-form answer, and the first FDTD rung of the ladder:

```
mie_theory.py  exact Mie series (Bohren & Huffman ch. 4), numpy/scipy only
                 checked against B&H's published sample run (Qsca = 3.10543), the Rayleigh limit,
                 and the large-sphere extinction asymptote 2 + 1.992 x^(-2/3)
meep_mie.py    MEEP 3D: Ez plane wave through a PML-bounded cube, six flux planes around the sphere,
                 empty-cell normalisation (load_minus_flux_data), two mirror planes; the source pulse is
                 1.5x wider than the monitored band (--src-bw)
run.py         n = 2 sphere, x = 2 pi r / lambda from 2 to 8, at 10, 20 and 40 pixels per radius
                 -> checks, plot, RESULTS.md, summary.json
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py           # resolutions 10, 20 and 40
~/miniconda3_arm64/envs/meep_arm/bin/python run.py --quick   # 6 and 10: plumbing only, gates still apply
```

`meep_mie.py` also handles a Drude-metal sphere (`--drude-fp/--drude-gamma/--drude-eps-inf`) and then
measures absorption as well as scattering; `../antenna-heating` uses that to validate MEEP's dispersive
material against Mie theory with a complex, frequency-dependent index.

## What it checks

| stage | check | gate (fixed before the first run) |
|---|---|---|
| theory | Bohren & Huffman App. A sample (m = 1.55, x = 5.2128) | Qsca = 3.10543 +/- 1e-5 |
| theory | Rayleigh limit at x = 0.05 | ratio 1 +/- 1e-3 |
| theory | large-sphere extinction, m = 1.33 + 0.05i, x = 1000 | 2 + 1.992 x^(-2/3) +/- 1e-3 |
| MEEP | Q_sca(x) on the finest grid | rel rms < 5 % of mean Q_sca |
| MEEP | error shrinks under the last 2x refinement | error ratio > 1.5 |
| MEEP | strongest resonance position | < 2 % + one frequency bin |

The large-sphere check first used "Q_ext = 2 +/- 0.03 at x = 500" for a lossless sphere and failed with
2.0304: that criterion was wrong, not the code. Extinction approaches 2 through the diffraction edge term
1.992 x^(-2/3) (0.032 at x = 500) plus interference ripple for lossless spheres. The check now uses the
documented asymptote on an absorbing sphere (residual 3e-4 at x = 1000).

The first run (2026-10-15) used 10 and 20 px per radius, expecting a few percent at 20 (7.9 px per
in-sphere wavelength at x = 8, as in MEEP's own tutorial). It measured 19.6 % and 7.3 %: converging
(ratio 2.68) and with the strongest resonance at x = 2.122 against 2.113, but above the 5 % gate. The gate
was kept and a 40 px per radius run added; the 5 % gate applies to the finest grid.

The 40 px run then failed too: 8.0 %, and no better than 20 px (error ratio 0.91). The error had moved
to the band edges (MEEP/Mie = 0.81 at x = 2 and 1.24 at x = 8, against 0.997-1.027 for 2.5 < x < 7):
the Gaussian pulse had fwidth equal to the monitored band, which leaves exp(-pi^2) ~ 5e-5 of the peak
power at the band edges, and there the spectra sat on the noise floor at every resolution. With the pulse
1.5x wider (about 1 % of the peak power at the edges) the same gates pass: 19.8 %, 7.0 % and 2.0 % at
10, 20 and 40 px per radius, error ratio 3.43 for the last doubling (second order is 4), strongest
resonance at x = 2.122 vs 2.113.

Results: `results/RESULTS.md` after a run. Wall times there are observations on a shared machine.
