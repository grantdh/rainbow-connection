# lib/inverse — live 2D topology optimization

Adjoint inverse design of 2D silicon-photonic devices (Si core, SiO₂ cladding, out-of-plane Ez) that runs
fast enough to steer interactively: change a target, filter radius or wavelength, and the design reacts within
a fraction of a second to a couple of seconds.

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/live/server.py      # http://127.0.0.1:8770
~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/test_inverse.py     # PASS/FAIL checks
~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/run_design.py demux --out d.json   # headless
~/miniconda3_arm64/envs/meep_arm/bin/python lib/inverse/meep_check.py d.json --res 50      # MEEP check
```

Only numpy and scipy (the page server uses the standard library).

## Files

| File | Role |
|---|---|
| `fdfd.py` | FDFD Helmholtz operator with stretched-coordinate PML, slab mode solver, ports, and `Condensed` (static condensation onto the design window) |
| `devices.py` | Problem catalog: 90° bend, power splitter, TE0→TE1 mode converter, two-wavelength demux |
| `optimize.py` | Conic filter → tanh projection → permittivity; adjoint objective; Adam optimizer |
| `live/server.py`, `live/index.html` | Optimizer thread + server-sent-event stream + control page |
| `run_design.py` | Headless optimization with the page's β continuation; writes the same design JSON as "Download design" |
| `test_inverse.py` | Validation checks |
| `meep_check.py` | Independent 2D MEEP check of an exported design, against an FDFD sweep (writes `runs/`, gitignored) |

## Method

- **Objective.** Modal power transmissions T = β_out|c_out|² / (β_in|c_ref|²), where c_ref comes from a straight copy of
  the input guide. The loss is Σ w (T − target)², so maximizing, splitting and suppressing crosstalk all share one form.
- **Gradient.** One forward solve and one adjoint (transpose) solve per (wavelength, source), regardless of the
  number of design pixels: dL/dε_i = −2 Re(λ_i k₀² x_i).
- **Fabrication constraints.** A conic filter sets the minimum feature radius, and tanh projection with β continuation
  drives the design toward binary. "Auto-binarize" raises β by 1.6× whenever the loss stalls, up to β = 64,
  and then thresholds the design at 0.5 and reports its T (the "Binary" column).

## Why it is fast

A plain sparse LU of the full 170×170 grid takes ~425 ms (SuperLU, COLAMD) per solve, too slow for live use.
Two changes bring an iteration down to 25–45 ms:

1. **Static condensation.** Only the 80×80 design window changes. The fixed exterior is eliminated once
   per wavelength (`Condensed`): its LU, plus a dense ring×ring Schur correction on the ~316 boundary pixels.
   That costs about 1 s per wavelength and is cached by the server. Each step then solves only 6.4k unknowns.
   Ports, monitors and full-grid fields are linear in the window solution, so they come from precomputed vectors.
2. **Recycled factorization.** Even 6.4k unknowns take ~110 ms to refactor in SuperLU. So each step runs
   GMRES preconditioned by the LU of an earlier permittivity (0.9 ms per apply). Typical Adam steps converge in 5–11
   iterations to rtol 1e-9. The LU is refreshed only when GMRES needs more than 20 iterations.

## Measured (M2 Max, meep_arm Python, scipy 1.16)

| Check | Result |
|---|---|
| Straight 500 nm guide through the window | T = 0.99999 |
| Condensed vs full-grid LU | max rel. error 1.4e-12 |
| Adjoint vs central finite difference (6 directions) | max rel. error 4e-7 |
| Bend, 60 iterations | T 0.001 → 0.995, 44 ms/iter |
| Bend in the live page, auto-binarize | converged at iter 140: T 0.997, binary T 0.969, 4.9 % gray |
| Demux 1550/1310 nm in the live page | converged at iter ~520: binary T 0.961 / 0.973, crosstalk ≤ 0.002 |

## MEEP check (`meep_check.py`, or "Verify in MEEP" on the page)

The check rebuilds the exported binary pixel map exactly in 2D MEEP (eps_averaging off, ODD_Z, each pixel looked up a
quarter cell off its edges). It runs one broadband pulse with eigenmode decomposition at every port. T is normalized
by the forward input-mode power downstream of the source in the same run, and plotted against an FDFD sweep of the
identical structure. `lib/fdtd`'s case schema has no mode sources or decomposition, so this is a standalone script.

| Structure | MEEP px/µm | T at design λ: MEEP / FDFD (40 nm) | Notes |
|---|---|---|---|
| Straight guide (`--selftest`) | 50 | 0.999–1.000 over 1.2–1.7 µm / 1.000 | R < 7e-5 |
| Bend | 25 (same grid) | 0.968 / 0.969 | the methods agree at equal dx |
| Bend | 50 | 0.912 / 0.969 | ~5 % lower across the whole band; reversed launch (y-port → x-port) gives the same 0.912 |
| Bend | 75 | 0.980 / 0.969 | |
| Bend | 100 | 0.964 / 0.969 | |
| Bend | 150 | 0.974 / 0.969 | 300 s |
| Demux 1310 / 1550 | 25 | 0.969 / 0.961 vs 0.973 / 0.961 | |
| Demux 1310 / 1550 | 50 | 0.894 / 0.895 | passband peaks 0.972 / 0.968 at 1291 / 1540 nm |
| Demux 1310 / 1550 | 100 | 0.853 / 0.869 | passband peaks 0.970 / 0.968 at 1291 / 1534 nm |
| Demux, held over ±20 nm | 50 | 0.930 / 0.938 vs 0.925 / 0.944 | ±20 nm edges 0.86–0.95; crosstalk ≤ 0.004 |
| Demux, held over ±20 nm | 100 | 0.928 / 0.937 | ±20 nm edges 0.88–0.95; crosstalk ≤ 0.003 |

The readings:
- At equal grid spacing the two solvers agree to 0.004, so the adjoint/FDFD machinery is not the issue.
- The demux keeps its passbands at T ≈ 0.97 but they move 17–19 nm bluer once the grid is refined. That is the
  numerical dispersion of the 40 nm optimization grid, which detunes narrowband designs by ~1.2 %.
  Fix: `bandwidth` (default ±20 nm for the demux, a slider on the page) optimizes each target at λ−Δ, λ and λ+Δ.
  At the design wavelengths that holds 0.93 / 0.94 in MEEP at 50 and 100 px/µm, against 0.85–0.90 for the
  single-wavelength design. In exchange, the peak drops from 0.97 to 0.94–0.96 and each iteration costs about
  3× more (~135 ms, six solves; more threads do not help).
- The bend is broadband, so a shift barely matters. Its 25/75/100/150 px/µm values lie within 0.964–0.980, but
  50 px/µm is 5 % lower everywhere. That outlier is unexplained (it is not the y-port or the pixel sampling).
  Until it is understood, check a design at two MEEP resolutions.

## Limits

- 2D, Ez polarization only. dx = 40 nm gives about 9 cells per wavelength in Si at 1310 nm: enough to steer
  interactively, not enough to sign off a design (see the MEEP check above). 3D is not covered.
- Ports and monitors must lie outside the design window; the condensation asserts this.
- At high β, permittivity changes per step are large, so the preconditioner is refreshed more often
  (iterations then cost 100–250 ms).
