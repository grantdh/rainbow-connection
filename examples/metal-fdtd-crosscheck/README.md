# Example 3: three FDTD engines, one problem

Cross-checks the two Swift/Metal FDTD programs against MEEP and against analytic cavity modes, without
writing anything into their repositories (patched copies and binaries go to `results/bin`):

1. **LoopFDTD validation suite**: `Sources/{FDTDSimulation,FDTDValidation,RunValidation}.swift` built
   against `Shaders/FDTD.metal` (the runner is compiled as `main.swift`, the only file swiftc allows
   top-level code in) and run: plane wave, Hertzian dipole, resonant cavity, PML.
2. **Empty PEC box** (1.6 x 1.6 x 3.2 m, 64 x 64 x 128 cells): a soft source at the centre on LoopFDTD,
   MEEPhistopheles and MEEP 3D. Every resonance must be an analytic TM_mnp mode (m, n odd; p even) of the
   box the engine actually solves: N dx for MEEP, (N-1) dx for the Swift programs (coarse 2 % gate; the
   precise test is `../yee-dispersion`).
3. **Gap-fed dipole in that box** on the same three engines, the Swift programs both as shipped and
   with a one-line gap fix (see Findings), MEEP mirroring the Swift cell pattern exactly.
4. `analyze.py`: normalised gap-field spectra, resonance positions, throughput table, `crosscheck.png`.

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py        # ~40 s; --skip-meep for the Swift-only part
```

## Results (2026-10-15; results/RESULTS.md)

| engine | fields | throughput (Mcells/s, 64x64x128, 2000 steps, machine shared) | empty-box resonance (2000-step FFT peak) vs the continuum TM(3,1,0) of the box it solves |
|---|---|---|---|
| LoopFDTD (Swift + Metal) | Ex Ey Ez Hx Hy Hz | ~1.0k | 293.9 MHz vs 301.2 MHz of the (N-1) dx box: 2.3 % low, **fails** the 2 % gate (boundary-guard defect, see below) |
| MEEPhistopheles (Swift + Metal) | Ez Hx Hy only | ~1.5k | 304.4 MHz vs 301.0 MHz of the (N-1) dx box (2D): 1.2 %, within the coarse estimator's resolution |
| MEEP 1.31 (C++, CPU, FP64) | full Yee | ~240 | 293.7 MHz vs 296.3 MHz of the N dx box: 0.9 % low (numerical dispersion at lambda/40) |

Gated checks that pass: MEEP and MEEPhistopheles resonances match the analytic modes of their boxes to
< 2 %. Gated checks that fail, kept red on purpose: "LoopFDTD validation suite: all tests pass" (1 passed,
2 failed, 1 warning) and "loopfdtd_empty: resonances match analytic TM modes of its box". The precise
version of this test (32,000 steps, matrix-pencil lines against the discrete Yee dispersion relation) is
`../yee-dispersion`: there MEEP, MEEPhistopheles and a guard-fixed LoopFDTD land on their boxes' Yee
modes to 1e-5 %, and LoopFDTD as coded is 1 % off in a mode-dependent way.

The four engine-vs-engine comparison rows (empty-box mode weights, gap-fixed and as-shipped dipole,
MEEPhistopheles 3-field model) are filed with `checks.report()`: reported, not gated, not counted. The
`results/RESULTS.md` on disk (2026-10-15) predates that change and lists them as PASS (20 PASS, 2 FAIL); on
the same data the next run prints 16 gated checks pass, 2 fail, 4 reported.

## Findings

- **Gap bug in both Swift dipole programs.** `gapCells` is 1 for a 1 cm gap on 2.5 cm cells, and
  `gapCells/2` is integer 0, so the upper arm (dz >= 0) and the lower arm (dz <= 0) both own the dz = 0
  cells: the feed gap is closed and the programs simulate a solid 21-cell rod (105 PEC cells). The fix
  is `(gapCells+1)/2` (100 PEC cells, two 10-cell arms). The example applies it to generated copies in
  `results/bin/src/*_fixed.swift`; the originals are untouched. Files: `LoopFDTD/dipole_s11.swift:145,148`
  and `MEEPhistopheles/src/metal/dipole_s11.swift:129,130`.
- The arm cross-section is the 5-cell "+" pattern (`sqrt(di^2+dj^2) <= 1` cell), not the 2 mm wire the
  banner prints.
- **The LoopFDTD kernel has a boundary defect, and its validation suite would not catch it.** The E
  update's single guard `if (i == 0 || i >= Nx-1 || ...) return;` skips all three components on both
  outer planes; with the kernel's staggering (Ex at i-1/2, Ey at j-1/2, Ez at k-1/2) that also freezes
  the normal component half a cell inside the far walls, so the box is not a PEC cavity of any size (an
  earlier version of this example compared it with the N dx box and passed by accident at 0.8 %). The
  per-component guard fix and the proof that it makes the kernel exact are in `../yee-dispersion`. The
  suite's cavity test reports 23 GHz on a 707 MHz cavity (its frequency extraction), the PML test
  measures -10 dB against a -40 dB target, and the dipole test cannot measure a current.
  `QUICK_START.md` says all four pass; that is stale.
- MEEPhistopheles' dipole program evolves only Ez, Hx, Hy, so it is not the same physics as a 3D Yee
  solver; it is listed for throughput and as a 2D-mode check only.
- Which cavity modes dominate differs between the Swift programs (soft E injection) and MEEP (current
  sheet), so spectra are compared by resonance position, not by weight. With the dipole present, a
  5-cell PEC rod is defined cell-wise in Swift and Yee-position-wise in MEEP, which shifts resonances by
  ~10 %; that comparison is reported, not gated. Probes are uncalibrated in all programs, so no absolute
  S11 is claimed.

## Files

- `run.py` driver, `meep_dipole_box.py` (MEEP mirror; `--empty`, `--as-shipped`), `analyze.py`
- `results/` (gitignored): `validation/validation.log`, `bin/src/*.swift` (patched copies),
  `<engine>_<variant>/timedomain.txt`, `comparison.json`, `crosscheck.png`, `summary.json`, `RESULTS.md`
