# Example 6: Metal FDTD kernels against the exact discrete Yee dispersion relation

A rectangular PEC box has analytic TM_mnp modes, and on a Yee grid those modes remain exact
eigenmodes with frequencies given by the discrete dispersion relation

    (2/dt) sin(w dt/2) = c sqrt( sum_i [(2/dx) sin(k_i dx/2)]^2 ),   k_i = m_i pi / L_i .

A correct kernel therefore puts every resonance on this relation to the precision of the frequency
estimator, slightly below the continuum value (numerical dispersion). The example rings an empty
64 x 64 x 128 box (dx = 2.5 cm) for 32,000 steps with the programs' soft Gaussian-modulated 300 MHz
source (centre 33 ns; the envelope spread is shortened in the generated copies from 3 periods to 0.5
periods = 1.67 ns, so modes from ~120 to ~480 MHz are driven above 10 % instead of only those within
+/-40 MHz of 300 MHz; the centre at exactly 10 periods makes the net injected charge vanish) and
analyses only the ring-down (t > 90 ns, long after the drive) with a
matrix-pencil harmonic inversion (Hua and Sarkar 1990), which resolves lines closer than an FFT bin
and returns each line's damping. Four engines:

| engine | what runs | box it must solve |
|---|---|---|
| `loopfdtd` | LoopFDTD `dipole_s11.swift`, conductor removed, as coded | none: see below |
| `loopfdtd_fixed` | same with a per-component E boundary guard (patch shown in `run.py`, `GUARD_FIX_*`) | (N-1) dx Yee box |
| `meephistopheles` | MEEPhistopheles `src/metal/dipole_s11.swift`, conductor removed (Ez/Hx/Hy only) | (N-1) dx box, 2D TM modes (p = 0) |
| `meep` | MEEP 3D, metallic cell boundary, one Yee step per sample (`../metal-fdtd-crosscheck/meep_dipole_box.py --empty`) | N dx Yee box |

Why the boxes differ: MEEP's metallic boundary is the cell surface, so its box is N dx (checked in 1D:
at resolution 20 and Courant 0.5 the cavity modes are 0.49961, 0.99691, 1.48955, 1.97517 c/L against
the Yee prediction 0.49961, 0.99691, 1.48955, 1.97518; continuum 0.5, 1.0, 1.5, 2.0). The Swift
programs hold tangential E to zero on the outermost node planes, so their walls sit at 0 and (N-1) dx.

The LoopFDTD finding: its single guard line `if (i == 0 || i >= Nx-1 || ...) return;` skips all three
E components on both outer planes. On the Yee staggering the kernel uses (Ex at i-1/2, Ey at j-1/2,
Ez at k-1/2) that also freezes the normal component half a cell inside the far wall (Ex at i = Nx-1,
Ey at j = Ny-1, Ez at k = Nz-1), so the box is not a PEC cavity of any size and its lines sit
0.5-1.5 % off the (N-1) dx Yee modes in a mode-dependent way. `loopfdtd_fixed` replaces the guard by
three per-component guards (a component is held at zero only on the walls it is tangential to) and
must then land on the (N-1) dx modes exactly. The as-coded deviation is reported, not gated.

Per-engine details that matter at this precision: LoopFDTD uses c = 3e8 m/s exactly (eps0 derived from
mu0 and c0), MEEPhistopheles uses eps0 = 8.854e-12 (c = 2.99796e8), MEEP uses 299792458 m/s; the Swift
programs print a float32 time axis, so the drivers rebuild t = step * dt and gate the recorded axis
against it (2e-6 of the record for float32, 1e-9 for MEEP).

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--steps N] [--skip-meep]   # ~5 min, MEEP dominates
```

## Gates

Lines at or above 10 % of the strongest are gated; lines down to 3 % are reported. Poles closer than
0.3 % form a cluster whose gated line is the least-damped pole (the source sits half a cell off the box
centre, so even-parity modes are weakly excited and show up as faint poles next to strong lines; the
Hann amplitude cannot separate them). On synthetic 32,000-step sums of Yee modes with float32 round-off
the pencil places the gated lines to 5e-4 % and places 0 % of them within 0.05 % of a mode of the
wrong-size box. Losslessness is not tested with a half-record rms ratio: mode pairs 0.3 % apart beat
with a 2.4 us period, longer than the record.

| check (per engine) | criterion |
|---|---|
| uniform time axis | recorded t equals step * dt (one Yee step per sample) |
| every gated line is a discrete Yee mode of the expected box | < 0.05 %, at least 3 lines |
| the other box is excluded | < 50 % of gated lines within 0.05 % of a mode of the other box (all modes differ by 1/64) |
| the (3,1,0) line is present | a line within 0.05 % of its Yee frequency (the matcher alone cannot fail on a missing line) |
| poles set aside by the cluster rule | also on Yee modes of the box, any parity, < 0.1 % (weak even-parity modes from the half-cell source offset, or close odd-parity neighbours) |
| the isolated (3,1,0) line is undamped | |damping| < 0.01/us (nearest other mode 4 % away) |
| undamped model fits the ring-down | relative residual < 5 % with all poles at zero damping, and below the fit with the pencil's dampings |
| MEEP only: numerical dispersion resolved | continuum formula worse than the Yee formula by > 3x |
| LoopFDTD as coded | deviation from the (N-1) dx box reported (finding) |

## Results

29 gated checks pass, 0 fail, 1 reported (2026-10-15, 155 s, machine shared with another session; the
as-coded LoopFDTD deviation is filed with `checks.report()`, listed as PASS in the `results/RESULTS.md` on
disk and printed as REPORTED by the next run). Worst error of the gated
lines against the discrete Yee modes of the expected box, and against the other box size:

| engine | box | gated lines | worst error, expected box | worst error, other box | (3,1,0) damping (1/us) | undamped-model fit residual |
|---|---|---|---|---|---|---|
| loopfdtd_fixed | (N-1) dx | 15 | 0.0003 % | 1.59 % | +0.0003 | 0.0066 (damped model 0.041) |
| meephistopheles | (N-1) dx, 2D | 4 | 0.0000 % | 1.59 % | -0.0001 | 0.0033 |
| meep | N dx | 11 | 0.0019 % | 1.56 % | +0.0008 | 0.018 |
| loopfdtd as coded | none | 16 | 1.32 % below the (N-1) dx modes, mode-dependent (-1.3 to +0.6 %) | 2.17 % | | (finding, not gated) |

MEEP's lines are 0.145 % off the continuum formula and 0.002 % off the Yee formula: the numerical
dispersion of the scheme is resolved and matches the discrete relation. The poles set aside by the
cluster rule (six for the fixed LoopFDTD, five for MEEP) all lie on Yee modes of the box too, within
0.03 %. Per-engine line tables are in `results/RESULTS.md`.
