# Example 5: dipole antennas — MEEP 3D against closed-form antenna theory

```
antenna_theory.py  induced-EMF input impedance (Balanis eqs. 8-60a/b, 8-61), sinusoidal-current pattern
                     (4-58a) and directivity; self-test: 73.13 + j42.55 ohm and D = 1.641 for lambda/2
meep_dipole.py     --mode hertzian: point Jz source; --mode halfwave: PEC wire of radius a, length lambda0/2,
                     delta-gap current source. Box flux, near2far pattern, and for the wire the complex power
                     S out of a small box around the gap and the arm current I from Ampere loops two and four
                     cells beyond the gap, so Z_in(f) = S / |I|^2. Three mirror planes (1/8 of the cell)
run.py             checks, plot, RESULTS.md, summary.json
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py                  # both stages; half-wave at 100 and 200 px/lambda0
~/miniconda3_arm64/envs/meep_arm/bin/python run.py --skip-halfwave  # Hertzian only (seconds)
```

## What it checks (gates fixed before the first run)

| stage | check | gate |
|---|---|---|
| Hertzian | directivity | 1.5 +/- 2 % |
| Hertzian | pattern vs sin^2 theta | max deviation < 0.02 |
| Hertzian | azimuthal symmetry | < 1 % between phi = 0 and 45 deg |
| Hertzian | far-field power vs box flux | 1 +/- 2 % |
| half-wave | feed power Re(S) vs radiated power | 1 +/- 5 % for 0.75 < f/f0 < 1.25 (lossless PEC) |
| half-wave | arm current, 4-cell vs 2-cell loop | 1 +/- 2 % (added with the S/\|I\|^2 port, before its first run) |
| half-wave | directivity at l = lambda/2 | 1.641 +/- 3 % |
| half-wave | pattern vs sinusoidal-current model | max deviation < 0.05 |
| half-wave | resonance (X = 0) vs induced EMF, same l and a | +/- 5 % |
| half-wave | input resistance at resonance vs induced EMF | +/- 20 % |

The last two gates carry the induced-EMF method's own uncertainty: the method assumes a sinusoidal
current, which is exact only for vanishingly thin wires, and this wire is fat (l / 2a = 12.5). The
feed-power check is the model-free one: it ties the impedance measurement to the radiated power with no
antenna theory at all. The gates apply to the finer grid.

Power quantities follow MEEP's DFT convention (flux = Re(E* x H) with no factor 1/2), so R = Re(S) / |I|^2.

## History of the port, and what the runs show

The first run (2026-10-14, 100 px/lambda0, gap 2 cells) used Z = V/I with V the line integral of E_z
across the gap on the axis. Its feed power Re(V I*) came out 12-16 % below the radiated power and its
resonance at 0.781 f0 (R = 51.1 ohm). A diagnostic showed both V and the loop current to be ambiguous at
the feed: V differs between the axis and the gap cross-section, and a loop in the gap plane also encloses
the gap's fringing displacement current, so it grows with loop size. The port became the complex power S
out of a closed box around the gap (no voltage needed, Re S is exactly the delivered power) with the
current taken on the arm two cells beyond the gap, and the run moved to 100 and 200 px/lambda0.

The 2026-10-15 run (see results/RESULTS.md):

- passes: Hertzian directivity, pattern and power; feed power Re(S) = radiated power to 0.5 %; half-wave
  directivity 1.682, pattern, resonance 0.873 f0 vs 0.889 f0 from induced EMF;
- fails: the arm current still depends on the loop size by 4.5 % (gate 2 %). Two cells beyond a two- to
  four-cell gap the loop still encloses fringing displacement current, so |I| carries that ambiguity (and
  R about twice it);
- fails: R at resonance, 75.2 ohm vs 52.5 ohm from induced EMF at its own resonance (gate 20 %).

Neither failure is resolved. The 200 px/lambda0 grid is not converged at the feed: the resonance moved
from 0.8405 f0 to 0.8730 f0 between 100 and 200 px/lambda0 (gap 2 and 4 cells, wire radius 2 and 4 cells),
and both X and R near resonance depend on the gap's stored energy, which is part of the port. The R
comparison therefore mixes discretization error with the thin-wire model's error, and these runs cannot
separate them. A 300-400 px/lambda0 run (5-16x the 200 px wall time) or a reference from an integral-
equation solver for the same l, a and gap is the way to settle it.
