# Example 6: probe-fed microstrip patch — MEEP 3D against the textbook models

Balanis, *Antenna Theory* (4th ed.), Example 14.1: a 10 GHz patch on a 1.588 mm, eps_r = 2.2 substrate,
W = 1.186 cm, L = 0.906 cm.

```
patch_theory.py  transmission-line model (eqs. 14-1, 14-2, 14-5..14-7) and cavity-model edge resistance
                   1 / (2 (G1 + G12)) (14-12, 14-17, 14-18); self-test reproduces Example 14.1 and R_edge = 228 ohm
meep_patch.py    MEEP 3D: PEC ground and patch sheets 1.5 cells thick on a 3 cm substrate exactly px_per_h
                   cells thick, ideal probe a quarter-length from the feed edge; Harminv at the far radiating
                   edge, the power the probe delivers (complex Poynting flux out of a small box around it),
                   near2far patterns and directivity; mirror plane y = 0
run.py           checks, plot, RESULTS.md, summary.json
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--px-per-h 4]
```

## What it checks (gates fixed before the first run)

| check | gate | why |
|---|---|---|
| design reproduces Example 14.1 | W, L within 0.002 cm | the formulas are typed correctly |
| Harminv finds a cavity mode in 7.5-12.5 GHz | at least one | the patch resonates |
| TM010 resonance vs transmission-line model | +/- 3 % | the model's accuracy for h << lambda |
| broadside radiation | U(0) / U_max > 0.95 | a TM010 patch radiates broadside |
| input power vs far-field power | Re(S_feed) / P_far = 1 +/- 5 % | nothing in the model is lossy |

There is no absolute input impedance. An ideal line current between two conductors has no clean
terminal current (a loop around it also encloses the cavity's displacement current), so the probe is
characterised by the power it delivers per unit source strength, which is proportional to R_in and
peaks at the resonance seen from the feed. A coax-pin port model is the way to an absolute Z_in.

## How the simulation is built, and why

- **Metal on grid planes.** The first run (2026-10-14) used one-cell PEC sheets with subpixel averaging.
  Averaging blended them into the dielectric, and the eps profile through the patch read 1.5, 6.6, 2.2,
  ..., 3.3, 3.0 with no metal anywhere: it had simulated a bare substrate. The sheets are now 1.5 cells
  thick, overlap their surface plane by a quarter cell, and averaging is off. The cell has an even number
  of pixels on every side because MEEP anchors its grid at the cell centre. With an odd count (197 cells in
  z at 6 px per h) the planes counted in from the boundary fall half a cell off the grid, and the metal
  vanished again.
- **Harminv's Q threshold.** MEEP's Harminv discards modes with Q < 50 by default. This patch's TM010 has
  Q of about 10, so the second run reported no modes at all. The threshold is now 1.
- **The feed box.** Its side faces first ended on the metal planes. The end samples, interpolated
  half-way into the metal, then received half weight, and the box read 7-10 % short of the radiated power
  (0.90, 0.93, 0.95 and 0.96 at 3, 4, 6 and 8 cells across h, for any box size). The faces now run one cell
  into each sheet, where the fields are zero: 0.985, 0.986, 0.987 and 0.988 at 3, 4, 6 and 8 cells per h. The far field (near2far)
  agreed with a flux box around the whole antenna to 0.05 % throughout.

## What the runs show

TM010 from Harminv (same mode at both radiating edges), against 9.99 GHz from the transmission-line model:

| cells across h | 3 | 4 | 6 | 8 |
|---|---|---|---|---|
| f (GHz) | 8.765 | 9.268 | 9.095 | 9.174 |
| Q | 11.2 | 10.5 | 10.5 | 10.6 |

MEEP resonates 7-12 % below the transmission-line model at every resolution tried, and the 3 % gate
fails. From 4 cells across h onward the resonance stays within 9.10-9.27 GHz (+/- 1 %), so the
simulation has settled about 8 % below the model; the failure is not a coarse-grid artifact. The sequence
is not monotonic: with averaging off, the patch edges (L = 17.1, 22.8 and 34.2 cells
at 3, 4 and 6 px per h) snap to the grid differently at each resolution, which moves the effective
length by up to a cell (5.8 % of L at 3 px per h). The patch sheet is also 1.5 cells thick (0.04-0.08 cm
at these grids, against copper's 35 um), and its raised edge adds fringing capacitance. Both are
plausible contributors, but these runs do not separate them from the transmission-line model's own
error. A run with the patch dimensions snapped to whole cells (so the model can be evaluated for the
geometry actually simulated) and thinner sheets would.

Results: `results/RESULTS.md` after a run. Wall times there are observations on a shared machine.
