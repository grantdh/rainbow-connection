# Worked examples: several MUPPET tools linked end to end

Each example is one driver script that runs real solvers, hands data from one tool to the next through
files, and ends with pass/fail checks plus a `results/RESULTS.md`. They are the smoke tests of the
ecosystem: if an example breaks, a handoff broke.

| example | chain | wall | status (2026-10-15) |
|---|---|---|---|
| [si-thin-film](si-thin-film/) | QE -> apple-bottom parity -> p2y -> Yambo IP-RPA -> Lorentz fit -> Airy + MEEP 1D; replayed in muppet-studio | ~30 s | all checks pass |
| [gaas-emi-membrane](gaas-emi-membrane/) | Yambo IP-RPA (on the GaAs SAVE) -> Lorentz fit -> Drude -> transfer matrix (mm-wave SE, IR fringes, multilayer) -> MEEP 1D at 65-85 GHz -> published measurements | ~80 s | all checks pass (SE within 1.3 dB of measurement; MEEP = TMM to 0.10 dB) |
| [metal-fdtd-crosscheck](metal-fdtd-crosscheck/) | LoopFDTD validation suite; empty-box cavity modes vs analytic and the PEC-box dipole on LoopFDTD, MEEPhistopheles and MEEP 3D (as shipped and with a gap fix) | ~40 s | 2 true failures kept red: LoopFDTD's own validation suite (2 of 4 tests) and LoopFDTD's empty box, which is not a Yee cavity of any size (boundary-guard defect, see yee-dispersion); MEEP and MEEPhistopheles pass |
| [si-eos-bands](si-eos-bands/) | QE E(V) at 9 volumes -> Birch-Murnaghan vs published PBE and all-electron values -> the same 9 SCFs on the frozen apple-bottom pw.x (AB_MODE=gpu) -> bands at a_exp vs textbook PBE gaps | ~5 min | all checks pass (a0 5.4695 A, B0 88.7 GPa, B0' 4.31; apple-bottom identical to 1e-8 Ry with 115,101/115,101 BLAS calls on the GPU; gaps 0.575 / 2.563 eV) |
| [si-sum-rules](si-sum-rules/) | QE NC-pseudo SCF + NSCF -> p2y -> Yambo retarded IP-RPA with 10-80 bands, plus a control without the non-local commutator -> f-sum, Kramers-Kronig, loss function, plasmon, gaps vs QE | ~1 min | all checks pass (KK closes to 0.0 %; gaps equal QE to < 0.1 meV; f-sum 1.169 of (hbar wp)^2 with the full velocity, 1.387 with p/m alone) |
| [yee-dispersion](yee-dispersion/) | Empty PEC box on LoopFDTD (as coded and guard-fixed), MEEPhistopheles and MEEP; matrix-pencil ring-down lines vs the discrete Yee dispersion relation of the box each engine solves | ~3 min | all checks pass (fixed LoopFDTD, MEEPhistopheles and MEEP on their boxes' Yee modes to 3e-4 %, 0 % and 2e-3 %; LoopFDTD as coded 1 % off, reported) |
| [skin-effect](skin-effect/) | Transfer matrix vs Schelkunoff R + A + B and the classical skin-depth, reflection and thin-film limits -> MEEP 1D slab at 65-85 GHz | ~3 min | all checks pass (identity to 7e-15 dB; MEEP = transfer matrix to 0.006 dB at 45 dB SE) |
| [mie-scattering](mie-scattering/) | Exact Mie series (checked against Bohren & Huffman) -> MEEP 3D plane wave on an n = 2 sphere at 10, 20 and 40 px per radius; replayed in muppet-studio (workflow 4) | ~17 min | all checks pass (2.0 % rms at 40 px/r, error 3.4x smaller per doubling; strongest resonance x = 2.122 vs 2.113) |
| [dipole-antenna](dipole-antenna/) | Induced-EMF impedance and sinusoidal-current pattern -> MEEP 3D Hertzian dipole and gap-fed half-wave PEC wire at 100 and 200 px/lambda; workflow 5 | ~15 min | 2 failures kept red: arm current depends on the loop size by 4.5 % (gate 2 %); R at resonance 75.2 vs 52.5 ohm from induced EMF (gate 20 %; the feed is not converged). Hertzian checks, feed power = radiated power (0.5 %), directivity, pattern and resonance (0.873 vs 0.889 f0) pass |
| [patch-antenna](patch-antenna/) | Transmission-line design of Balanis Ex. 14.1 -> MEEP 3D probe-fed patch: Harminv, probe power, near2far; workflow 6 | ~5 min | 1 failure kept red: TM010 9.27 GHz vs 9.99 GHz from the model (gate 3 %; MEEP settles at 9.1-9.3 GHz from 4 to 8 cells across h). Probe power = radiated power (0.986), broadside, D = 8.2 dBi pass |
| [antenna-heating](antenna-heating/) | QE GaAs bands near Gamma -> Kane m* -> Drude eps of 1e19 cm^-3 GaAs -> MEEP Drude sphere vs Mie -> MEEP 3D GaAs dipole (transmitting) and rod (receiving) -> heat per length -> 1D conduction T(z); workflow 7 | ~1.8 h | 6 failures kept red: Kane m* 0.089 vs 0.067; Drude sphere 17 % / 33 % at 10 px/um (converging at first order, 8 % / 15 % at 20); transmit efficiency and receive resonance (peak and frequency) not converged at 15 px/um. Passing: QE gap and isotropy, transmit energy balance (0.3 %), receive absorption from Yee E.D = box flux (0.06 %). At 15 px/um the rod peaks at 12.0 THz, sigma_abs 28 um^2; under 1 kW/cm^2 it absorbs 285 uW and its middle warms 13.8 K |

## Running

```bash
cd examples/<name>
~/miniconda3_arm64/envs/meep_arm/bin/python run.py
```

The meep_arm interpreter is used for every driver because it carries numpy, scipy, matplotlib and MEEP.
Paths to pw.x, yambo, p2y, the frozen apple-bottom build and muppet-studio are resolved in
`common/exlib.py` from the same variables as `../env.sh` (`SF_QE_BIN`, `SF_YAMBO_BIN`,
`MUPPET_MEEP_PYTHON`, `MUPPET_AB_FROZEN`, `MUPPET_STUDIO`), plus `MUPPET_QE_REF` for the
apple-bottom-free reference pw.x (`/usr/local/bin/pw.x`; the drivers verify it with `nm`, since the
`~/qe-test/q-e-qe-7.4.1` binary is relinked with static apple-bottom).

## Machine etiquette

`exlib.run()` wraps every solver call: it waits while `~/Dev/.timing-lock` exists (another session's
timing campaign; honoured in every mode), serialises through `~/qe-test/xp/tools/machlock.py bench` when
that tool is present, runs under `nice -n 10`, and appends a line to `common/solver-runs.log` (time, rc,
wall, 1-minute load, whether the machine was idle, which lock applied). `MUPPET_SHARE=1` lets untimed
correctness runs share the machine with other sessions (no machlock, no wait for their solvers; see
`~/Dev/PROTOCOL.md` section 9); none of the examples' gates depend on wall time. The frozen
apple-bottom build under `~/qe-test/ab-frozen/<hash>/` is used instead of the library repo's `build/`,
which other sessions rebuild.

## Conventions

- `results/` is gitignored; `RESULTS.md` and `summary.json` inside it are the record of a run.
- Every check prints `[PASS]`/`[FAIL]` with the value and the criterion; drivers exit 1 on any failure.
- Every number in a README is tagged with where it was measured; the comparison targets (reference
  energies, published measurements) are declared at the top of each driver.
- Nothing is written into other repositories: Swift programs are compiled into `results/bin`, and
  patched copies are generated at run time with the patch visible in the driver.
