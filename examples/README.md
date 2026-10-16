# Worked examples: several MUPPET tools linked end to end

Each example is one driver script that runs real solvers, hands data from one tool to the next through
files, and ends with pass/fail checks plus a `results/RESULTS.md`. They are the smoke tests of the
ecosystem: if an example breaks, a handoff broke.

| example | chain | wall | status (2026-10-16; gated checks counted, reported rows listed separately) |
|---|---|---|---|
| [si-thin-film](si-thin-film/) | QE -> apple-bottom parity -> p2y -> Yambo IP-RPA -> Lorentz fit -> Airy + MEEP 1D; replayed in muppet-studio | ~30 s | all checks pass |
| [gaas-emi-membrane](gaas-emi-membrane/) | Yambo IP-RPA (on the GaAs SAVE) -> Lorentz fit -> Drude -> transfer matrix (mm-wave SE, IR fringes, multilayer) -> MEEP 1D at 65-85 GHz -> published measurements | ~80 s | 21 gated checks pass, 3 reported (SE within 1.3 dB of measurement; MEEP = TMM to 0.10 dB) |
| [metal-fdtd-crosscheck](metal-fdtd-crosscheck/) | LoopFDTD validation suite; empty-box cavity modes vs analytic and the PEC-box dipole on LoopFDTD, MEEPhistopheles and MEEP 3D (as shipped and with a gap fix) | ~40 s | 16 gated checks pass, 2 true failures kept red, 4 reported: LoopFDTD's own validation suite (2 of 4 tests) and LoopFDTD's empty box, which is not a Yee cavity of any size (boundary-guard defect, see yee-dispersion); MEEP and MEEPhistopheles pass |
| [si-eos-bands](si-eos-bands/) | QE E(V) at 9 volumes -> Birch-Murnaghan vs published PBE and all-electron values -> the same 9 SCFs on the frozen apple-bottom pw.x (AB_MODE=gpu) -> bands at a_exp vs textbook PBE gaps | ~5 min | 19 gated checks pass, 1 reported (a0 5.4695 A, B0 88.7 GPa, B0' 4.31; apple-bottom identical to 1e-8 Ry with 115,101/115,101 BLAS calls on the GPU; gaps 0.575 / 2.563 eV) |
| [si-sum-rules](si-sum-rules/) | QE NC-pseudo SCF + NSCF -> p2y -> Yambo retarded IP-RPA with 10-80 bands, plus a control without the non-local commutator -> f-sum, Kramers-Kronig, loss function, plasmon, gaps vs QE | ~1 min | 15 gated checks pass, 2 reported (KK closes to 0.0 %; gaps equal QE to < 0.1 meV; f-sum 1.169 of (hbar wp)^2 with the full velocity, 1.387 with p/m alone) |
| [yee-dispersion](yee-dispersion/) | Empty PEC box on LoopFDTD (as coded and guard-fixed), MEEPhistopheles and MEEP; matrix-pencil ring-down lines vs the discrete Yee dispersion relation of the box each engine solves | ~3 min | 29 gated checks pass, 1 reported (fixed LoopFDTD, MEEPhistopheles and MEEP on their boxes' Yee modes to 3e-4 %, 0 % and 2e-3 %; LoopFDTD as coded 1 % off, reported) |
| [skin-effect](skin-effect/) | Transfer matrix vs Schelkunoff R + A + B and the classical skin-depth, reflection and thin-film limits -> MEEP 1D slab at 65-85 GHz | ~3 min | all checks pass (identity to 7e-15 dB; MEEP = transfer matrix to 0.006 dB at 45 dB SE) |
| [mie-scattering](mie-scattering/) | Exact Mie series (checked against Bohren & Huffman) -> MEEP 3D plane wave on an n = 2 sphere at 10, 20 and 40 px per radius; replayed in muppet-studio (workflow 4) | ~17 min | all checks pass (2.0 % rms at 40 px/r, error 3.4x smaller per doubling; strongest resonance x = 2.122 vs 2.113) |
| [dipole-antenna](dipole-antenna/) | Induced-EMF impedance and sinusoidal-current pattern -> MEEP 3D Hertzian dipole and gap-fed half-wave PEC wire at 100 and 200 px/lambda; workflow 5 | ~15 min | 2 failures kept red: arm current depends on the loop size by 4.5 % (gate 2 %); R at resonance 75.2 vs 52.5 ohm from induced EMF (gate 20 %; the feed is not converged). Hertzian checks, feed power = radiated power (0.5 %), directivity, pattern and resonance (0.873 vs 0.889 f0) pass |
| [patch-antenna](patch-antenna/) | Transmission-line design of Balanis Ex. 14.1 -> MEEP 3D probe-fed patch: Harminv, probe power, near2far; workflow 6 | ~5 min | 1 failure kept red: TM010 9.27 GHz vs 9.99 GHz from the model (gate 3 %; MEEP settles at 9.1-9.3 GHz from 4 to 8 cells across h). Probe power = radiated power (0.986), broadside, D = 8.2 dBi pass |
| [antenna-heating](antenna-heating/) | QE GaAs bands near Gamma -> Kane m* -> Drude eps of 1e19 cm^-3 GaAs -> MEEP Drude sphere vs Mie -> MEEP 3D GaAs dipole (transmitting) and rod (receiving) -> heat per length -> 1D conduction T(z); workflow 7 | ~1.8 h | 6 failures kept red: Kane m* 0.089 vs 0.067; Drude sphere 17 % / 33 % at 10 px/um (converging at first order, 8 % / 15 % at 20); transmit efficiency not converged (junction loss); receive resonance 10 vs 15 px/um (grid phase of the rod faces). Passing: QE gap and isotropy, transmit energy balance (0.3 %), receive absorption from Yee E.D = box flux (0.06 %). Scratch runs at 12.5 and 20 px/um agree: the rod peaks near 11.45 THz with sigma_abs 24 um^2; under 1 kW/cm^2 it absorbs 236 uW and its middle warms 11.5 K |
| [si-supercell-folding](si-supercell-folding/) | QE Bloch-theorem identities on 2-, 8- and 64-atom Si cells (reference 7.4.1 at 4 ranks) -> the 64-atom cells on the frozen apple-bottom pw.x in AB_MODE=cpu and gpu (every GEMM) with the profiler -> exact fold, band unfolding, term identities, forces/stress symmetry, parity, dispatch evidence and GEMM-dimension pins; every launch through tools/abrun | 27 min measured with another session's campaign running (lock share, load1 3-10); not measured on a quiet machine | 77 gated checks pass, 0 fail, 6 reported (2026-10-16; gates restated after a first pass with 16 red rows, run-first-pass.log, each restatement with its control): the density-dependent energy folds to 4.7e-10 Ry/atom (gate 3e-9); the total-energy fold of 4.3e-8 is QE's Ewald rmax rule dropping the nearest-neighbour shell on the 64-atom cells (4.5e-8 Ry/atom, replicated and closed to 1e-12), reported; per-term identities at an empirical 1e-6 (first-order SCF scatter, control 4.0e-7); fcc-vs-cubic 5.4e-5 Ry/atom consistent with xc-grid aliasing (1.0e-5 at ecutrho 200), reported; Gamma trick vs complex 1.0e-8 Ry (gate 2e-8); Td-site forces exactly zero, the other atoms' symmetry-allowed residuals below QE's own SCF correction; unfolding exact to 0.1 meV (128 and 156 states); frozen build's final-iteration c_bands warning (4/640 at ethr 1e-12, cause not isolated; eigenvalues equal to 0.1 meV). apple-bottom cpu == gpu == reference to the last digit on R4 and R6 (21062/21062 and 80808/80808 GEMMs on the GPU, 0 fallbacks) |
| [si-forces-stress-fd](si-forces-stress-fd/) | Forces and stress vs finite differences of E: NC Si on the reference 7.4.1 (ecut scan at pinned ecutrho/grid, Nielsen-Martin stress vs -dE/dV with the Francis-Payne Pulay identity, frozen Gamma phonon, Hellmann-Feynman vs FD, S4/C3/translation identities, stress isotropy on the nsym = 1 cells) -> PAW 64-atom cell at Gamma: FD on the reference (4 ranks) and the frozen apple-bottom pw.x in AB_MODE=cpu and gpu (every GEMM) with the profiler: energy, 192 force components, stress tensor, dispatch counters; every launch through tools/abrun; --reuse matches each record's input, pw.x/dylib/pseudo sha256, mode, ranks and nk | ~22 min (+5 min of control runs) | 34 gated checks pass, 0 fail, 13 reported. Four first-pass red rows: three gates were restated to the physical identity with the original comparison kept as a reported row (the uncorrected 40-Ry stress gate fails by exactly the Pulay stress, 0.81 kbar, corrected residual 0.05; the +d-only frozen phonon is 1.5 % anharmonic at 0.02 A, the +-d estimator gives 513 cm^-1 to 0.1 %; the 0.1 A egg-box is 5.9e-6 Ry on the 48^3 grid, 0 for one grid step and 2.6e-6 at 64^3), and the B1 SCF-correction precondition was re-read from the 8-decimal per-atom block (6.4e-7, printed 1.0e-6). Reported rows also hold the by-construction identities (17-run symmetrised isotropy, the F1 = -F2 phonon pair, forces after a rigid translation). HF = FD to 2.6e-6 (NC, the O(h^2) truncation) and 7.5e-7 Ry/bohr (PAW, at the 6.6e-7 floor of the printed energies); cpu = gpu_all to 1e-8 Ry, 2e-8 Ry/bohr, 0.00 kbar with 43,491/43,491 GEMMs on the GPU; ab_cpu = 7.4.1 reference to 1e-8 Ry |

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

Drivers that launch pw.x through `tools/abrun` (`runner.run_sync`; si-supercell-folding, si-forces-stress-fd) get the same etiquette from
abrun's runner (timing lock polled in every mode, `share` by default, `os.nice(10)` in its shim, one line per run in
`common/solver-runs.log`) and keep abrun's run records (`run.json`, `summary.json`, `ab_profile.json`, `machine.json`)
under their own `results/`; the library's counters, not the mode flag, decide what ran on the GPU.

## Conventions

- `results/` is gitignored; `RESULTS.md` and `summary.json` inside it are the record of a run.
- Every check prints `[PASS]`/`[FAIL]` with the value and the criterion; drivers exit 1 on any failure.
- Every number in a README is tagged with where it was measured; the comparison targets (reference
  energies, published measurements) are declared at the top of each driver.
- Nothing is written into other repositories: Swift programs are compiled into `results/bin`, and
  patched copies are generated at run time with the patch visible in the driver.
