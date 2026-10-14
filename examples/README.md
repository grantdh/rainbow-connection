# Worked examples: several MUPPET tools linked end to end

Each example is one driver script that runs real solvers, hands data from one tool to the next through
files, and ends with pass/fail checks plus a `results/RESULTS.md`. They are the smoke tests of the
ecosystem: if an example breaks, a handoff broke.

| example | chain | wall | status (2026-10-14) |
|---|---|---|---|
| [si-thin-film](si-thin-film/) | QE -> apple-bottom parity -> p2y -> Yambo IP-RPA -> Lorentz fit -> Airy + MEEP 1D; replayed in muppet-studio | ~30 s | all checks pass |
| [gaas-emi-membrane](gaas-emi-membrane/) | Yambo IP-RPA (on the GaAs SAVE) -> Lorentz fit -> Drude -> transfer matrix (mm-wave SE, IR fringes, multilayer) -> MEEP 1D at 65-85 GHz -> published measurements | ~80 s | all checks pass (SE within 1.3 dB of measurement; MEEP = TMM to 0.10 dB) |
| [metal-fdtd-crosscheck](metal-fdtd-crosscheck/) | LoopFDTD validation suite; empty-box cavity modes vs analytic and the PEC-box dipole on LoopFDTD, MEEPhistopheles and MEEP 3D (as shipped and with a gap fix) | ~40 s | kernel gates pass (modes to < 1 %); 1 known failure kept red: LoopFDTD's own validation suite (2 of 4 tests) |

## Running

```bash
cd examples/<name>
~/miniconda3_arm64/envs/meep_arm/bin/python run.py
```

The meep_arm interpreter is used for every driver because it carries numpy, scipy, matplotlib and MEEP.
Paths to pw.x, yambo, p2y, the frozen apple-bottom build and muppet-studio are resolved in
`common/exlib.py` from the same variables as `../env.sh` (`SF_QE_BIN`, `SF_YAMBO_BIN`,
`MUPPET_MEEP_PYTHON`, `MUPPET_AB_FROZEN`, `MUPPET_STUDIO`).

## Machine etiquette

`exlib.run()` wraps every solver call: it waits while another session's `pw.x`, `yambo` or `mpirun` is
running (timing campaigns must not be disturbed), runs under `nice -n 10`, and appends a line to
`common/solver-runs.log` (time, rc, wall, 1-minute load, whether the machine was idle). The frozen
apple-bottom build under `~/qe-test/ab-frozen/<hash>/` is used instead of the library repo's `build/`,
which other sessions rebuild.

## Conventions

- `results/` is gitignored; `RESULTS.md` and `summary.json` inside it are the record of a run.
- Every check prints `[PASS]`/`[FAIL]` with the value and the criterion; drivers exit 1 on any failure.
- Every number in a README is tagged with where it was measured; the comparison targets (reference
  energies, published measurements) are declared at the top of each driver.
- Nothing is written into other repositories: Swift programs are compiled into `results/bin`, and
  patched copies are generated at run time with the patch visible in the driver.
