# Example 1: ab initio optics of a silicon thin film

One command runs the whole MUPPET chain on a 2-atom Si cell and checks every handoff:

```
QE pw.x SCF ──┬─> apple-bottom parity leg: the same SCF on the frozen apple-bottom pw.x,
              │   AB_MODE=cpu and AB_MODE=gpu, energies compared, AB_PROFILE JSON kept
              └─> QE NSCF (20 bands, Gamma-centred 4x4x4) ─> p2y ─> yambo setup ─> yambo IP-RPA + 0.65 eV scissor
                  ─> Lorentz fit with the static sum rule (lib/converters/yambo_eps_to_meep.py)
                  ─> analytic Airy T/R/A of a 500 nm film  and  MEEP 1D FDTD at two resolutions
                  ─> checks, plot, RESULTS.md
```

and `run_studio.py` replays the same chain as a muppet-studio flowgraph (headless executor, cached nodes).

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py            # ~30 s on an M2 Max
~/Dev/Claude/muppet-studio/.venv/bin/python run_studio.py       # same graph through muppet-studio
```

Options: `--skip-ab` (no apple-bottom leg), `--skip-meep`. Solver paths come from `../common/exlib.py`
(env.sh variable names). Before each solver step the driver waits while another session's `pw.x`/`yambo`
is running and records every run in `../common/solver-runs.log`.

## What it checks (all PASS on 2026-10-14, see results/RESULTS.md)

| stage | check |
|---|---|
| QE | the reference pw.x (`/usr/local/bin/pw.x`, OpenBLAS) is verified apple-bottom-free with `nm`; SCF reproduces the March energy to 1e-6 Ry (-22.83836927 Ry) |
| apple-bottom | frozen pw.x gives the same energy in AB_MODE=cpu and AB_MODE=gpu, and the same as the apple-bottom-free reference |
| Yambo | the scissor moves the reported direct gap by exactly 0.65 eV; the eps2 onset sits at the scissored gap |
| converter | eps_inf + sum(sigma) = eps1(0) exactly; fit rms within 15 % of max\|eps\| |
| optics | Airy T+R+A conservation; MEEP converges to Airy (rms dT 0.018 -> 0.004 from 200 to 800 px/um) |

## What the numbers mean

- eps1(0) = 21.9 against 11.7 in experiment: IP-RPA with a rigid scissor on a coarse Gamma-centred mesh
  overshoots the static response (13.4 on the shifted 4x4x4 mesh, so the number is k-sampling sensitive).
  BSE on a denser k-grid is the known fix (docs/LESSONS_LEARNED_PIPELINE.md).
- The muppet-studio replay (`run_studio.py`) reproduces the script's fit and T/R to 5e-7: the flowgraph
  nodes and the script call the same converter on the same solver outputs, and the executor's cache
  makes the second run instant (0.3 s).
- The apple-bottom leg is a correctness gate, not a speed test. On a 2-atom cell every GEMM is tiny
  (avg 303 x 4 x 206 complex), so the forced-GPU run is slower (4.5 s of BLAS against 27 ms on the CPU
  path) while the energies agree to all printed digits. That is the expected regime behaviour; the times
  are observations on a shared machine, not a benchmark, and the frozen build's apple-bottom commit is
  not pinned (its PROVENANCE.txt says so).
- Note that `~/qe-test/q-e-qe-7.4.1/bin/pw.x`, which env.sh calls the production QE, is relinked with
  static apple-bottom (7 `_ab_*` exports); it is not vanilla. The example therefore uses
  `/usr/local/bin/pw.x` as the apple-bottom-free reference (override with `MUPPET_QE_REF`).
- MEEP's residual error against the analytic slab result halves per resolution doubling (first-order
  interface error of the dispersive medium), which is how you know the FDTD leg is converged rather
  than merely "close".

## Files

- `run.py` driver, `meep_thin_film.py` (1D MEEP with a Lorentz-pole medium), `run_studio.py` (flowgraph replay)
- `results/` (gitignored): QE and Yambo outputs, `material.json`, MEEP spectra, `summary.json`, `RESULTS.md`, `si_thin_film.png`
