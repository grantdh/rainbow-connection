# CLAUDE.md

## Project Identity

**MUPPET: Rainbow Connection** (Multiphysics Unified Pipeline for Photonic Engineering and Technology)

This is a multiphysics orchestration project that connects quantum mechanics solvers (QE, Yambo) to electromagnetic simulation (MEEP) and thermal analysis (Elmer). It runs on a single Apple Silicon Mac. The project is pure orchestration — it does NOT contain solver code. It contains converters, workflow scripts, and configuration that tie existing solver installs together.

## Owner

Grant Heileman, UNM ECE, graduating 2026. Principal engineer level. Prefers direct communication, minimal comments (only "why" not "what"), no ASCII art in code, no em-dashes in prose. Uses parenthetical asides instead. Writing uses "so" and "but" as connectors, short punchy sentences mixed with technical depth.

## Repository Location

```
~/Dev/silicon-forge/        (canonical path)
~/silicon-forge              (symlink)
```

GitHub target: `TechnologyResidue/muppet-rainbow-connection`

## Related Projects (DO NOT MODIFY)

These are separate repos. MUPPET references them via env.sh but never copies or modifies their code:

```
~/apple-bottom        → ~/Dev/arm/metal-algos       # DD-BLAS library (Metal GPU)
~/acme-meep           → ~/Dev/acme-meephistopheles   # Metal FDTD kernels for MEEP
~/qe-test/q-e-qe-7.4.1/                             # Quantum ESPRESSO 7.4.1
~/yambo-build/yambo/                                 # Yambo 5.3.0
```

## Solver Environments (DO NOT CROSS-CONTAMINATE)

| Env | Activation | Solvers | Deps |
|-----|-----------|---------|------|
| meep_arm | `conda activate meep_arm` | MEEP 1.31.0 FP64 | conda-forge |
| meep-single | `conda activate meep-single` | MEEP 1.31.0 FP32 | conda-forge + source |
| qe-src | `conda activate qe-src` | (QE build only) | conda |
| (none/base) | `conda deactivate` | Yambo, QE runtime | Homebrew |

Never install packages into the wrong env. Never mix Homebrew and conda deps in the same build.

## Architecture

Read ARCHITECTURE.md for full technical design. Key points:

1. **Orchestration only.** This repo contains Python converters, shell workflow scripts, and documentation. Solver binaries live elsewhere.

2. **Converters are the glue.** Each converter translates output from one solver into input for the next:
   - `yambo_eps_to_meep.py` — Yambo BSE ε(ω) → MEEP Lorentzian material (WORKING)
   - `qe_structure_to_meep.py` — QE geometry → MEEP primitives (PLANNED)
   - `meep_fields_to_thermal.py` — MEEP absorbed power → Elmer heat source (PLANNED)

3. **Workflows are self-contained pipelines.** Each workflow directory has inputs/, scripts/, results/. Scripts are numbered: 01_run_qe.sh, 02_convert_p2y.sh, etc. A `run_all.sh` ties them together.

4. **env.sh knows where everything is.** Source it before doing anything:
   ```bash
   bash              # subshell to contain env vars
   source ~/silicon-forge/env.sh
   ```

## File Conventions

- Python scripts: standard library + numpy + scipy + meep. No exotic deps.
- Shell scripts: bash, not zsh. Use `set -euo pipefail`.
- Converter output: JSON for machine consumption, stdout for human consumption.
- Workflow results: JSON in `results/` dir, gitignored.
- No Jupyter notebooks. Scripts only.
- Flat directory structure within each module (no deep nesting).

## Code Style

- Python: PEP 8 with relaxed line length (100 chars). Type hints where they help, not everywhere.
- Shell: quoted variables, explicit error handling, comments for non-obvious flags.
- No ASCII art banners in code.
- Comments explain WHY, not WHAT. The code should be obvious; the physics might not be.
- Docstrings on every converter and public function. Include units.

## Key Technical Context

### What's bandwidth-limited vs compute-limited

This distinction drives every acceleration decision:

- **MEEP (FDTD):** Bandwidth-limited. Stencil reads ~48 bytes/cell, does ~6 FLOPs. One CPU core saturates DRAM at ~60 GB/s. GPU wins via 6.7x more bandwidth (361 GB/s). OpenMP is useless.

- **QE/Yambo (BLAS):** Compute-limited. ZGEMM does O(N³) FLOPs on O(N²) data. GPU wins via more ALUs (DD arithmetic). OpenMP helps here (more cores = more FLOPS).

- **FEM (sparse):** Mixed. SpMV is bandwidth-limited; iterative solvers add compute. Accelerate sparse BLAS is the right path.

### Precision requirements by solver

- QE: Needs ~14 digits (total energy). FP32 is garbage. DD (48-bit) works. FP64 ideal.
- Yambo GW/BSE: Same as QE — dense BLAS at high precision.
- MEEP: FP32 is fine. Discretization error dominates floating-point error.
- Thermal FEM: FP32 + iterative refinement is sufficient.

### The Lorentzian fitting caveat

The Yambo → MEEP converter fits ε(ω) to a sum of Lorentzian oscillators. Current known issues:
- ε∞ hits the lower bound (0.5). Should raise to 1.0 minimum.
- 4 poles captures E1 and E2 transitions but misses 5-6 eV features.
- For photovoltaic workflows (solar spectrum < 4 eV), 3 poles is sufficient.
- The fit quality depends on Yambo's BSE convergence (k-grid, bands, screening).

## Building and Testing

### Running the existing workflow

```bash
bash
source ~/silicon-forge/env.sh

# QE + Yambo steps require prior data in ~/yambo-build/si2-gw/
# MEEP step uses pre-fitted material from lib/converters/
cd workflows/si-photovoltaic/scripts
conda activate meep_arm
python 06_run_meep.py
```

### Running the converter standalone

```bash
conda activate meep_arm
cd lib/converters
python yambo_eps_to_meep.py ~/yambo-build/si2-gw/o-gw_mpi.eps_q1_diago_bse --poles 4 --plot
```

### Validating results

The Si photovoltaic workflow should produce:
- ~59% absorption at 1.6 eV for 500 nm film
- Strong reflectance above E2 peak (~3.3 eV)
- "Epsilon < 1" warning from MEEP (cosmetic, from ε∞ = 0.5)

## What NOT To Do

- Do not modify files in ~/apple-bottom, ~/acme-meep, ~/qe-test, ~/yambo-build
- Do not install conda packages into meep_arm or meep-single without asking
- Do not run `brew upgrade` (breaks Yambo's dylib linkage)
- Do not use `String(format: "%s")` with Swift strings (segfaults — use %@ or interpolation)
- Do not assume MEEP benefits from OpenMP on Apple Silicon (it doesn't)
- Do not put solver source code in this repo (orchestration only)
- Do not create Jupyter notebooks (scripts only)

## Current Priority

See TODO.md for the full prioritized list. The immediate goals are:

1. Get the repo committed to GitHub as muppet-rainbow-connection
2. Fix ε∞ lower bound in the Lorentzian fitter
3. Write run_all.sh for the si-photovoltaic workflow
4. Build the iterative thermal loop (the "infrared" in the rainbow)
