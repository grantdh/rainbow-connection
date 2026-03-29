# Architecture

## Design Principles

1. **Orchestration, not implementation.** MUPPET contains no solver code. Solvers live in their own repos and are referenced by environment variables. This repo is glue.

2. **Converters are the product.** The value of this project is in the data translators that connect solver outputs to solver inputs. Each converter is a standalone Python script with no state.

3. **Every number is measured.** No claimed speedups without benchmark data. No "roughly 2x" estimates when we can measure. The docs directory contains lessons-learned files with every benchmark result and their measurement methodology.

4. **Fail-safe environment isolation.** Each solver has its own dependency stack. Conda envs don't share packages. Homebrew packages are pinned. Building one solver never breaks another.

## Data Flow

```
                    ┌──────────────┐
                    │  QE pw.x     │
                    │  SCF + NSCF  │
                    └──────┬───────┘
                           │ wavefunctions (tmp/prefix.save/)
                           ▼
                    ┌──────────────┐
                    │  p2y         │
                    │  QE → Yambo  │
                    └──────┬───────┘
                           │ SAVE/ database
                           ▼
               ┌───────────────────────┐
               │  Yambo                │
               │  GW: band gap fix     │
               │  BSE: ε(ω) spectrum   │
               └───────────┬───────────┘
                           │ o-*.eps_q1_diago_bse (text: energy, Re ε, Im ε)
                           ▼
               ┌───────────────────────┐
               │  yambo_eps_to_meep.py │
               │  Lorentzian fitting   │
               └───────────┬───────────┘
                           │ yambo_material.json (poles: ω, γ, σ)
                           ▼
               ┌───────────────────────┐
               │  MEEP FDTD            │
               │  Light propagation    │
               └───────────┬───────────┘
                           │ T, R, A spectra (JSON)
                           ▼
               ┌───────────────────────┐
               │  meep_fields_to_      │
               │  thermal.py (future)  │
               └───────────┬───────────┘
                           │ volumetric heat source
                           ▼
               ┌───────────────────────┐
               │  Elmer FEM (future)   │
               │  Thermal transport    │
               └───────────┬───────────┘
                           │ T(r) temperature field
                           ▼
                    ┌──────────────┐
                    │  Varshni     │
                    │  T → Eg(T)   │──── back to QE if converging
                    └──────────────┘
```

## Converter Interface Contract

Every converter follows this pattern:

```python
# Input: file path (solver output format)
# Output: JSON file (next solver's input format) + stdout summary
# CLI: python converter.py input_file [--options] [--output output.json]
# Dependencies: numpy, scipy, standard library only
# No global state, no side effects beyond file I/O
```

### yambo_eps_to_meep.py (implemented)

- **Input:** Yambo BSE output (`o-*.eps_q1_diago_bse`). Columns: energy(eV), Im[ε]_BSE, Re[ε]_BSE, Im[ε]_IP, Re[ε]_IP.
- **Output:** `yambo_material.json` with fitted Lorentzian parameters.
- **Algorithm:** scipy.optimize.least_squares fitting ε(ω) = ε∞ + Σ σₙωₙ²/(ωₙ² - ω² - iγₙω).
- **Unit conversion:** eV → MEEP frequency units via f_meep = E_eV / 1.23984 (hc in eV·μm).

### qe_structure_to_meep.py (planned)

- **Input:** QE output file (cell parameters + atomic positions in Bohr/Angstrom).
- **Output:** Python dict of MEEP geometric primitives.
- **Challenge:** QE uses crystal coordinates; MEEP uses Cartesian μm. Need unit cell → device mapping.

### meep_fields_to_thermal.py (planned)

- **Input:** MEEP HDF5 field output (E, H fields on grid).
- **Output:** Volumetric heat source Q(r) = ½ωε″|E|² for Elmer.
- **Challenge:** MEEP and Elmer use different meshes. Need interpolation or shared grid.

## Workflow Structure

Each workflow is a numbered sequence of scripts that can be run individually or via run_all.sh:

```
workflows/si-photovoltaic/
├── README.md           # What this workflow computes
├── inputs/             # QE input files, Yambo input files
├── scripts/
│   ├── 01_run_qe.sh         # pw.x SCF + NSCF
│   ├── 02_convert_p2y.sh    # QE → Yambo database
│   ├── 03_run_yambo_gw.sh   # GW quasiparticle corrections
│   ├── 04_run_yambo_bse.sh  # BSE optical spectrum
│   ├── 05_fit_epsilon.py    # Lorentzian fitting
│   ├── 06_run_meep.py       # FDTD with ab initio material
│   └── run_all.sh           # Full pipeline
└── results/            # Output data (gitignored)
```

Each script:
- Sources env.sh at the top
- Checks for prerequisite files before running
- Writes output to results/ or to the next solver's expected location
- Prints a summary of what it produced

## Hardware Routing

The routing decision is made per-solver, not per-call. The env.sh sets up paths so each solver finds its acceleration library:

| Solver | Decision | Threshold | Fast Path | Slow Path |
|--------|----------|-----------|-----------|-----------|
| QE | Matrix size | N ≥ 1024 | Metal DD-BLAS | AMX Accelerate |
| Yambo | (same BLAS) | N ≥ 1024 | Metal DD-BLAS | AMX Accelerate |
| MEEP | Grid size | ≥ 64³ | Metal compute shaders | CPU (single-core) |
| Elmer | (future) | — | Accelerate sparse | — |

QE's routing is handled by apple-bottom's Fortran bridge (libapple_bottom.a linked into QE, routes ZGEMM calls by size). MEEP's routing would be handled by a C bridge in acme-meephistopheles (future). The routing logic is in the acceleration libraries, not in this repo.

## Iterative Coupling (Future)

The outer loop for self-consistent multiphysics:

```python
T = 300  # initial temperature (K)
for iteration in range(max_iter):
    Eg = varshni(T)                    # T → band gap
    eps_omega = run_yambo_bse(Eg)      # band gap → optical spectrum
    material = fit_lorentzian(eps_omega)
    Q_absorbed = run_meep(material)    # spectrum → absorbed power
    T_new = run_elmer(Q_absorbed)      # absorbed power → temperature
    if abs(T_new - T) < tol:
        break
    T = mix(T, T_new, alpha=0.3)       # linear mixing for stability
```

Key design decisions for the iterative loop:
- Linear mixing (not direct substitution) to prevent oscillation
- Convergence on temperature (not on ε), since T is the slowest-changing quantity
- Each iteration runs the full pipeline but reuses wavefunctions from previous QE run
- Adaptive precision: early iterations use fewer k-points and bands, final iterations converge fully

## Testing Strategy

- **Converter tests:** Known input → expected output. Si BSE from Yambo should produce Lorentzian parameters within 10% of literature E1/E2 peak positions.
- **Workflow tests:** End-to-end on 2-atom Si cell (fast, ~minutes). Validate absorption onset matches GW band gap.
- **Benchmark tests:** Timing comparisons logged to benchmarks/ directory. Must include hardware spec, thread count, grid/matrix sizes.
- **No unit test framework.** Tests are standalone scripts that print PASS/FAIL. Keep it simple.
