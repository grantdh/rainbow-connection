# MUPPET: Rainbow Connection 🌈

**Multiphysics Unified Pipeline for Photonic Engineering and Technology**

*"Why are there so many songs about rainbows, and what's on the other side?"*  
*— Because each color is a different solver, and the other side is a converged answer.*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)]()

## What This Is

An open-source multiphysics simulation pipeline that connects quantum mechanics to electromagnetic device design, running entirely on a MacBook. No cloud GPU, no HPC cluster, no $10K/year software licenses.

The pipeline computes optical properties of materials from first principles (no empirical parameters) and propagates light through photonic device structures using those computed properties:

```
QE pw.x (DFT)  →  Yambo (GW+BSE)  →  MEEP (FDTD)  →  [Elmer FEM]
ground state       optical spectrum     light propagation  thermal effects
  ~minutes           ~minutes             ~seconds          (future)
```

Each solver is GPU-accelerated on Apple Silicon where it matters:

| Solver | Bottleneck | Acceleration | Measured Speedup |
|--------|-----------|-------------|-----------------|
| QE pw.x | Dense BLAS (ZGEMM) | apple-bottom DD-BLAS on Metal | +14-43% vs AMX |
| Yambo GW/BSE | Dense BLAS (ZGEMM) | apple-bottom DD-BLAS on Metal | Same profile as QE |
| MEEP FDTD | Memory bandwidth (stencil) | Metal compute shaders | 7.8x vs FP32 CPU |
| Elmer FEM | Sparse solvers | Accelerate sparse + IR | (planned) |

## Quick Start

```bash
# Clone
git clone https://github.com/grantdh/rainbow-connection.git
cd rainbow-connection

# Load environment
bash
source env.sh

# Run the Si photovoltaic workflow (requires QE, Yambo, MEEP installed)
cd workflows/si-photovoltaic/scripts
bash run_all.sh
```

See [INSTALL.md](INSTALL.md) for setting up each solver.

## The Rainbow Connection

Each "color" in the rainbow is a different physics domain connected through data converters:

| Color | Solver | Physics | Data Produced |
|-------|--------|---------|--------------|
| Violet | QE pw.x | DFT ground state | Wavefunctions, charge density |
| Blue | Yambo GW | Quasiparticle corrections | Corrected band gap |
| Green | Yambo BSE | Excitonic optics | ε(ω) dielectric function |
| Yellow | yambo_eps_to_meep.py | Lorentzian fitting | MEEP-compatible material |
| Orange | MEEP FDTD | Light propagation | T, R, A spectra |
| Red | Elmer FEM (future) | Thermal transport | Temperature field |
| Infrared | (future) | Back-coupling | T → band gap → ε(ω) → MEEP |

The "infrared" step is where it gets interesting: temperature affects the band gap (Varshni relation), which changes ε(ω), which changes absorption, which changes heat generation. That outer loop is the iterative multiphysics solver.

## Project Structure

```
rainbow-connection/
├── README.md                    # This file
├── ARCHITECTURE.md              # Technical design (for developers and Claude Code)
├── CLAUDE.md                    # Instructions for Claude Code
├── INSTALL.md                   # Solver installation guide
├── TODO.md                      # Prioritized task list
├── env.sh                       # Environment setup
│
├── lib/
│   ├── converters/              # Data format translators between solvers
│   │   ├── yambo_eps_to_meep.py # Yambo BSE ε(ω) → MEEP Lorentzian material
│   │   ├── qe_structure_to_meep.py  # (planned)
│   │   └── meep_fields_to_thermal.py # (planned)
│   ├── materials/               # Fitted material parameter files
│   └── routing/                 # Hardware dispatch logic
│
├── solvers/                     # Solver configuration (references external installs)
│   ├── qe/README.md
│   ├── yambo/README.md
│   ├── meep/README.md
│   └── elmer/README.md
│
├── workflows/                   # Self-contained simulation pipelines
│   ├── si-photovoltaic/         # QE → Yambo → MEEP for solar cell
│   ├── photonic-crystal/        # MEEP + QE for PhC design
│   └── led-extraction/          # Emission + extraction + thermal
│
├── benchmarks/                  # Performance measurements
│   ├── qe/                      # DD-BLAS timing
│   ├── meep/                    # Metal FDTD vs CPU
│   ├── yambo/                   # GW/BSE timing
│   └── coupled/                 # End-to-end workflow timing
│
├── shared/
│   ├── pseudo/                  # Pseudopotentials
│   └── structures/              # Crystal structure files (CIF, XSF)
│
└── docs/
    ├── LESSONS_LEARNED_MEEP.md
    ├── LESSONS_LEARNED_GENERALIZATION.md
    └── figures/
```

## Dependencies

| Solver | Install Method | Location |
|--------|---------------|----------|
| QE 7.4.1 | Source (with apple-bottom) | ~/qe-test/q-e-qe-7.4.1 |
| Yambo 5.3.0 | Source (Homebrew deps) | ~/yambo-build/yambo |
| MEEP 1.31.0 | Conda (meep_arm env) | conda: meep_arm |
| Metal FDTD | Swift source | ~/acme-meep/src/metal |
| apple-bottom | Swift source | ~/apple-bottom |

## Measured Performance (M2 Max)

### MEEP FDTD (256³ grid)

| Configuration | Mcells/s | Speedup |
|--------------|----------|---------|
| MEEP FP64 CPU (1 thread) | 211 | 1× |
| MEEP FP32 CPU (1 thread) | 335 | 1.59× |
| Metal GPU FP32 | 2601 | **7.8× vs FP32** |

Decomposition: 12.3× total = 1.59× (FP32 precision) × 7.8× (GPU bandwidth)

### QE DD-BLAS (N ≥ 1024)

| Operation | Speedup vs AMX |
|-----------|---------------|
| DD-DGEMM | +14-23% |
| Gauss DD-ZGEMM | +23-43% |

### Key Architectural Insight

OpenMP threading provides zero speedup for MEEP on Apple Silicon. A single CPU core already saturates DRAM bandwidth (~60 GB/s) for the FDTD stencil. The GPU is the only path to more bandwidth (361 GB/s, 90% of theoretical peak).

## Publications

- G. Heileman, "DD-BLAS: Double-float BLAS for Apple Silicon GPUs" (CPC paper, in preparation)
- G. Heileman, "Faithfully-Rounded FP64 Emulation on FP32 Hardware" (Numerical Algorithms, in preparation)

## Citation

```bibtex
@software{heileman2026muppet,
  author = {Heileman, Grant David},
  title = {MUPPET: Rainbow Connection — Multiphysics Unified Pipeline for Photonic Engineering and Technology},
  year = {2026},
  publisher = {Grant D. Heileman Ph.D.},
  url = {https://github.com/grantdh/rainbow-connection}
}
```

## License

Apache 2.0

## Acknowledgments

- apple-bottom DD-BLAS kernels informed by Philip Turner's metal-benchmarks
- QE/Yambo integration lessons from the Yambo wiki macOS ARM guide
- MEEP source code analysis of step_generic.cpp guided kernel architecture
