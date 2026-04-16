# rainbow-connection

**MUPPET: Multi-Physics Pipeline for Electronic Transport**

Orchestrates first-principles computational workflows across Quantum ESPRESSO (DFT), Yambo (GW/BSE), and MEEP (FDTD) to predict electronic and optical properties from the atomic scale through device-level electromagnetics.

## Pipeline

```
QE (pw.x)          Yambo (yambo)         MEEP (meep)
  SCF/NSCF    -->    GW/BSE        -->    FDTD
  band structure      dielectric fn       S-parameters
  charge density      optical spectra     field profiles
```

Each solver stage produces outputs consumed by the next. Converter scripts in `lib/converters/` bridge file formats between codes (e.g., Yambo epsilon to MEEP material definitions).

## Workflows

| Workflow | Description | Status |
|----------|-------------|--------|
| `gaas-emi-shield` | GaAs membrane EMI shielding (reproduces experimental data) | Active |
| `si-photovoltaic` | Si photovoltaic optical properties | Active |
| `gaas-waveguide-sim` | GaAs waveguide simulation | Active |
| `piezo-resonators` | Piezoelectric resonator S-parameters | Active |
| `photonic-crystal` | Photonic crystal band structure | Active |
| `si-qgt` | Si quantum geometric tensor | Active |
| `led-extraction` | LED light extraction efficiency | Active |
| `epw_gan_test` | GaN electron-phonon (EPW) test case | Active |

Each workflow directory contains QE/Yambo/MEEP input files and run scripts. Results directories are gitignored (generated data is large).

## Solver Configurations

Solver-specific build notes and runtime configs are in `solvers/`:

- `solvers/qe/` — QE 7.5 with apple-bottom acceleration
- `solvers/yambo/` — Yambo 5.3.0 GW/BSE configuration
- `solvers/meep/` — MEEP FDTD configuration
- `solvers/elmer/` — Elmer FEM for coupled physics

## Libraries

- `lib/converters/` — Format bridges between solver codes (Yambo epsilon -> MEEP materials)
- `lib/materials/` — Material property databases
- `lib/qgt/` — Quantum geometric tensor utilities

## Requirements

- [Quantum ESPRESSO 7.5](https://www.quantum-espresso.org/) (see [Quantum-Espressivo](https://github.com/grantdh/Quantum-Espressivo) for Metal GPU build)
- [Yambo 5.3.0](http://www.yambo-code.eu/) (see [YAMBOrghini](https://github.com/grantdh/YAMBOrghini) for Metal GPU build)
- [MEEP](https://meep.readthedocs.io/) (see [MEEPhistopheles](https://github.com/grantdh/MEEPhistopheles) for Metal FDTD kernels)
- Python 3.9+ with NumPy, Matplotlib

## Ecosystem

- [apple-bottom](https://github.com/grantdh/apple-bottom) — FP64-class BLAS on Apple Silicon GPU (the engine)
- [Quantum-Espressivo](https://github.com/grantdh/Quantum-Espressivo) — QE + Metal GPU acceleration
- [YAMBOrghini](https://github.com/grantdh/YAMBOrghini) — Yambo GW/BSE + Metal GPU acceleration
- [MEEPhistopheles](https://github.com/grantdh/MEEPhistopheles) — MEEP FDTD + Metal GPU acceleration

## License

MIT — see [`LICENSE`](LICENSE).
