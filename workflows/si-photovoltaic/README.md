# Silicon Photovoltaic Device Workflow

## Pipeline
```
QE pw.x (SCF)         → Ground state, charge density
QE pw.x (NSCF)        → Expanded band structure (20+ bands)
p2y                    → Convert QE → Yambo database
Yambo (GW)             → Quasiparticle band gap correction
Yambo (BSE)            → Optical absorption spectrum ε(ω)
convert_eps_to_meep.py → Fit ε(ω) to Lorentzian model
MEEP (FDTD)            → Light propagation + absorption profile
(future) Elmer         → Thermal: heat from absorbed photons
```

## Data flow
```
inputs/si_scf.in       → QE SCF input
inputs/si_nscf.in      → QE NSCF input (nbnd=20, force_symmorphic)
inputs/yambo_gw.in     → Yambo GW input
inputs/yambo_bse.in    → Yambo BSE input
inputs/meep_device.py  → MEEP simulation with fitted material

scripts/01_run_qe.sh        → SCF + NSCF
scripts/02_convert_p2y.sh   → p2y conversion
scripts/03_run_yambo_gw.sh  → GW calculation
scripts/04_run_yambo_bse.sh → BSE optical spectrum
scripts/05_fit_epsilon.py   → Lorentzian fitting
scripts/06_run_meep.py      → FDTD with ab initio material
scripts/run_all.sh           → Full pipeline

results/                → All output data (gitignored)
```

## Hardware routing
- QE: apple-bottom DD-BLAS for ZGEMM (si64 or larger)
- Yambo: AMX via OpenBLAS (MPI+OpenMP)
- MEEP: Metal GPU stencil kernels (grid ≥ 64³)
