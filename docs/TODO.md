# TODO

Prioritized task list for MUPPET: Rainbow Connection.

## P0: Ship It

These must be done before the first GitHub push.

- [ ] **Git init and first commit**
  - `git init`, `.gitignore` (results/, *.pyc, __pycache__, .DS_Store)
  - Commit all docs, env.sh, converters, workflow scripts, solver READMEs
  - Push to `grantdh/rainbow-connection`

- [ ] **LICENSE file** — Apache 2.0

- [ ] **Copy docs to project**
  - LESSONS_LEARNED_MEEP.md → docs/
  - LESSONS_LEARNED_GENERALIZATION.md → docs/

- [ ] **Write .gitignore**
  - `workflows/*/results/`
  - `lib/materials/*.json` (generated, not committed)
  - `*.pyc`, `__pycache__/`, `.DS_Store`
  - `shared/pseudo/` (pseudopotentials are large)

## P1: Make the Pipeline Robust

- [ ] **Fix ε∞ lower bound in yambo_eps_to_meep.py**
  - Current: lb = 0.5 → fit pushes ε∞ to 0.5 → "Epsilon < 1" warning in MEEP
  - Fix: raise lower bound to 1.0 (vacuum is the physical floor)
  - Also add option to fix ε∞ to Yambo's reported static value

- [ ] **Write run_all.sh for si-photovoltaic**
  - Sources env.sh
  - Checks each prerequisite before proceeding
  - Runs steps 01-06 in sequence
  - Reports timing for each step and total
  - Usage: `bash run_all.sh [--skip-qe] [--skip-yambo]`

- [ ] **Write 01_run_qe.sh through 04_run_yambo_bse.sh**
  - Input files for 2-atom Si (already in ~/qe-test and ~/yambo-build)
  - Copy QE input files into workflows/si-photovoltaic/inputs/
  - Each script: source env.sh, check prereqs, run solver, validate output

- [ ] **Write 05_fit_epsilon.py**
  - Thin wrapper calling yambo_eps_to_meep.py with workflow-specific paths
  - Copies output to lib/materials/si_bse.json

## P2: Convergence and Validation

- [ ] **Converge the BSE calculation**
  - Current: 4×4×4 k-grid, 2-atom cell, bands 3-6
  - Target: 8×8×8 k-grid, bands 1-20, larger screening cutoff
  - Compare ε(ω) to experimental Si data (Aspnes & Studna 1983)
  - Document convergence in workflows/si-photovoltaic/README.md

- [ ] **Validate MEEP absorption against known results**
  - Compare thin-film T/R/A to transfer-matrix method (analytical for planar)
  - This validates the Lorentzian fitting quality, not MEEP itself

- [ ] **Add error bars / sensitivity analysis**
  - How much does T/R/A change with 3 vs 4 vs 5 Lorentzian poles?
  - How much does it change with different BSE convergence parameters?
  - Document in results/

## P3: Thermal Loop (The "Infrared")

- [ ] **Install Elmer FEM**
  - Homebrew: `brew install elmer` or build from source
  - Validate with simple heat equation test case
  - Document in solvers/elmer/README.md and INSTALL.md

- [ ] **Write meep_fields_to_thermal.py**
  - Read MEEP output (absorbed power density)
  - Convert to volumetric heat source for Elmer
  - Handle mesh interpolation (MEEP structured grid → Elmer FEM mesh)

- [ ] **Write thermal_to_bandgap.py**
  - Implement Varshni relation: Eg(T) = Eg(0) - αT²/(T+β)
  - For Si: Eg(0)=1.166 eV, α=4.73e-4, β=636
  - Output: shift to apply to Yambo QP corrections

- [ ] **Write iterative driver**
  - Python script implementing the outer convergence loop
  - Linear mixing for stability
  - Convergence criterion on temperature
  - Log each iteration's results

## P4: Additional Workflows

- [ ] **Photonic crystal workflow**
  - MEEP band structure of PhC slab
  - QE provides material ε(ω) for the constituent
  - Optimize geometry for target bandgap / Q-factor

- [ ] **LED extraction workflow**
  - Yambo BSE for emission spectrum
  - MEEP for light extraction through surface texturing
  - Thermal solver for junction self-heating

## P5: Additional Converters

- [ ] **qe_structure_to_meep.py**
  - Read QE pw.x output for cell vectors and atom positions
  - Convert Bohr → μm
  - Generate MEEP Block/Cylinder/Sphere primitives
  - Handle supercell → device geometry mapping

- [ ] **yambo_emission_to_meep.py**
  - For LED workflow: BSE emission → MEEP source spectrum
  - Different from absorption (need oscillator strengths, not ε)

## P6: Apple-Bottom Integration into Yambo

- [ ] **Profile Yambo BSE BLAS calls**
  - Log ZGEMM sizes during BSE kernel construction
  - Determine if calls cross the N≥1024 threshold

- [ ] **Test apple-bottom interposition with Yambo**
  - Same DYLD_INSERT_LIBRARIES approach as QE
  - Or direct library link (Yambo uses OpenBLAS, same ABI)
  - Measure speedup on BSE for larger systems (16+ atoms)

## P7: Metal FDTD Integration into MEEP

- [ ] **Build C bridge library in acme-meephistopheles**
  - `src/bridge/metal_fdtd_bridge.{h,c,m}`
  - C API: `metal_step_curl(float *f, float *g1, float *g2, ...)`
  - Links against Metal framework via ObjC

- [ ] **Patch MEEP step.cpp**
  - `if (use_metal) metal_step_curl(...) else step_curl(...)`
  - Add `--enable-metal` configure flag
  - Chunk-level dispatch: each MEEP chunk → one Metal dispatch

- [ ] **Python API**
  - `sim = mp.Simulation(..., use_metal=True)`
  - Or `meep.use_backend('metal')`

## Done

- [x] MEEP conda install (meep_arm env)
- [x] MEEP single-precision source build (meep-single env)
- [x] Metal FDTD prototype (meep_metal_fdtd.swift) — validated, 7.8× speedup
- [x] Full benchmark decomposition (FP32 vs FP64, vDSP, GPU)
- [x] OpenMP thread scaling test (confirmed useless)
- [x] Accelerate vDSP test (1.25× bare stencil, ~5% in MEEP)
- [x] Yambo GW calculation on Si
- [x] Yambo BSE calculation on Si (bands 3-6)
- [x] yambo_eps_to_meep.py converter (Lorentzian fitting)
- [x] First end-to-end QE → Yambo → MEEP simulation
- [x] LESSONS_LEARNED_MEEP.md (with all measured data)
- [x] Project directory structure (silicon-forge)
- [x] env.sh with all solver paths
- [x] README.md, CLAUDE.md, ARCHITECTURE.md
