# MUPPET Ecosystem: Repository Map and Path Forward

**Author:** Grant Heileman, UNM ECE  
**Date:** March 2026  
**Hardware:** Apple M2 Max (30-core GPU, 96GB unified memory)

---

## The Big Picture

Seven repos. Three shared infrastructure libraries. Three solver integration projects. One orchestration umbrella. Each name encodes its purpose on multiple levels.

```
MUPPET: Rainbow Connection          ← orchestration umbrella
├── apple-bottom (BLASphemy)        ← Metal FP64 BLAS        [WORKING]
├── iron-butterfly                   ← Metal FFT              [PLANNED]
├── maxwells-silver-hammer           ← Metal structured stencils [PROTOTYPE]
├── acme-meephistopheles             ← MEEP integration        [PROTOTYPE]
├── yamborghini                      ← Yambo integration       [IN PROGRESS]
└── corretto                        ← QE integration          [WORKING]
```

---

## 1. Shared Infrastructure (Libraries)

These have multiple consumers. They don't know or care who calls them.

### 1.1 apple-bottom (BLASphemy)

**What:** Metal-native DD arithmetic BLAS. Emulates FP64 via FP32x2 pairs on Apple Silicon GPUs.

**Repo:** `~/apple-bottom` → `~/Dev/arm/metal-algos`  
**GitHub:** `grantdh/apple-bottom`  
**Status:** WORKING. Production kernels validated.

**Key results:**
- DD-DGEMM: +14-23% vs AMX at N >= 1024
- Gauss DD-ZGEMM: +23-43% vs AMX at N >= 2048
- Comp-13 kernel: 760 GFLOP/s, 34.2 bits precision
- QE integration validated: correct energy on si64

**Consumers:** corretto (QE), yamborghini (Yambo), any future BLAS-heavy code

**Key files:**
- `src/fortran_bridge.c` — Fortran ABI bridge (ab_dgemm_, ab_zgemm_)
- `ex10_production_dd_dgemm.swift` — Production 4x4 kernel
- `ex11b_split_zgemm.swift` — Gauss 3-multiply ZGEMM

**Paper:** "ULP Fiction: Faithfully-Rounded FP64 Emulation on FP32 Hardware" (Numerical Algorithms, in prep)

### 1.2 iron-butterfly

**What:** Metal-native FFT library. Butterfly = FFT butterfly operation. Iron = Metal API.

**Repo:** Not yet created  
**Status:** PLANNED

**Why it exists:** QE and Yambo are both FFT-heavy (plane-wave basis). Accelerate vDSP_fft exists but may not be optimal for the specific transform sizes and batching patterns these codes need. This is the least urgent of the three infrastructure libraries because Accelerate's FFT is already decent.

**Consumers:** corretto (QE), yamborghini (Yambo)

**Priority:** LOW. Build only after apple-bottom and maxwells-silver-hammer are production-ready. Accelerate FFT is adequate for now. Profile QE and Yambo FFT calls first to determine if this is even worth building.

### 1.3 maxwells-silver-hammer

**What:** Metal structured-grid stencil library. Maxwell = Maxwell's equations (the physics). Silver = Metal (the API). Hammer = computational force.

**Repo:** Not yet created (kernels currently live in acme-meephistopheles)  
**Status:** PROTOTYPE (kernels validated, need to factor out of MEEPhistopheles)

**Key results (from acme-meephistopheles prototype):**
- step_curl: 361 GB/s at 256^3 (90% of M2 Max theoretical)
- Full timestep: 2601 Mcells/s (7.8x vs FP32 CPU MEEP)
- Crossover at ~64^3 grid
- OpenMP provides zero benefit (single core saturates DRAM bandwidth)

**Consumers:** acme-meephistopheles (MEEP), potentially openEMS, acoustic FDTD, heat solvers

**Priority:** MEDIUM. Factor out of MEEPhistopheles when a second consumer appears. For now, the kernels live in MEEPhistopheles and that's fine.

---

## 2. Solver Integration Projects

These wire specific solvers to the shared infrastructure. They contain build scripts, patches, environment setup, and solver-specific glue.

### 2.1 corretto

**What:** QE integration. Espresso "corrected" with GPU acceleration. Contains the patched cegterg.f90, build scripts, benchmark harness, and the Espressivo auto-configurator.

**Repo:** Currently embedded in `~/qe-test/`. Needs extraction.  
**Status:** WORKING

**Key components:**
- Patched `cegterg.f90` with `EXTERNAL :: ab_zgemm` and 12 ZGEMM call replacements
- `fortran_bridge.c` compiled and archived into QE's libqemod.a
- `espressivo` auto-configurator (bash, routes CPU vs GPU by matrix size)
- Benchmark suite: si8, si16, si32, si64, si128
- Recovery checklist for broken QE builds

**Dependencies:** apple-bottom (for DD-BLAS), eventually iron-butterfly (for Metal FFT)

**Key finding:** GPU wins on si64+, loses to 8-thread OpenBLAS on si32. Crossover at ~100M FLOPs in fortran_bridge.c.

### 2.2 yamborghini

**What:** Yambo integration. Build system patches, environment setup, wiring Yambo's BLAS through apple-bottom.

**Repo:** Not yet created as standalone. Build/integration work currently scattered across `~/yambo-build/`.  
**Status:** IN PROGRESS

**Current state:**
- Yambo 5.3.0 builds and runs (serial + MPI + OpenMP + SLK + HDF5)
- GW calculation validated on 2-atom Si
- BSE calculation working (bands 3-6, produces epsilon output)
- Double-precision rebuild completed (`--enable-dp`)
- p2y requires `force_symmorphic = .true.` in QE inputs

**Critical finding from transcript:** Yambo's single-precision build (default) uses CGEMM, not ZGEMM. The `#if defined _DOUBLE` guards mean all ZGEMM call sites are dead code in SP mode. Must rebuild with `--enable-dp` for apple-bottom integration.

**BLAS integration path:** All ZGEMM calls funnel through `mod_wrapper.F` (3 call sites) and `mod_wrapper_omp.F` (6 call sites). Zero direct ZGEMM calls elsewhere. Clean interposition point.

**Dependencies:** apple-bottom (for DD-BLAS), eventually iron-butterfly (for Metal FFT)

**Key environment variables:**
```bash
export OMPI_CC=/opt/homebrew/bin/gcc-15
export OMPI_FC=/opt/homebrew/bin/gfortran-15
```

### 2.3 acme-meephistopheles

**What:** MEEP acceleration and integration. Contains Metal FDTD stencil kernels, C bridge for MEEP integration, benchmarks.

**Repo:** `~/acme-meep` → `~/Dev/acme-meephistopheles`  
**Status:** PROTOTYPE (kernels validated, C bridge not yet built)

**Key components:**
- `src/metal/meep_metal_fdtd.swift` — Three validated kernels: step_curl, update_edhb, step_curl_pml
- Benchmark suite with correctness validation
- LESSONS_LEARNED_MEEP.md (comprehensive, all numbers measured)

**The Faustian bargain:** Surrender FP64 precision (FDTD doesn't need it), gain 7.8x speedup. Mephistopheles offers forbidden knowledge in exchange for something you thought you needed but didn't.

**Dependencies:** maxwells-silver-hammer (when factored out). Currently self-contained.

**Integration plan:**
1. Build C bridge: `metal_fdtd_bridge.{h,c,m}`
2. Patch MEEP step.cpp: `if (use_metal) metal_step_curl(...)`
3. Add `--enable-metal` configure flag
4. Python API: `meep.use_backend('metal')`

---

## 3. Orchestration Umbrella

### 3.1 MUPPET: Rainbow Connection

**What:** Multi-Physics Unified Pipeline for Photonic Engineering and Technology. Orchestrates QE, Yambo, MEEP, and thermal solvers. Contains converters, workflow scripts, the QGT module, and documentation.

**Repo:** `~/silicon-forge` → `~/Dev/silicon-forge`
**GitHub target:** `grantdh/rainbow-connection`
**Status:** FUNCTIONAL (first end-to-end QE → Yambo → MEEP completed)

**Key components:**
- `lib/converters/yambo_eps_to_meep.py` — Lorentzian fitting (WORKING)
- `lib/qgt/qgt_from_dft.py` — Quantum metric from DFT wavefunctions (WORKING)
- `lib/qgt/qgt_prototype.py` — Toy model validation (WORKING)
- `workflows/si-photovoltaic/` — First end-to-end workflow
- `workflows/gaas-emi-shield/` — Your paper reproduction (QE SCF+NSCF done, QGT computed)
- `env.sh` — Knows where all solvers live
- `CLAUDE.md`, `ARCHITECTURE.md`, `TODO.md`, `INSTALL.md`

**Does NOT contain:** Solver code, acceleration kernels, build systems. Pure orchestration + converters + science workflows.

---

## 4. Papers

| Working Title | Target Journal | Status | Repos Involved |
|---|---|---|---|
| ULP Fiction | Numerical Algorithms | Revision in prep | apple-bottom |
| More Than Skin Deep | TBD (Phys. Rev. B?) | Early stage | MUPPET (QGT module) + corretto |
| On the Same Wavelength | (alternate title) | | |
| Fields All The Way Down | TBD | Separate | |

---

## 5. Build Order (What to Do Next)

### Phase 1: Commit What Works (This Week)

1. **MUPPET to GitHub** — git init, .gitignore, first commit with all docs, converters, QGT module, workflow scripts. This is the P0 from TODO.md.

2. **apple-bottom cleanup** — Already on GitHub. Update README with latest benchmark numbers from ULP Fiction paper.

3. **acme-meephistopheles to GitHub** — git init, commit Metal FDTD prototype + LESSONS_LEARNED_MEEP.md + benchmarks.

### Phase 2: GaAs EMI Shielding Paper (Next 2 Weeks)

This is the highest-impact deliverable — reproducing your own experimental data from first principles.

4. **Complete GaAs QGT → ε(ω) pipeline:**
   - Fix QGT normalization (Ω_I cross-check against Wannier90, need num_wann=9)
   - Add Drude weight from QGT at Fermi surface (Queiroz framework)
   - Kramers-Kronig → full ε(ω) from 0 to UV
   - Compare to experimental ε from ellipsometry

5. **MEEP simulations of your membranes:**
   - IR transmittance 750-2500 nm for 200/500/1000 nm films → compare to Fig. 2
   - mm-wave SE 65-85 GHz → compare to Fig. 4
   - GaAs/Kapton/GaAs multilayer → compare to Fig. 6

6. **Write the paper:** "More Than Skin Deep" or "On the Same Wavelength"

### Phase 3: Yambo DP + apple-bottom Integration (Month 2)

7. **yamborghini as standalone repo:**
   - Extract Yambo build scripts, env.sh, lessons learned from ~/yambo-build
   - Document the DP rebuild procedure
   - Patch mod_wrapper.F: ZGEMM → ab_zgemm (3 call sites)
   - Link apple-bottom's libapple_bottom.a into Yambo's build
   - Benchmark BSE kernel builder with GPU ZGEMM on larger systems (16+ atoms)

8. **corretto as standalone repo:**
   - Extract QE integration from ~/qe-test
   - espressivo auto-configurator
   - Benchmark suite
   - Recovery checklist

### Phase 4: Stencil Factoring + MEEP Integration (Month 3)

9. **maxwells-silver-hammer:** Factor stencil kernels out of MEEPhistopheles into standalone library. Only do this when a second consumer appears.

10. **MEEP C bridge:** Build the actual integration that lets MEEP call Metal kernels. `--enable-metal` configure flag.

### Phase 5: Thermal Loop + iron-butterfly (Month 4+)

11. **Elmer FEM integration** for thermal solver
12. **Iterative coupling:** T → Eg → ε(ω) → MEEP → Q_absorbed → T
13. **iron-butterfly:** Only if profiling shows Accelerate FFT is a bottleneck

---

## 6. Environment Isolation Rules

| Solver | Dep Source | Conda Env | Notes |
|--------|-----------|-----------|-------|
| QE 7.4.1 | Homebrew | qe-src (build only) | GCC-15, OpenBLAS, FFTW, Open MPI |
| Yambo 5.3.0 | Homebrew | none (subshell) | Same compiler stack as QE. OMPI_CC/FC critical. |
| MEEP FP64 | Conda | meep_arm | Brings own FFTW, HDF5, MPI. DO NOT MIX with Homebrew. |
| MEEP FP32 | Conda + source | meep-single | Source build with --enable-single against conda deps. |
| Wannier90 | /usr/local/bin | any | Pre-installed, works everywhere. |

**Golden rule:** Never install conda packages into the wrong env. Never mix Homebrew and conda deps. Never run `brew upgrade` without pinning.

---

## 7. Current File System Map

```
~/apple-bottom/           → ~/Dev/arm/metal-algos        [apple-bottom repo]
~/acme-meep/              → ~/Dev/acme-meephistopheles    [MEEPhistopheles repo]
~/silicon-forge/           → ~/Dev/silicon-forge           [MUPPET repo]
~/qe-test/q-e-qe-7.4.1/                                  [QE install + corretto work]
~/yambo-build/yambo/                                      [Yambo install + yamborghini work]
~/yambo-build/si2-gw/                                     [Yambo test data (Si GW+BSE)]
~/meep-single-bench/                                      [FP32 MEEP source build]

Conda envs:
  meep_arm        MEEP 1.31.0 FP64
  meep-single     MEEP 1.31.0 FP32 (source build)
  qe-src          QE build dependencies
```

---

## 8. Naming Reference

| Repo | Full Name | Technical Embedding |
|------|-----------|-------------------|
| apple-bottom (BLASphemy) | — | BLAS in BLASphemy; "bottom" = low-level |
| iron-butterfly | Iron Butterfly (band) | Iron = Metal API; Butterfly = FFT operation; In-A-Gadda-Da-Vida = Garden of Eden = Apple |
| maxwells-silver-hammer | Maxwell's Silver Hammer (Beatles) | Maxwell = Maxwell's equations; Silver = Metal; Hammer = computational force |
| acme-meephistopheles | ACME + Mephistopheles | ACME = acceleration brand; Mephistopheles = Faustian bargain (surrender FP64 for speed) |
| yamborghini | Yambo + Lamborghini | Italian car + Italian code; speed/performance |
| corretto | Caffè corretto | Espresso "corrected" with liquor = QE corrected with GPU; also SCF iterative correction |
| MUPPET | Multi-Physics Unified Pipeline for Photonic Engineering and Technology | Beaker says "meep"; Kermit = Green's functions; Rainbow = EM spectrum |
