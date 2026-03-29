# Lessons Learned: March 28, 2026 Session

**Session scope:** MEEP benchmarking, FP32 source build, Accelerate exhaustion, first end-to-end QE-Yambo-MEEP pipeline, MUPPET project setup, QGT from DFT prototype, GaAs EMI shielding workflow setup.

---

## Benchmark Findings (All Measured, M2 Max, 256^3)

| Configuration | Mcells/s | GB/s (eff) | Source |
|---|---|---|---|
| MEEP FP64, 1 thread | 211 | ~60 | conda pymeep |
| MEEP FP64, 8 threads | 208 | ~60 | conda pymeep |
| MEEP FP32, 1 thread | 335 | ~48 | source build --enable-single |
| Bare stencil FP64 (C loop) | 1125 | 54.0 | stencil_bench.c |
| Bare stencil FP32 (C loop) | 1311 | 31.5 | stencil_bench.c |
| Bare stencil FP32 (vDSP) | 1598 | ~77 | stencil_accelerate.c |
| Metal GPU FP32, full timestep | 2601 | 375 | meep_metal_fdtd.swift |

**Speedup decomposition (exact):** 12.3x = 1.59x (FP32 precision) x 7.8x (GPU bandwidth)

**OpenMP is useless for MEEP on M2 Max.** One core saturates DRAM bandwidth. Adding threads adds contention.

**Accelerate vDSP gives 1.25x on bare stencil** (~5% in actual MEEP via Amdahl). gcc -O2 was NOT auto-vectorizing well. But 5% vs 7.8x from GPU — Accelerate is exhausted.

## Building Single-Precision MEEP from Source

**Strategy that worked:** conda provides all deps (FFTW, HDF5, libctl, Harminv, MPB) via `pymeep` package. Build only MEEP from the release tarball with `--enable-single`.

**Use the release tarball, not git clone.** Git repo lacks pre-generated `meep_wrap.cxx`. The tarball has it.

**Three patches required:**
1. `sed` the configure script to search `$CONDA_PREFIX/include` for `ctl.h` (hardcoded to only search `/usr/include/ctl` and `/usr/local/include/ctl`)
2. Create stub `$CONDA_PREFIX/include/guile/gh.h` (removed in Guile 2.2+, conda ships Guile 3.x)
3. Build piecewise: `cd src && make install`, then `cd libpympb && make`, then `cd python && make install`. The scheme directory fails on `gen-ctl-io` — skip it.

**We never built MEEP from source before.** The `meep_arm` conda env installed a precompiled binary. This session was the first source build. The `ctl.h` and `gh.h` issues only appear when compiling, not when installing from conda.

## Yambo BSE

**BSE bands matter enormously.** Bands 1-20 caused a 2+ hour hang. Bands 3-6 finished in 8 minutes.

**DP vs SP build:** Yambo defaults to single precision. `--enable-dp` activates the `#if defined _DOUBLE` code paths where ZGEMM is called. Without DP, all ZGEMM call sites are dead code (CGEMM is used instead). apple-bottom integration requires DP rebuild.

**All ZGEMM calls funnel through two wrappers:** `mod_wrapper.F` (3 sites) and `mod_wrapper_omp.F` (6 sites). Zero direct calls elsewhere. Clean interposition point.

**p2y with DP build requires force_symmorphic = .true.** The DP p2y is stricter about symmetry operations than the SP version. Must be set in both SCF and NSCF QE inputs.

## First End-to-End Pipeline

**QE → Yambo → MEEP worked on first attempt** (after fixing `import os`). 500 nm Si thin film, 59% absorption at 1.6 eV.

**epsilon_inf = 0.5 warning:** The Lorentzian fitter pushes epsilon_inf to its lower bound (0.5), triggering MEEP's "Epsilon < 1" warning. Fix: raise lower bound to 1.0 in `yambo_eps_to_meep.py`.

**Pole 4 is dead** (sigma = 0.001). Three poles are sufficient for the photovoltaic workflow where solar spectrum is below 4 eV.

## QGT from DFT

**Quantum metric prototype validated:** g_xx(Gamma) = 42.59 A^2 matches analytical v^2/Eg^2 exactly for the 2-band k.p model.

**GaAs DFT calculation works:** SCF converged to -453.97 Ry. Band gap 0.51 eV (DFT-PBE underestimate, GW will fix). Wannier90 .mmn overlaps computed (39 MB, 512 k-points, 16 bands, 8 neighbors).

**GaAs has 9 occupied bands, not 4.** The Ga pseudopotential includes 3d electrons (Ga: 3d10 4s2 4p1 = 13e, As: 4s2 4p3 = 5e, total 18e = 9 bands). This caused Wannier90's `dis_froz_max = 8.0` to fail ("more states in frozen window than target WFs").

**We don't need Wannier90 to converge.** The quantum metric is gauge-invariant and computed directly from the .mmn overlaps. Wannier90's spread minimization is for the gauge-dependent part (Omega_D), which we don't need.

**First DFT quantum metric of GaAs:** Omega_I = 2.81 A^2. Absorption onset at 0.87 eV, peak sigma at 5.68 eV (near E2 critical point). Needs normalization cross-check against Wannier90's reported spread (requires fixing num_wann = 9 with appropriate projections).

## Swift/Metal Bugs (Recurring)

**String(format: "%s") with Swift Strings:** Still causing segfaults. Fixed TWICE in this session (meep_metal_fdtd.swift benchmark output). Use `%@` with `as NSString` or string interpolation. This is documented in LESSONS_LEARNED.md and keeps biting.

**Redeclared `dev` variable:** The `let dev = device` fix for init capture conflicted with `guard let dev = MTLCreateSystemDefaultDevice()` earlier in the same init. Just remove the duplicate.

**Bounds check in GPU stencil kernel:** GPU reads g1[i + stride] past buffer end at boundaries. CPU had bounds check but GPU didn't. Fixed with `if (i1 < 0 || i1 >= N) return;` in Metal kernel. Correctness went from FAIL (8.55e-03 error) to PASS (0.0 error).

## Pseudopotential Downloads

**QE's pseudopotential server is unreliable.** The As pseudopotential downloaded as a 6.9 KB HTML error page instead of a real UPF file. Always check `head -3 *.UPF` after download. QE's own test suite has working pseudopotentials: `PP/simple_transport/examples/scf/`.

## Architecture Decisions Made

1. **MUPPET is orchestration only.** No solver code, no acceleration kernels. Just converters, workflows, and docs.

2. **apple-bottom stays shared.** Both QE (corretto) and Yambo (yamborghini) consume it. The BLAS interface is universal.

3. **Stencil kernels stay in MEEPhistopheles for now.** Factor into maxwells-silver-hammer only when a second consumer appears.

4. **iron-butterfly is lowest priority.** Accelerate FFT is adequate. Profile first, build only if FFT is a measured bottleneck.

5. **QGT module lives in MUPPET.** It's a converter (DFT wavefunctions → material properties), same conceptual level as yambo_eps_to_meep.py. Not a standalone repo.

6. **The GaAs EMI shielding paper is the highest-impact next deliverable.** Reproduces your own experimental data from first principles using the QGT framework. Potentially publishable as "More Than Skin Deep" or "On the Same Wavelength."
