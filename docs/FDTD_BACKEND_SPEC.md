# FDTD backend spec (draft v0, 2026-11-11)

**Status:** draft for review. Nothing in this document is implemented yet.
**Scope:** the contract between MUPPET and any FDTD engine: MEEP, MEEPhistopheles, LoopFDTD, and later openEMS.
One case file goes in. Raw time series and a run record come out. MUPPET does all the analysis.

## 1. Why this exists, and the evidence behind each choice

`examples/metal-fdtd-crosscheck` and `examples/yee-dispersion` already run one problem on three engines. They drive
the Swift engines by editing their source text (`run.py`: `GAP_FIX`, `EMPTY_BOX`, then `swiftc`) and parse each
engine's own `timedomain.txt`. That works for two problems, but it does not scale. It also hid real differences
between engines. Each rule below comes from something those examples measured:

| finding (where it was measured) | rule it sets |
|---|---|
| MEEP's metallic boundary is the cell surface (box N·dx). The Swift engines guard index 0 and N-1, so they solve an (N-1)·dx box. Comparing against the wrong box passed by accident at 0.8 % (`metal-fdtd-crosscheck` README) | §3.2: the case fixes where the walls are. The record reports the box the engine actually solved. |
| A 5-cell PEC rod marked cell-wise (Swift) vs by Yee position (MEEP) shifts resonances by about 10 % | §3.4: conductors are defined per E sample, on a stated Yee convention. Engine-specific "cell marking" is translated by the adapter, never approximated. |
| `gapCells/2` integer division closed the feed gap in both Swift dipole programs | §3.4: geometry arrives as explicit index sets. Engines never derive it from lengths. |
| The Swift engines use c0 = 3e8, MEEP uses 299792458, so dt and every frequency differ by 0.07 % (`yee-dispersion/run.py` `C_LOOP`) | §3.3, §5: dt is set from a Courant number with a declared c. The record reports the realised dt and c. |
| MEEPhistopheles' dipole program evolves only Ez, Hx, Hy, with no z-derivative, so it is 128 independent 2-D planes | §3.1, §4: engines declare the fields and dimensions they support. Its cases are 2-D TM (`dimensions: 2`). A 3-D case is refused, never run as something else. |
| LoopFDTD's own validation suite reports "all pass" in `QUICK_START.md`, but 2 of 4 fail when run | §7: an engine counts only by passing MUPPET's conformance cases, never by its own report. |
| The V/I outputs in all three programs are uncalibrated proxies | §3.6, §6: engines write raw field samples. MUPPET computes voltage and current from them in one place. |

## 2. Data flow

```
case.json ──► adapter (rainbow-connection, lib/fdtd/backends/<engine>.py)
                 │  validates the case against the engine's capability manifest; refuses if unsupported
                 ▼
              engine (its own repo; run as an external process, never imported or linked)
                 │
                 ▼
          run_dir/record.json + run_dir/probes.npz
                 │
                 ▼
          lib/fdtd/analyze.py (shared: matrix pencil, FFT, V/I, S11) ──► gates / reports (exlib.Checks)
```

- Engines live in their own repos. rainbow-connection holds only the adapters, the schema, the analyzer and
  the conformance cases. This is the same split as for apple-bottom.
- **Interim mode (now):** an adapter may generate a patched copy of an engine's Swift program from the case
  (what `metal-fdtd-crosscheck` does by hand today) and build it under `results/bin/`. **Target mode:** the
  engine reads the case file itself (`<engine> --case case.json --out run_dir`). Target mode needs changes in
  the engine repos, so it is the owner's call (§9).

## 3. Case file (`case.json`, schema `muppet-fdtd-case/0`)

SI units throughout. One file defines one simulation. The case's identity is the sha256 of its canonical JSON:
sorted keys, no whitespace, and any referenced `.npy` mask files hashed in.

### 3.1 Top level

```json
{
  "schema": "muppet-fdtd-case/0",
  "name": "empty-pec-box-64x64x128",
  "dimensions": 3,
  "grid":   { "n": [64, 64, 128], "dx_m": 0.025 },
  "time":   { "courant": 0.5715767664977295, "c_m_per_s": 299792458.0, "steps": 2000 },
  "precision": "any",
  "boundaries": { "x-": "pec", "x+": "pec", "y-": "pec", "y+": "pec", "z-": "pec", "z+": "pec" },
  "materials": { "vacuum": { "eps_r": 1.0, "sigma_S_per_m": 0.0 } },
  "background": "vacuum",
  "regions": [],
  "sources": [ ... ],
  "probes":  [ ... ]
}
```

- `dimensions`: 1, 2 (TM-z: Ez, Hx, Hy) or 3. A 2-D case is the honest home for MEEPhistopheles' 3-field program.
- `precision`: `fp64`, `fp32` or `any`. A case that needs FP64 is refused by FP32 engines. All three Metal
  programs are FP32.
- Only uniform grids in v0. No subgridding and no non-uniform dx.

### 3.2 Yee convention and walls

Nodes sit at integer indices 0..N along each axis. The domain is [0, N·dx].

| component | position (units of dx) | component | position |
|---|---|---|---|
| Ex(i,j,k) | (i+½, j, k) | Hx(i,j,k) | (i, j+½, k+½) |
| Ey(i,j,k) | (i, j+½, k) | Hy(i,j,k) | (i+½, j, k+½) |
| Ez(i,j,k) | (i, j, k+½) | Hz(i,j,k) | (i+½, j+½, k) |

- `pec` on a face means the tangential E on that face plane is zero. The PEC box is then exactly N·dx (MEEP's
  convention). An engine that guards one extra plane (the Swift programs) must be given N+1 cells by its adapter.
  The record states the box it actually solved (`realised.box_m`).
- **Engine index maps (read from the stencils, checked in `lib/fdtd/test_case.py`).** The Swift dipole programs
  and `Shaders/FDTD.metal` stagger the other way: their Ex(i) sits at x = (i-½)·dx, Ey(j) at (j-½)·dx, Ez(k) at
  (k-½)·dx, and Hx(i,j,k) at (i, j-½, k-½), and so on. So their Ez(i,j,k) is the spec's Ez(i,j,k-1). Their node
  i sits at x = i·dx, and they hold tangential E at nodes 0 and N'-1, so a run with N' cells solves the spec box
  n = N'-1. Every adapter carries its engine's map, and the case file never uses engine indices.
- `pml`: `{ "type": "pml", "layers": 10, "grading": 3, "R0": 1e-6 }`. The layers sit inside the N cells, as in
  MEEP. The analyzer never places probes in them.
- `periodic` is reserved and not in v0.

### 3.3 Time

- dt = `courant` · dx / c, with c from the case. If an engine hard-codes a different c (the Swift programs: 3e8),
  the adapter keeps the case's Courant number, so the engine's own dt is `courant`·dx/c_engine. It also records
  `c_rel_mismatch` (-6.9e-4 for 3e8). The run is not blocked, but every frequency gate uses the engine's realised
  c and dt, the way `yee-dispersion` already does (`C_LOOP`). Patching c to match is a recorded patch, not a silent fix.
- Step n is time n·dt for E and (n+½)·dt for H. Sources and probes name the field they act on, so the half-step
  is never ambiguous.

### 3.4 Materials and regions

- `materials`: `eps_r`, `sigma_S_per_m` (electric conductivity), and optionally `mu_r`.
  `lorentz: [{omega_eV, gamma_eV, sigma}]` uses the same pole format as `yambo_material.json`, so the converters'
  output plugs in unchanged. Lorentz needs the `lorentz` capability, which none of the Metal engines has today.
- `regions` are applied in order, so later ones win:
  - `{ "kind": "box", "material": "<id>", "lo": [i,j,k], "hi": [i,j,k] }`: every E sample whose position lies in
    the closed box [lo, hi] takes that material.
  - `{ "kind": "pec_samples", "component": "Ez", "indices": "<file>.npy" | [[i,j,k], ...] }`: these E samples are
    held at zero. **This is the canonical way to describe conductors.**
  - `{ "kind": "pec_cells", "convention": "e_minus_half" | "e_plus_half", "indices": ... }`: sugar for "this
    cell's three E samples are zero". The convention is required because "the cell's E samples" depends on the
    layout. `e_minus_half` is the Swift programs' (and `FDTD.metal applyPEC`'s): cell (i,j,k) owns spec
    Ex(i-1,j,k), Ey(i,j-1,k) and Ez(i,j,k-1). `case.py` always expands it into `pec_samples`, so every engine
    receives the same samples. That makes the about 10 % cell-vs-Yee shift something you can see and choose,
    not something hidden in an adapter.
  - Index sets are written inline as `[[i,j,k], ...]`, as an inclusive box `{"lo": [...], "hi": [...]}`, or as an
    `.npy` file (hashed into the case). Overlapping regions are fine, since samples are de-duplicated.
- No continuum geometry (cylinders, radii) in the case file. Shapes are rasterised **once**, by MUPPET, into index
  sets. Then every engine runs the same conductor, and the 10 % cell-vs-Yee shift can't come back unnoticed.

### 3.5 Sources

```json
{ "kind": "soft_e", "component": "Ez", "indices": [[30,30,64], ...], "amplitude_V_per_m": 1.0,
  "waveform": { "type": "gaussian_modulated", "f0_Hz": 3.0e8, "t0_s": 3.333e-8, "width_s": 1.0e-8 } }
```

- `soft_e`: iteration n (n = 0..steps-1) updates H, then E, then adds A·s(n·dt) at the listed samples, with
  s(t) = sin(2π f0 t)·exp(-((t - t0)/w)²). This is the Swift programs' loop, exactly. If a source sample is also a
  PEC sample, PEC wins (it stays at zero). The Swift programs add the drive after zeroing their conductor, so there
  it survives for one step. `Case.sources_on_pec()` counts the overlap (5 in `c4-gap-dipole-closed-gap`, 0 in
  `c4-gap-dipole`).
- `hard_e` (E = s(t)) and `current` (J source, MEEP's native kind) are separate kinds. An adapter must not
  silently swap one for another: the mode weights differ, as the cross-check found.
- `plane_wave` (a total-field/scattered-field box) is reserved for v1. The si-thin-film and mie cases need it.

### 3.6 Probes

```json
{ "name": "gap", "kind": "e_samples", "component": "Ez", "indices": [[32,32,64]], "every": 1 }
{ "name": "arm_current", "kind": "h_loop", "around": "z", "center": [32,32,70], "half_width": 1, "every": 1 }
{ "name": "flux_top", "kind": "dft_plane", ... }
```

- Engines write **raw** samples only: `e_samples` and `h_samples`, plus `h_loop`, which is just the list of H
  samples on a rectangle. The analyzer turns them into V = -Σ E·dl and I = ∮ H·dl. No engine prints its own V
  or I, and no "voltage proxy".
- `dft_plane` (frequency-domain flux) is reserved for v1, for T/R/A cases.

## 4. Capability manifest (`lib/fdtd/engines.json`)

Each engine entry gives where it lives, how to build it, and what it supports. The adapter checks a case against
this list before running anything. What the code shows today (2026-11-11, from the source):

| capability | MEEP 1.31 (`meep_arm`) | MEEPhistopheles `meep_metal_fdtd.swift` | MEEPhistopheles `dipole_s11.swift` | LoopFDTD `Shaders/FDTD.metal` |
|---|---|---|---|---|
| dimensions | 1, 2, 3 | 3 (step_curl kernels) | 2 (Ez, Hx, Hy only) | 3 |
| precision | fp64 (fp32 in `meep-single`) | fp32 | fp32 | fp32 |
| eps_r per cell | yes | yes (`eps_inv` per voxel) | no | no (single `eps0`) |
| sigma | yes | yes (`step_curl_pml_cond`) | no | PML only |
| lorentz | yes | no | no | no |
| PEC per sample / per cell | per sample (via geometry) | not in the kernels | per cell | per cell (`applyPEC`) |
| PML | yes | yes (`step_curl_pml`) | no | yes (cubic grading) |
| walls | N·dx | to be measured | (N-1)·dx | (N-1)·dx as coded; boundary-guard defect (`yee-dispersion`) |
| c used | 299792458 | to be read | 3e8 | 3e8 in the program's host; the adapter's host uses the case's c (the kernels only see Cb, Db) |
| driven how | Python | Swift kernels only, no host program for cases | patched Swift program | patched Swift program |

openEMS isn't surveyed here. It enters the same way once `si-dipole-power-split`'s drivers are ported.

## 5. Run record (`run_dir/record.json`) and probes (`run_dir/probes.npz`)

```json
{
  "schema": "muppet-fdtd-record/0",
  "case_sha256": "…",
  "engine": { "name": "loopfdtd", "repo": "~/Dev/Claude/LoopFDTD", "commit": "…", "dirty": false,
              "binary_sha256": "…", "patches": ["boundary_guard_fix"], "precision": "fp32" },
  "realised": { "dt_s": 4.763139720814412e-11, "c_m_per_s": 3.0e8, "c_rel_mismatch": -6.918e-4,
                "box_m": [1.575, 1.575, 3.175], "grid": [64, 64, 128], "steps": 2000 },
  "wall_s": 1.9, "timing_grade": "observation", "throughput_Mcells_per_s": 979.9,
  "probes": { "gap": { "component": "Ez", "t0_s": 0.0, "dt_s": 4.763139720814412e-11, "shape": [2000, 1] } }
}
```

- `patches` names every source edit the adapter made. An empty list means the unmodified engine ran. This
  matters as much as abrun's `same_binary` / `cross_build` split: a patched engine is a different engine.
- `timing_grade` follows abrun and the house rules: `bench` only under the timing lock. Throughput is otherwise an
  observation and never becomes a "speedup" claim.
- `concurrent_jobs` is how many backend runs the test runner let overlap (`MUPPET_FDTD_JOBS`, from `--jobs N`).
  It is the effective number: 1 under the shared machine lock, which serialises every launch (`exlib.lock_mode()`).
  It is null when a backend was launched any other way, since then nothing knows. Above 1, `wall_s` and the
  throughput were measured while sharing the machine with up to N - 1 other runs, and every reported wall time
  says so. No gate reads a wall time; the solved numbers do not depend on N.
- The record is written by the adapter, not trusted from engine stdout. The adapter hashes the binary it launched.

## 6. Analysis (shared, `lib/fdtd/analyze.py`)

- Resonances: the matrix-pencil harmonic inversion from `yee-dispersion` (moved, not copied), plus an FFT for
  quick looks.
- V, I and Z from the raw probes. S11 only when a probe has been calibrated against a reference case. Until
  then, impedances are reported, not gated.
- Reference values come from analysis, not from another engine wherever possible: the discrete Yee dispersion
  relation for boxes, Airy (`exlib.airy_tra`) for 1-D slabs, and Mie for spheres.

## 7. Conformance cases (`lib/fdtd/cases/`)

An engine is usable for a class of problem only after it passes that class's cases. This mirrors abrun's
citable classes.

| id | case | gate | engines today |
|---|---|---|---|
| C1 | empty PEC box, 63x63x127, ring-down (`c1-empty-pec-box-ringdown.json`) | every line of amplitude >= 0.1 on the box's discrete Yee relation, to 1e-4 relative (0.01 %) | Through the adapters: MEEP pass (worst 2.05e-3 %); LoopFDTD guard-fixed pass (2.05e-3 %); LoopFDTD as shipped **fail** (1.32 %) |
| C1-2D | empty PEC box, 63x63, 2-D TM ring-down (`c2d-empty-pec-box-ringdown.json`) | same gate, TM_mn0 modes | MEEP pass and MEEPhistopheles pass (both worst 1.16e-4 %) |
| C2 | PML reflection, point source in a vacuum box | reflected/incident below -40 dB at the PML's design R0 | LoopFDTD's own test measured -10 dB; not run through MUPPET yet |
| C3 | 1-D dielectric slab, eps_r 12, plane wave | T/R vs Airy within 1e-3 | needs `plane_wave`, `dft_plane` (v1) |
| C4 | gap-fed dipole, conductors as `pec_cells` | resonance agrees between engines to 0.5 %, **reported, not gated**, until the probes are calibrated | MEEP, LoopFDTD, MEEPhistopheles (2-D) |
| C5 | 1-D Lorentz slab from `yambo_material.json` | T/R/A vs Airy with the same ε(ω), within 1e-3 | MEEP only |

**Running them.** `lib/fdtd/test_all.py` runs everything: the case-file tests, then every backend launch of the
three engine suites in one pool, then each suite's gates. Each launch writes `lib/fdtd/runs/<engine>/<variant>/<case>/`,
and the directory is its identity: two suites asking for the same run share one launch.
- `--jobs N` runs N launches at a time, with the 32,000-step ring-downs first.
- Use `MUPPET_SHARE=1` for these untimed correctness runs (PROTOCOL §9). Without it, the shared machine lock
  (machlock) serialises the launches whatever N is, and the runner says so.
- Each engine suite (`test_meep_backend.py`, `test_loopfdtd_backend.py`, `test_meephistopheles_backend.py`) also
  runs on its own with the same flags. The LoopFDTD suite then launches the MEEP C1/C4 runs it compares with.
- A full run exits 1 with exactly one FAIL, the as-shipped LoopFDTD C1 gate, which stays red by design.

C1 and C4 already run in `examples/`. They now exist as case files in `lib/fdtd/cases/`:
- `c1-empty-pec-box.json`
- `c4-gap-dipole.json`
- `c4-gap-dipole-closed-gap.json`, the as-shipped defect, kept as a regression case.

All three use the Swift programs' problem: a 63 x 63 x 127-cell box, conductors in `e_minus_half` cells. The
cross-check's MEEP mirror solves a different box (64 x 64 x 128 cells, 1.6 x 1.6 x 3.2 m) with block geometry. A
MEEP adapter has to run the case box and zero the case's exact samples before its C4 numbers mean anything.
MEEPhistopheles' dipole program evolves only Ez, Hx and Hy. Its recorded conductor size matches C4, but it can't
run a 3-D case, so it needs its own 2-D case once `dimensions: 2` is built.

## 8. What this spec doesn't do

- **No routing.** MUPPET doesn't pick "GPU when the grid is above 64³". A case names the engine, or a study runs
  several. Speed claims come only from bench-grade records of the same case on two engines.
- **No in-process linking** to Metal code from Python. Every engine is a subprocess. That keeps crashes, FP
  modes and Metal device state out of the orchestrator.
- **No continuum geometry**, no subgridding, no non-uniform grids, no near-to-far transforms in v0.

## 9. Order of work

1. **Done 2026-11-11.** `lib/fdtd/case.py` loads, validates, hashes and rasterises cases (3-D
   only). The C1/C4 case files are in `lib/fdtd/cases/`. `lib/fdtd/test_case.py`: 35 gated checks pass, 0 fail,
   3 reported. They show three things:
   - The case conductors equal the Swift program's own marking loop, re-run in Python: 100 and 105 cells.
   - The derived dt matches every engine's recorded value: Swift 4.763140e-11 s at c = 3e8, rel 5.9e-8; MEEP
     4.766437173827514e-11 s, exact.
   - The Swift engine grids are the case n + 1.

   Not built yet: `dimensions: 2`, sources other than `soft_e`/`hard_e`, and material boxes beyond rasterising
   them (no case uses one yet).
2. **MEEP adapter done 2026-11-11:** `lib/fdtd/backends/meep_backend.py`, run through
   `lib/fdtd/test_meep_backend.py`, which launches via `exlib.run`. 17 gated checks pass, 0 fail, 9 reported, on
   C1, C4, C4 closed-gap and a 32,000-step C1 ring-down. Facts about MEEP 1.31 that the adapter relies on, each
   measured on a small grid:
   - **Same Yee layout as the spec.** With a = dx and resolution 1, MEEP's staggering is the spec's, and a point
     source on a sample hits only that sample.
   - **Odd cell counts misalign.** With an odd count, MEEP's grid sits half a cell off the walls, and
     `geometry_center` makes sources and readback disagree by one cell. So each odd axis gets one extra cell plus
     a metal slab at n - ¼. Readback puts the walls at the case box to 1e-19.
   - **One conductor sample at a time.** A ½-cell `mp.metal` block, with averaging off, zeroes one E sample
     (readback 4e-18 against a probe peak of 774).
   - **Soft source reproduced exactly.** MEEP's point current adds -dt·J((n+½)dt) to E, so a current of
     -A·s(n·dt)/dt reproduces `soft_e` with a relative error of 0.
   - **C1 ring-down.** 15 matrix-pencil lines lie on the case box's discrete Yee relation, the worst at
     2.05e-3 %. The strongest, (1,3,2) at 315.2956 MHz, is off by -2.7e-5 %.

   Reported, not gated: with the conductor given as the same samples, C4's coarse 2000-step FFT peak is 280.0 MHz
   in MEEP against 279.5 MHz recorded by LoopFDTD. The cross-check's block-geometry mirror had about a 10 %
   shift. The first samples differ by exactly the c = 3e8 vs 299792458 factor (1.000712).

   **LoopFDTD adapter done 2026-11-11:** `lib/fdtd/backends/loopfdtd_backend.py` on the shared interim driver
   (`metal_interim.py`, `metal_host.swift`), run through `lib/fdtd/test_loopfdtd_backend.py`. 42 gated checks pass. 1 fails: the
   as-shipped C1 ring-down, a known defect, kept red.

   How it runs: the `updateE`/`updateH` kernels of `dipole_s11.swift` are extracted verbatim at build time
   (program sha256 30d2d98e…; LoopFDTD is not a git repository). Only the host loop is replaced. Optional
   recorded patches are `--guard-fix`, `--engine-c` and `--swift-source-order`.
   - **The adapter is the engine.** With the program's float32 constants and source order, it reproduces the
     gap-voltage series LoopFDTD recorded for C1, C4 and C4 closed-gap, to at most 6.3e-6 of the peak over 2000
     steps.
   - **c = 3e8 lives in the host, not the kernels.** So the adapter uses the case's c, and the 0.07 % offset
     disappears without touching LoopFDTD.
   - **Guard-fixed kernel vs MEEP, same case.** The gap-Ez series agree to 1.4e-7 to 5.2e-7 of the peak over 2000
     steps, and 1.9e-7 over the 32,000-step ring-down. That is float32 against float64 on the identical discrete
     problem. C1 ring-down: worst line 2.05e-3 %, the same as MEEP, so that residue is the frequency estimator's,
     not either engine's.
   - **As-shipped kernel.** It differs from MEEP by 1.8e-2 (C1), 0.15 (C4) and 2.3e-3 (closed gap) of the peak.
     Its C1 lines miss the Yee relation by up to 1.32 %.
   - **Observations, not benchmarks.** 32,000 steps on 64 x 64 x 128 cells: LoopFDTD 17 to 18 s; the MEEP
     backend 72 s. Both include per-step Python or CPU probe work, and neither was run under the timing lock.

   **MEEPhistopheles adapter and 2-D TM cases done 2026-11-11:**
   - `case.py` now validates `dimensions: 2` (Ez, Hx, Hy; width-2 indices; Courant ≤ 1/√2; four faces).
   - The MEEP backend runs 2-D cases (`dimensions=2`, blocks through z).
   - New files: `lib/fdtd/backends/meephistopheles_backend.py` (same shared driver and host);
     `lib/fdtd/cases/c2d-*.json`; `lib/fdtd/test_meephistopheles_backend.py`, which passes 32 gated checks with
     0 failing.

   What the work established:
   - **Why 2-D.** The program's kernels have no z-derivative, so its 64 x 64 x 128 runs are 128 independent 2-D
     TM planes, and only the feed plane k = 64 is driven.
   - **Its "fixed" dipole is an empty box.** The open gap leaves no conductor in the driven plane, so the "fixed"
     run is the empty box: same series, same 299.8 MHz peak. Its "shipped" run is the rod's 5-cell cross-section.
   - **Layout and walls.** The kernels' layout is the spec's 2-D layout with no index shift, and the box is
     (N-1) dx. Its ε0 is the literal 8.854e-12 and its source amplitude is 0.5.
   - **The adapter is the engine.** It runs only the driven plane (Nz = 3) and reproduces the program's recorded
     empty, fixed and shipped series to at most 9.4e-6 of the peak. The non-driven planes stay exactly 0.
   - **Against MEEP 2-D on the same case.** The gap-Ez series agree to 5.2e-7 (empty) and 4.3e-7 (cross) over
     2000 steps, and 9.4e-6 over the 32,000-step ring-down.
   - **Ring-down.** Both engines put the 2-D lines on the Yee relation to 1.16e-4 %. The strongest, (1,3,0) at
     300.8071 MHz, is off by under 1e-5 %.
   - **Observations, not benchmarks.** One 63 x 63 plane, 32,000 steps: MEEP 2.3 s, MEEPhistopheles 8.4 s. The
     Metal host waits on the GPU every step, which dominates at this size.

   **Follow-ups done 2026-11-26.** None of them changes a solved number. Every one of the 59 probe arrays in the
   23 conformance runs is byte-identical to the runs before these changes. The only record difference is the
   guard-fix patch label, now `guard_fix: per-component E boundary guard`, which no longer names a file.
   - **One definition of the LoopFDTD guard fix.** `GUARD_FIX_OLD/NEW`, `guard_fix()` and its record label moved
     from `examples/yee-dispersion/run.py` to `lib/fdtd/backends/loopfdtd_patches.py`.
     - The backend imports it, so it no longer runs an example script to get the fix.
     - The example loads it by path with `exlib.lib_module()`, the generalised `exlib.converter()` loader, without
       touching `sys.path`.
     - The lib/fdtd tests still load yee-dispersion's matrix pencil and Yee mode tables until step 3. They now
       refuse a case whose dx differs from that example's.
     - The constants and both of yee-dispersion's generated Swift programs are byte-identical, and the backend's
       guard-fixed kernel hash is unchanged (89aa3b18…).
     - yee-dispersion before and after: 29 gated checks pass, 0 fail, 1 reported. Every line and engine result
       is identical; only wall times differ.
   - **Cheaper MEEP probe reads.** The per-step loop calls `fields.get_field_from_comp` on field vectors built
     once. That is the call `get_field_point` makes, so the values are the same by construction, and the
     comparison above confirms it. The 32,000-step 2-D ring-down went from 2.26 s to 0.75 s, where the reads
     dominated. The 3-D ring-down is about 100 s either way: there the MEEP step dominates.
   - **Parallel runs.** `lib/fdtd/test_all.py --jobs N` (see §7). Supporting changes:
     - Metal host builds are atomic, so concurrent first builds of one kernel cannot race.
     - Every record carries the effective `concurrent_jobs`. `exlib.lock_mode()` is the one statement of which
       lock `exlib.run` uses, so the runner knows when machlock serialises everything.
     - A launch is named by its run directory, and two different launches naming one directory are refused.
       Identical launches run once; the old MEEPhistopheles suite ran one engine-c job twice.
     - 3-D MEEP runs moved from `runs/meep/<case>` to `runs/meep/case/<case>`, the layout of every other run.

     All 23 launches, same machine, same day: 190 s sequentially, 114 s with `--jobs 4`. The floor is the MEEP 3-D
     ring-down alone (about 104 s). Wall times are observations, not benchmarks.

   Every engine in the cross-check now runs from case files. Next: port
   `metal-fdtd-crosscheck` and `yee-dispersion` to them. The gate lines must be unchanged, the same rule as for
   any `examples/` refactor.
3. Move the matrix-pencil analysis into `lib/fdtd/analyze.py`. Both examples import it.
4. **(Needs the engine owner's decision; changes LoopFDTD and MEEPhistopheles.)** A `--case` host program in
   each engine repo, so the adapters stop patching source. The boundary-guard and gap fixes would land upstream
   at the same time, and the `patches` list in the records would go empty.
5. v1 features: `plane_wave`, `dft_plane` and per-cell eps_r in LoopFDTD. Then C3 and C5, which are where
   MUPPET's ab initio ε(ω) actually meets an accelerated engine.

## 10. Open questions

- C4 is written in the Swift cell semantics (`pec_cells`, `e_minus_half`), because that is what the Metal engines
  ran. Should a second C4, in `e_plus_half`, be added so the 10 % layout effect becomes a measured difference
  between two cases, not between two engines?
- `probes.npz` vs plain text: npz is proposed for size. The examples' existing `timedomain.txt` would become an
  analyzer export.
- Where `engines.json` lives: next to abrun's `binaries.json` under one "backend catalog" (the ab+FDTD
  unification discussed on 2026-11-11), or separately until that exists.
