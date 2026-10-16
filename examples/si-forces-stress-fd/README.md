# Example 2: forces and stress against finite differences of the total energy, on the reference and the apple-bottom pw.x

The question this example answers: are the analytic forces and the stress tensor that pw.x prints the true
derivatives of the total energy it computes, on the apple-bottom-free reference build and on the frozen
apple-bottom build with every GEMM on the GPU? A plain PAW SCF already exercises the `newd` DGEMM (QE calls
`newd` in every SCF iteration, `electrons.f90`; the B runs show 20-22 calls each), so that path is not new here.
What an SCF energy never exercises are `addusforce`/`addusstress` and the `stres_knl`/`stres_us` paths; forces
and stress do, and that is what this example adds.

```
Part A  norm-conserving Si (ONCV PBE-1.1, no NLCC), reference /usr/local/bin/pw.x, 1 rank
  A1  ecutwfc 16 -> 50 Ry on the 8-atom cubic cell at FIXED ecutrho 200 Ry and a PINNED 48^3 grid
        -> E(ecut) non-increasing (Rayleigh-Ritz, gated); the PW count (nested spheres, cannot decrease) and the
           Pulay staircase reported; one unpinned 20/24 Ry pair is run as a control (reported; it also decreased)
  A2  V(1 +- 0.5 %) cells at 20 and 40 Ry -> P_num = -dE/dV vs the printed P.  The Nielsen-Martin stress is the
        derivative at fixed basis, -dE/dV at fixed Ecut adds the Pulay stress, so the gated identity is
        P_num = P_an - (2 Ecut / 3V) dE/dEcut (Francis-Payne) from the (18,22) and (38,42) secants: the 40-Ry row
        bounds the stress to 2 % of P_an, the 20-Ry row (Pulay term 6.6 kbar) tests the identity itself;
        the uncorrected |P_an - P_num| is reported
  A3  frozen zone-centre optical phonon, 2-atom primitive cell, d = +-0.01 and +-0.02 A -> omega_TO from the +-d
        pair (cubic term cancelled) vs Raman 520.2 cm^-1; the one-sided omega(+d) and the cubic coefficient E3
        reported; the F1 = -F2 / [111] pair reported (imposed by net-force removal and the D3d symmetrisation)
  A4  one atom displaced in the 8-atom cell, nosym/noinv, the explicit 64-point k list in every run
        -> F_HF . u vs -(E(d+h) - E(d-h)) / 2h;  E(+x) == E(-x) (S4);  |F| equal on the C3 triple;
           rigid translation: one exact grid step (a/48) and a 64^3 grid gate the egg-box of the pinned grid; the
           0.1 A egg-box energy, the net force forces.f90 removes and the printed |F| after the translation are
           reported;  E(+d) - E(-d) (cubic term, reported);  stress isotropy gated on the five nsym = 1
           undisplaced/translated cells (the only runs where QE's symmatrix cannot impose it)
Part B  PAW Si (psl 1.0.0), 64-atom cubic cell at Gamma, atom 1 displaced 0.02 A along [111], nosym/noinv (nsym = 1:
        a [111] displacement of one atom still leaves the C3v about that axis, 6 ops, unless symmetry is switched off), nbnd 136
  B1  reference pw.x at delta and delta +- h (4 ranks)    -> PAW Hellmann-Feynman force vs finite difference
  B2  frozen apple-bottom pw.x 33fbaa8c8944 at delta, AB_MODE=cpu and AB_MODE=gpu AB_MIN_GPU_DIM=0, profiler on, 1 rank
        -> energy, all 192 force components, the stress tensor: cpu vs gpu (same pw.x) and ab_cpu vs reference
           (cross-build); dispatch evidence from the library's own counters only
```

Every launch goes through `tools/abrun` (`runner.run_sync`): the timing lock `~/Dev/.timing-lock` is honoured,
the lock mode is `share` by default (untimed correctness runs share the machine, PROTOCOL section 9; `--lock
auto` takes machlock build while another session holds a bench, and then also waits behind every other
share-mode pw.x on the machine), the shim sets nice 10, the load and the foreign solvers are recorded in each
run's `machine.json`, and one line per run is appended to `examples/common/solver-runs.log`. Walls are untimed
observations and no speedup is computed (abrun's own untimed wall quotient is stripped from the stored rows).

Rows come in two kinds. A **gate** (`checks.add`) can fail and is counted. A **reported** row (`checks.report`)
is a finding, a control that explains a restated gate, or a number that cannot fail by construction (QE
symmetrises the stress and the forces and removes the net force before printing, so "isotropic", "equal and
opposite" and "the forces vanish" are not tests on the runs where symmetry is on); it is never counted. The
headline is `Checks.summary()`: "N gated checks pass, M fail, K reported".

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py            # 37 pw.x runs; --skip-ab drops B2, --skip-b drops Part B
#   --ranks 4 (Part B reference), --ranks-a 1 (Part A), --lock auto|share|build|bench, --binary 33fbaa8c8944
#   --reuse reloads a finished record only when its input text, pw.x sha256, dylib sha256, pseudopotential sha256,
#   mode, ranks and nk all equal what a relaunch would pin now (anything else is relaunched, with the reason printed);
#   --reuse --no-launch re-evaluates the gates and refuses to launch anything a record does not cover
```

Inputs handed to abrun are written to `results/inputs/`; each run lives in `results/A/<name>/` or
`results/B/<name>/` with abrun's `run.json`, `summary.json`, `out.txt`, `err.txt`, `machine.json` and, for the
apple-bottom legs, `ab_profile.json`. The NC pseudopotential is copied into `results/pseudo/` (sha256 recorded;
the symlink beside the inputs is what abrun's UPF resolver finds); the PAW UPF is resolved by abrun from
`~/qe-test/xp/inputs/` against its pinned sha256. The report describes the records: ranks, nk, the binary, dylib
and pseudopotential hashes in `RESULTS.md` are read from each run's `run.json`, not from the command line.

## Gates

| check | gate | why |
|---|---|---|
| provenance | reference pw.x has 0 `_ab_` symbols; frozen pw.x and dylib match `SHA256SUMS`; dylib string `apple-bottom 1.4.0 (c5a1fa6a8166)`; NC UPF header NC/PBE/no NLCC; generated 64-atom positions equal `~/qe-test/xp/inputs/C_si64_160b_gamma.in` | what ran, exactly |
| variational theorem (A1) | every E(ecut_i+1) - E(ecut_i) <= +1e-9 Ry; the max PW count is reported (nested spheres, cannot decrease) | Rayleigh-Ritz over nested plane-wave subspaces at fixed ecutrho and grid is exact for NC pseudopotentials; the decrements (staircase) are reported, not gated |
| stress theorem (A2) | at 40 Ry \|P_num - (P_an + P_Pulay)\| < 0.5 kbar with P_Pulay = -(2Ecut/3V) dE/dEcut from the (38,42) secant; the uncorrected \|P_an - P_num\| (0.81 kbar) is reported, not gated | Nielsen-Martin: the analytic stress is the strain derivative at fixed basis, while P_num = -[E(V+) - E(V-)]/(V+ - V-) at the same cutoff, pseudo and grid is the derivative at fixed Ecut and includes the Pulay stress (Francis-Payne), so the corrected identity is the one that can be gated. 0.5 kbar is 2 % of P_an and ~3x the staircase scatter of the secant (its (38,40)/(40,42) halves give +0.90/+0.61 kbar). This row bounds the stress to 2 % of P_an: with a 0.76 kbar Pulay term any value in [0.26, 1.26] kbar would pass, so the identity itself is tested by the 20-Ry row. The design's uncorrected gate assumed the Pulay stress is below 0.5 kbar at 40 Ry; it is 0.76 kbar, and the 0.81 kbar mismatch equals it to within the 0.05 kbar residual |
| Pulay identity (A2) | at 20 Ry P_num - P_an = -(2Ecut/3V) dE/dEcut within 20 % of the larger side, dE/dEcut from the (18,22) secant; the residual at 40 Ry (secant (38,42)) at least 4x smaller | P_num = P_an - (2Ecut/3V) dE/dEcut is the exact chain rule between the fixed-Miller-set and fixed-Ecut derivatives of a smooth E(V, Ecut); the approximations are the secant over the plane-wave staircase, which makes a +-1 Ry difference noisy (the (18,20)/(20,22) halves give +7.80/+5.35 kbar, which is where the 20 % comes from; the 10 irreducible k-points of the 4x4x4 mesh average it), and the O(eps^2) central difference (~0.02 kbar); the secant error scales with the term itself, which falls 8.7x from 20 to 40 Ry |
| stress isotropy (A4, nsym = 1) | diagonal spread and off-diagonals <= 0.01 kbar in the five undisplaced/translated cells run with nosym/noinv (A4_undisplaced_nosym, A4_translated_0.1A, A4_translated_1grid and the two nr64 cells); the same number over the 17 symmetrised A1/A2/A1u runs is reported, not gated | with nsym = 1 QE's `symmatrix` (`PW/src/stress.f90`) has nothing to project with, so every term of the cubic cell's stress must come out isotropic on its own; the 17 symmetrised runs have nsym = 24 and the printed tensor is the Td-invariant part, which is isotropic whatever the off-diagonal terms were, so that row cannot fail |
| Hellmann-Feynman vs FD, NC (A4) | \|F_HF . u + (E(d+h) - E(d-h))/2h\| < 1e-4 Ry/bohr with h = 0.004 A; Total SCF correction < 1e-6 in the triplet (recomputed from the 8-decimal per-atom block of verbosity='high', rounding bound sqrt(24) x 5e-9 = 2.4e-8; the printed total has only 6 decimals) | expected \|F\| ~ 0.01 Ry/bohr. The FD is built from energies printed to 1e-8 Ry, so its resolution floor is 1e-8/2h = 6.6e-7 Ry/bohr; the O(h^2) truncation is E3 h^2/6 = 3.9e-6 Ry/bohr with E3 = 0.41 Ry/bohr^3 from the [111] asymmetry, which is where the observed 2.6e-6 sits; conv_thr 1e-13 over 2h (7e-12) is negligible. The A4 runs use conv_thr 1e-13 because the design's 1e-11 left a 2e-6 SCF correction on this cell (scratch probe on 2026-10-16, record not retained) |
| Hellmann-Feynman vs FD, PAW (B1) | same gate on the 64-atom Gamma cell; Total SCF correction < 1e-6 recomputed from the per-atom block (rounding bound sqrt(192) x 5e-9 = 7e-8) | a single Gamma point keeps the basis fixed across the triplet. The observed \|diff\| = 7.5e-7 sits at the 6.6e-7 Ry/bohr floor of the printed energies, so a PAW force defect below ~5e-7 Ry/bohr would be invisible to this comparison (reading the energy from the XML data file would lower the floor). Part B runs at conv_thr 1e-13 because the design's 1e-9 left a 3.7e-5 SCF correction (scratch probe on 2026-10-16 without nosym, record not retained), 37x its own precondition |
| frozen phonon (A3) | 500 <= omega_TO(+-d) <= 530 cm^-1 for d = 0.01 and 0.02 A, with omega(+-d) = sqrt(k/M) and k = [F1(-d) - F1(+d)]/2d; the two amplitudes agree to 1 %. Reported: the one-sided omega(+d) = sqrt(\|F1(+d)\|/(M d)) (520.9, 529.0), the cubic coefficient E3 (from -(F(+d) + F(-d))/d^2 at each amplitude and from a fit of F1 = -k d - E3 d^2/2 - E4 d^3/6 through the four forces), and the F1 = -F2 / [111] pair | a one-sided displacement carries the cubic term, omega(+d)^2 = omega(+-d)^2 (1 + E3 d/2k): 3 % of k at 0.01 A and 6 % at 0.02 A, so the design's one-sided 1 % gate measured the anharmonicity of the bond, not a solver error; the +-d pair cancels it and leaves the quartic E4 d^2/6 (0.26 % of k at 0.02 A), which the 1 % gate (2 % in k) bounds with the SCF force noise 1000x below. E3 is the quadratic coefficient of the force on atom 1; both atoms move, so d^3E/dd^3 of the pattern coordinate is 2 E3 (checked on the A3 energies: E(+d) - E(-d) = 2 E3 d^3/3 to 0.1 %). The pair row cannot fail: forces.f90 subtracts sumfor/nat, which for nat = 2 makes F1 + F2 = 0 identically, and the displaced cell keeps D3d (12 operations), so symvector imposes the [111] axis. M = 28.0855 x 911.444 Rydberg mass units, 1 Ry = 109737.3 cm^-1; Raman 520.2 +- 0.5 cm^-1 at 300 K (Parker, Feldman and Ashkin 1967; the earlier Temple-Hathaway 1973 attribution was not confirmed from that paper), LDA DFPT 517 (Giannozzi et al. 1991, a linear-response paper; the figure was not re-verified); the window covers the functional/lattice-constant spread (+-3 %), and the gated 513 cm^-1 is 1.4 % below Raman, on the underbinding side of PBE at a_exp (no PBE literature value is cited); +d compresses the bond, -d stretches it |
| symmetry identities (A4) | E(+0.02 A along x) == E(-0.02 A along x) to 1e-9 Ry (identical printed energies); \|F\| equal to 1e-6 Ry/bohr on the three nearest neighbours related by the C3 about [111]; the mirror-image forces of the +x/-x pair are reported | S4 about x maps the pair onto each other; with nsym = 1 nothing is symmetrised, so these are numerical tests; E(+d) - E(-d) along [111] is the cubic anharmonic term (reported) |
| translational invariance (A4) | (a) a translation by exactly one grid step (a/48) leaves E to 1e-7 Ry and every F to 1e-6 Ry/bohr; (b) the 0.1 A egg-box energy is smaller on a 64^3 grid than on the pinned 48^3. Reported: the rigid (0.1, 0.1, 0.1) A translation's printed \|F\| (1.8e-7) beside the net force forces.f90 removed (-4.4e-6 Ry/bohr per atom and component), and its energy change (the 48^3 egg-box, -5.9e-6 Ry) | the grid representation is exactly invariant under its own period, so (a) tests E and F with the egg-box excluded; the egg-box is XC-grid aliasing and must fall with the spacing, (b), with the caveat that the same 0.1 A translation is 0.884 of a 48^3 step but 0.178 of a 64^3 step, so (b) also samples two phases of the periodic egg-box (a same-fraction translation on 64^3 was not run); the energy egg-box of a 48^3 XC grid is of order 1e-5 Ry for this cell, so the design's \|dE\| < 2e-6 Ry was an estimate, not a physical statement; after the translation the eight atoms share one grid offset, so the egg-box force is a pure net force that forces.f90 subtracts (recovered from the per-term blocks of verbosity='high'), and the printed forces are only the inter-sublattice residual, which is why "the forces vanish" is reported as a consistency check of that subtraction and not gated |
| apple-bottom parity, same pw.x (B2) | cpu vs gpu_all: \|dE\| <= 1e-8 Ry, all 192 force components to 1e-7 Ry/bohr, stress to the printed 0.01 kbar | the honest BLAS A/B: same binary, only `AB_MODE` differs |
| cross-build (B2) | ab_cpu vs 7.4.1 OpenBLAS reference at delta: \|dE\| < 1e-6 Ry, max \|dF\| < 1e-5 Ry/bohr, max \|dsigma\| < 0.1 kbar | two meanings of "CPU only", different QE build: 7.4.1 with OpenBLAS vs develop 5b7f8e1bc with the local cegterg/regterg patches, -O3, apple-bottom GEMMs (cpu: Accelerate) and OpenBLAS otherwise (`~/qe-test/ab-frozen/33fbaa8c8944/PROVENANCE.txt`); the frozen build's Davidson is locally patched, so the 21-vs-19 iteration count, or a stress difference above 0.1 kbar with cpu == gpu intact, is a build difference, not an apple-bottom defect |
| dispatch evidence (B2) | cpu leg: gpu_calls == 0; gpu_all leg: gpu_calls == calls > 0, no_device == gpu_failed == other == 0 (too_large amber); dgemm row only (no zgemm: Gamma trick); dgemm max_m == max_k == 2 x Dense-grid G-vectors; build string == `apple-bottom 1.4.0 (c5a1fa6a8166)`; lib_loaded == the 33fb dylib | library counters only (`AB_PROFILE_JSON` at 1 rank), never the mode flag |

The DGEMM shape gate follows the QE source the frozen pw.x was built from (develop 5b7f8e1bc,
`~/qe-test/builds/mpi-gpu-ab-o3`), not the design note: `PW/src/newd_acc.f90:262` and `addusforce.f90:187` call
`DGEMM('C','N', nij, nab, 2*ngm_l, ...)` (K = 2 ngm), `addusdens.f90:160` calls `DGEMM('N','T', 2*ngm_l, nij,
nab, ...)` and `addusstress.f90:159` calls `DGEMM('N','N', 2*ngm_l, nspin, nij, ...)` (M = 2 ngm in both), so the
largest M is 2 ngm, not the Davidson's `npw2 = 2*npw` (`KS_Solvers/Davidson/regterg.f90:122`); 2 x PWs is
reported beside it. QE 7.4.1 has the same shapes at the same lines, except that `newd` is `newd.f90:141` there
and `npw2 = 2*npw` is `regterg.f90:124`.

QE subtracts the net force (`PW/src/forces.f90`, `force - sumfor/nat`), so "sum of forces = 0" is not a test; the
FD comparison and the S4/C3 equalities are. The same subtraction hides the rigid-translation egg-box force, which
is a pure net force when every atom sits at the same grid offset; the driver recovers it as the mean of the
per-term blocks (`forcenl + forceion + forcelc + forcecc + forceh + forcescc`), which `verbosity='high'` prints
unmodified after the subtraction. The FD triplet and the +x/-x pair use the same explicit 64-point
crystal k list with `nosym = .true., noinv = .true.`; with an automatic mesh QE would re-reduce the k set per
geometry and the derivative would pick up a sampling artefact. Part B uses `nbnd = 136` (nelec/2 = 128 plus an
8-band Davidson buffer) so that the FD triplet converges alike; no `nbnd = 128` control was run, so that is a
choice, not a measured statement.

GEMMs are a small share of the displaced 64-atom run (PAW_pot dominates), so the apple-bottom legs are about the
correctness of F and sigma on the DGEMM path; the gpu_all wall is an untimed observation.

## Results

Run on 2026-10-16 (first pass 1337 s for 32 pw.x runs, plus 324 s for the 5 control runs added under `--reuse`;
37 runs, 1635 s of QE wall in total, all untimed observations with another session's campaign and a sibling
example sharing the machine, load1 2.4-7 per run). Reference `/usr/local/bin/pw.x` (QE 7.4.1, sha256
e70ae3ba77a7, 0 `_ab_` symbols, OpenBLAS); frozen `~/qe-test/ab-frozen/33fbaa8c8944/pw.x` (sha256 2f0f029a5607,
dylib sha256 33fbaa8c8944, `apple-bottom 1.4.0 (c5a1fa6a8166)`, SHA256SUMS verified; QE develop 5b7f8e1bc with
the local cegterg/regterg patches, -O3). Part A at 1 rank with `Si_ONCV_PBE-1.1.upf` (sha256 d66fa1bb7336);
Part B with `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (sha256 8f630aadb550), reference at 4 ranks, apple-bottom legs at 1
rank (all read from the records). **34 gated checks pass, 0 fail, 13 reported** (`results/RESULTS.md`, regenerated
under `--reuse --no-launch` from the 37 stored records, every one matched on input, pw.x/dylib/pseudo sha256,
mode, ranks and nk). The repo commit in the header (bcae4b7) pins neither `run.py` nor `tools/abrun` (both
untracked when this was run), so `RESULTS.md` records the sha256 of `run.py`, `exlib.py` and every
`tools/abrun/*.py` instead.

The first pass (`results/run_first_pass.log`) had four red rows. Three were gates that had encoded an assumption
rather than the physics (the uncorrected stress identity, the one-sided phonon estimator, an estimated egg-box
bound): each was restated to the physical identity and the original comparison is kept as a reported row. The
fourth, the B1 precondition "Total SCF correction < 1e-6", failed on the 6-decimal printed total (1.0e-06) and
passes at 6.4e-7 after the value is recomputed as sqrt(sum F_scc^2) from the 8-decimal per-atom block of
`verbosity='high'` (rounding bound sqrt(192) x 5e-9 = 7e-8): a resolution re-read, not a control run; the printed
value is kept beside it in the row. All four were re-evaluated under `--reuse` without a launch
(`run_reuse_pass1.log` launched the 5 controls, `run_reuse_pass2.log`, `run_reuse_pass3.log` and `run_reuse.log`
launched nothing). The two conv_thr plan-value numbers quoted in the preconditions (2e-6 at 1e-11 on the 8-atom
cell, 3.7e-5 at 1e-9 on the 64-atom cell, the latter without nosym) come from a scratch probe on 2026-10-16 whose
records were not retained; they are not rows of `RESULTS.md`.

- **Variational (A1)**: E(ecut) decreases monotonically from 16 to 50 Ry at fixed ecutrho 200 / 48^3 (decrements
  1.4e-2 -> 3.4e-4 Ry per step, max PWs 1189 -> 6488, reported); the unpinned 20/24 control pair also decreased,
  3e-6 Ry from the pinned one (at this resolution the pin prevented nothing).
- **Stress theorem and Pulay (A2)**: 20 Ry: P_an +21.93, P_num +28.04 kbar, Pulay -(2Ecut/3V) dE/dEcut = +6.58
  kbar, residual -0.46 kbar (7 %): this is the test of the identity. 40 Ry: P_an +23.53, P_num +24.34, Pulay +0.76
  kbar, residual +0.05 kbar: |P_num - (P_an + P_Pulay)| < 0.5 kbar passes at 0.052 kbar (a 2 % bound on the
  stress), and the 40-Ry residual is 9x smaller than the 20-Ry one. The uncorrected |P_an - P_num| = 0.81 kbar (the
  design's gate, reported) equals the Pulay stress to within that residual: the Nielsen-Martin stress is the
  derivative at fixed basis, -dE/dV at fixed Ecut is not. The half-secants (38,40)/(40,42) give +0.90/+0.61 kbar,
  the staircase scatter behind the 0.5 kbar. Stress isotropic to the printed 0.01 kbar in the five nsym = 1 cells
  (gated: numerical) and in the 17 symmetrised runs (reported: imposed by `symmatrix`). P(50 Ry) = +23.7 kbar
  (si-eos-bands Birch-Murnaghan at 5.431 A: +19.7 kbar, different pseudo).
- **Frozen phonon (A3)**: the +-d estimator, which cancels the cubic term, gives 513.1 (0.01 A) and 513.6 cm^-1
  (0.02 A), 0.10 % apart (gate 1 %, window 500-530), 1.4 % below Raman 520.2 (LDA DFPT 517). The one-sided
  omega(+d) = 520.9 and 529.0 cm^-1 (1.54 % apart) are reported as the anharmonicity finding: omega(+d)^2 =
  omega(+-d)^2 (1 + E3 d/2k) with E3 = +1.80 Ry/bohr^3 from both amplitudes; the fit F1 = -k d - E3 d^2/2 -
  E4 d^3/6 through the four forces gives k = 0.559 Ry/bohr^2 (513.0 cm^-1), E3 = +1.80 Ry/bohr^3, E4 = +6.1
  Ry/bohr^4 with an rms residual of 9e-8 Ry/bohr (the SCF noise); the pattern energies confirm d^3E/dd^3 = 2 E3
  (E(+d) - E(-d) = 6.49e-5 vs 2 E3 d^3/3 = 6.49e-5 Ry at 0.02 A). The four A3 cells have nsym = 12 (D3d), so F1 =
  -F2 along [111] to 0.0 is reported, not gated.
- **Hellmann-Feynman vs FD, NC (A4)**: F_HF . u = -0.010975 vs F_FD = -0.010978 Ry/bohr (|diff| 2.6e-6, the
  O(h^2) truncation; floor of the printed energies 6.6e-7); S4 identity E(+x) == E(-x) to the printed 1e-8 Ry; |F|
  on the C3 triple equal to 6.7e-9 Ry/bohr with nsym = 1; E(+d) - E(-d) = +7.4e-6 Ry (cubic term, E3 = 0.41
  Ry/bohr^3), harmonic k = 0.282 Ry/bohr^2; nosym 64-k energy equals the symmetry-reduced 10-k one to 1e-8 Ry.
- **Translational invariance (A4)**: gate (a), a translation by exactly one grid step (a/48) leaves E unchanged to
  1e-8 Ry and |F| at the printed 1e-8; gate (b), the 0.1 A egg-box energy falls from -5.9e-6 Ry on 48^3 to -2.6e-6
  Ry on 64^3 (different phases of the egg-box as well, see the table). Reported: the energy egg-box of the pinned
  48^3 XC grid is -5.9e-6 Ry, of order 1e-5 Ry for this cell (the design's 2e-6 bound was 3x too tight); the net
  egg-box force forces.f90 removed before printing is -4.4e-6 Ry/bohr per atom and component (non-local
  -1.1e-5/-1.3e-5 and local +7.0e-6/+8.4e-6 on the two sublattices), so the printed 1.8e-7 is the inter-sublattice
  residual, not the egg-box force, and "every printed |F| < 1e-5" is a consistency check of the subtraction.
- **Hellmann-Feynman vs FD, PAW (B1)**: F_HF . u = -0.011073 vs F_FD = -0.011074 Ry/bohr (|diff| 7.5e-7, at the
  6.6e-7 floor of the printed energies: a defect below ~5e-7 would not show), SCF correction 6.4e-7 at conv_thr
  1e-13 (printed 1.0e-6; 21 iterations, ~60 s at 4 ranks); E(delta) = -2989.94039480 Ry, P = +46.15 kbar.
- **apple-bottom parity (B2)**: cpu vs gpu_all on the same frozen pw.x: dE = 0 (printed 1e-8), max |dF| = 2e-8
  Ry/bohr over all 192 components, max |dsigma| = 0.00 kbar, 19 iterations each. ab_cpu vs the 7.4.1 OpenBLAS
  reference: dE = 0, max |dF| = 8e-8, max |dsigma| = 0.00 kbar, 19 vs 21 iterations (two meanings of CPU only;
  different QE build with a locally patched Davidson).
- **Dispatch evidence (B2)**: 43,491 of 43,491 GEMMs on the GPU in gpu_all with 0 fallbacks of every kind, 0 of
  43,491 in cpu; dgemm only (no zgemm); dgemm max_m = max_k = 143,666 = 2 x 71,833 dense-grid G-vectors (2 x PWs
  = 17,846 is the Davidson's M); build string `apple-bottom 1.4.0 (c5a1fa6a8166)`; lib_loaded the 33fb dylib.
  QE walls (observations, not comparable): reference 59 s at 4 ranks, ab_cpu 169 s and ab_gpu_all 268 s at 1 rank.
