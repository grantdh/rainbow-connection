# Si supercell folding: Bloch-theorem identities with apple-bottom parity and dispatch evidence

The question this example answers: does pw.x on this machine satisfy the exact identities that Bloch's
theorem imposes on an unrelaxed supercell, and does the frozen apple-bottom pw.x satisfy them on both of
its GEMM paths (real regterg/DGEMM at Gamma, complex cegterg/ZGEMM on a k mesh) while the library's own
counters show where every GEMM ran?

A 2x2x2 supercell of the 8-atom cubic silicon cell sampled at Gamma is the same calculation as the 8-atom
cell on its Gamma-centred 2x2x2 mesh: that mesh is exactly the set of parent k-points that fold onto the
supercell's Gamma (Popescu and Zunger, PRB 85, 085201 (2012)). With the FFT grids pinned so that the
supercell grid is exactly twice the parent grid (72^3 vs 36^3), the density-dependent energy per atom (total
minus Ewald), the pressure and the eigenvalue multiset agree to the printed precision of pw.x. The total
energy per atom differs by 4.3e-8 Ry, and that is the Ewald term alone: `PW/src/ewald.f90` truncates its
real-space sum at rmax = 4/sqrt(alpha), which on the 64-atom cells falls below the nearest-neighbour distance,
so the printed Ewald drops the erfc shell (reproduced here with a replica of the routine and an explicit sum
over the dropped shell). The 2-atom fcc primitive cell on the corresponding 32-point mesh is the same physics
on a non-commensurate xc grid; its 5e-5 Ry/atom discrepancy, which shrinks 5x when ecutrho is doubled, is
consistent with xc-grid aliasing (an attribution by elimination, see finding 3).

```
reference /usr/local/bin/pw.x (QE 7.4.1 OpenBLAS, nm-verified 0 _ab_ symbols), mpirun -np 4:
  R1  2-atom fcc primitive, 32 explicit tpiba points (the 2x2x2 conventional mesh unfolded onto the bcc reciprocal lattice)
  R2  8-atom cubic cell, 2x2x2, FFT 36^3 pinned          R3  8-atom, 4x4x4 (-nk 4)
  R4  64-atom supercell, K_POINTS gamma (Gamma trick)      R5  64-atom, automatic 1 1 1 0 0 0 (complex algorithm)
  R6  64-atom, 2x2x2 (-nk 4; = apple-bottom campaign input B)    R1', R2'  at ecutrho 200 Ry
frozen apple-bottom pw.x (~/qe-test/ab-frozen/33fbaa8c8944: apple-bottom 1.4.0 c5a1fa6, QE develop 7.5 -O3), mpirun -np 1:
  R4 and R6 in AB_MODE=cpu and in AB_MODE=gpu AB_MIN_GPU_DIM=0 with AB_REPORT, AB_PROFILE and AB_PROFILE_JSON
  (--auto-leg: R4 in AB_MODE=auto; --trace: R4 at the library floor on 3ae144a9cadd with AB_TRACE)
```

Every launch goes through `tools/abrun` (`runner.run_sync`): the timing lock `~/Dev/.timing-lock` is honoured
in every mode, the default lock is `share` (MUPPET_SHARE semantics: no machlock, `os.nice(10)` in the shim,
load and foreign solvers recorded in `machine.json`), the UPF sha256 is pinned here and in abrun's
`presets.UPF_SHA256`, the frozen build's SHA256SUMS are re-verified at every launch, and one exlib-format
line per run goes to `examples/common/solver-runs.log`. Run records live under `results/ref/<leg>/` and
`results/ab/<leg>/`.

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py     # 27 min measured 2026-10-16 with another session's campaign running
   --ranks N (reference legs; default 4)  --lock share|auto|build|bench  --binary <frozen id>
   --fast (skip R5 and the R6 gpu leg)  --auto-leg  --trace  --skip-ab
   --reuse (keep finished legs whose input, mode, binary id and sha256, dylib sha256, pseudo sha256, ranks and nk match)
   --reuse --no-launch (re-evaluate the gates from the run records; refuses to launch anything)
```

The 27 min (driver wall 1645 s, `results/run-first-pass.log`) was measured under lock `share` with load1 3-10;
it has not been measured on a quiet machine. Inputs: PAW `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (copied from
`~/qe-test/xp/inputs`, sha256 pinned), ecutwfc 25 / ecutrho 100 Ry (the campaign's deliberately low cutoff:
identities need no convergence), a = 10.2 bohr, `occupations fixed`, `nosym .false.`, `diago_full_acc .true.`,
`mixing_beta 0.5`, conv_thr 1e-10 on the 2- and 8-atom cells and 1e-9 on the 64-atom cells (conv_thr is per
cell), nbnd 24 / 160, tprnfor and tstress on. The 64 positions are asserted equal to `C_si64_160b_gamma.in`,
so the campaign's recorded energies apply.

## Gates

Gated rows print PASS/FAIL and are counted; REPORTED rows (findings, controls that explain a restated gate,
informational numbers) are never counted and cannot fail.

| gate | criterion |
|---|---|
| exact fold | density-dependent energy (total minus Ewald, which equals one-electron + Hartree + xc + one-centre PAW + descf; the descf remainder is printed beside it, <= 1e-8 Ry per cell): \|E(R4)/64 - E(R2)/8\| and \|E(R6)/64 - E(R3)/8\| < 3e-9 Ry/atom (Bloch theorem; the tolerance is one printed ulp of 1e-8 Ry per cell on each of the two printed values of both cells, 2 x 1e-8/8 + 2 x 1e-8/64 = 2.8e-9, rounded up; the SCF residual at conv_thr 1e-10 / 1e-9 adds < 2e-11); difference of differences of the four totals < 3e-9 by the same count. The total-energy fold is a REPORTED row (4.3e-8 and 4.4e-8 Ry/atom; gated < 2e-8 in the first pass): ewald.f90 picks alpha from its G-sum bound (1.1 on 8 atoms, 0.9 on 64) and truncates the real-space sum at rmax = 4/sqrt(alpha); on the 64-atom cells that rmax (4.216 bohr) is below d_nn = 4.417 bohr, so the printed Ewald misses the erfc shell by 4.5e-8 Ry/atom (2.9e-6 Ry per cell); the 8-atom cells (rmax 3.81 bohr, also below d_nn) miss 8e-10 because erfc(sqrt(alpha) d_nn) is 50x smaller there. The Ewald controls below gate that explanation |
| Ewald term | (a) a replica of ewald.f90's rule (alpha search, G-sphere of ecutrho, rmax = 4/sqrt(alpha), rgen's inclusive r <= rmax) reproduces every printed Ewald to 5e-9/nat + 1e-11 Ry/atom (the printed half-ulp per atom plus the replica's own floor: b1's alpha spread 3.6e-15, the G tail beyond its 300 Ry sphere <= 1.8e-12 by ewald.f90's own bound, the shell beyond 20 bohr < 1e-150); (b1) the converged lattice sum (G to 300 Ry, R to 20 bohr) is one number, alpha-independent to 1e-14 (float64 ulp 1.8e-15); (b2) printed == converged - dropped real-space shell (r > rmax) - dropped G tail (\|G\|^2 > ecutrho), both summed explicitly over the dropped regions (Ry, e^2 = 2), to 5e-9/nat + 1e-11 Ry/atom on every cell; (b3) on the fold cells R2-R6 the shell alone closes it to G tail + 5e-9/nat + 1e-11 (the 2-atom cells' deviation is the G tail, reported); fcc primitive Ewald equals the cubic cell's to 1e-7 (QE's G-sum bound per cell); the R2-R6 spread is a REPORTED row |
| term identities | one-centre PAW, Hartree, xc, one-electron per atom across R2/R4 and R3/R6 < 1e-6 Ry/atom, an empirical tolerance: 2.5x the largest same-cell scatter in the control (R4 vs R5, 4.0e-7 Ry/atom), not derived from conv_thr. Physics: an individual term is first order in the density residual while the total is variational (second order); the sum of the four terms is the 3e-9 fold gate; the scatter control (R4/R5, R4/R4_cpu, R4_cpu/R4_gpu, R6/R6_cpu, R6_cpu/R6_gpu) is a REPORTED row |
| unfolding | sorted R4 Gamma spectrum == sorted R2 mesh eigenvalues with multiplicities 1,3,3,1 below E* = min_k eps_21(k) (cut at E* - 2 meV): same count, max \|delta\| < 1 meV; the same below min_k eps_24 - 2 meV, which adds the conduction states (E* is the LUMO here) and is complete while the supercell's nbnd = 160 exceeds the count; degeneracy count per parent level; HOMO/LUMO of R4 = max_k eps_16 / min_k eps_17 of R2 to 1 meV |
| cubic vs primitive | E(R1)/2 - E(R2)/8 is a REPORTED row with the pressure signature beside it (consistent with xc-grid aliasing: the fcc 24^3 and cubic 36^3 grids are not commensurate; the attribution is by elimination, a grid-only control at nr = 48 / ecutrho 100 was not run); gate: the discrepancy at ecutrho 200 (R1' vs R2') is smaller by >= 2x (both grids 1.5x denser per direction) |
| Gamma trick vs complex | \|E(R4) - E(R5)\| <= 2e-8 Ry (two printed units, one per printed total; the estimated scf accuracy is below 1e-9 on both; was 1e-8 in the first pass); 160 eigenvalues equal to 1e-4 eV; only R4 prints `gamma-point specific algorithms are used`; iteration counts a REPORTED row |
| final-iteration c_bands | reference legs: no `eigenvalues not converged` in the last SCF iteration; frozen-build legs: a finding gated on its control only (every eigenvalue equals the reference's to 1e-4 eV, and cpu and gpu modes warn identically: same iterations, counts and ethr); the cause is not isolated (see finding 5) |
| symmetry | forces: the Td-site atoms (R1 both atoms; R2/R3 the 4 and R4-R6 the 8 atoms at (i/2, j/2, k/2)) are exactly zero as printed (fixed by every operation QE keeps); the other atoms' symmetry-allowed residuals ([111]-type on the C3v sites of sublattice 2, in a mirror plane on the Cs face-centre sites, since QE drops the fractional translations on a supercell) <= max(QE's printed `Total SCF correction`, 5e-7) Ry/bohr, a convergence check, with QE's `SCF correction compared to forces is large` warning reported; stress isotropic to 0.01 kbar (a symmetrization consistency check: QE's symmatrix makes the printed tensor isotropic by construction); pressure equal across fold pairs to 0.05 kbar; R1 `48 Sym. Ops., with inversion` and 32 k-points as given; R2/R4/R6 `This is a supercell`, `24 Sym. Ops. (no inversion)`; R2 and R6 report 4 k-points |
| recorded energies | \|E(R4) + 2989.94060060\| and \|E(R6) + 2990.44093535\| < 2e-6 Ry, all recorded at conv_thr 1e-6. R4: QE 7.4.1 (`~/qe-test/baseline/si64.out`, 2026-03-22) and two builds, -O0 and -O3, of QE develop 5b7f8e1bc at 1 and 8 ranks (apple-bottom `benchmarks/results/2026-10-14-1dbdb52-qe-si64` and `2026-10-15-c5a1fa6-qe-si64-opt`): -2989.94060060 three times, -2989.94060129 at 8 ranks. R6: the two develop builds only, -2990.44093535 twice and -2990.44093534 at 8 ranks / nk 4; no 7.4.1 value exists |
| apple-bottom parity | same frozen pw.x: \|E_cpu - E_gpu\| < 1e-8 Ry, eigenvalues to 1e-4 eV, forces and stress to printed precision; cross-build \|E_abcpu - E_ref\| < 1e-6 Ry (two meanings of "CPU only", labelled; the abrun compare tables use the same 1e-6) |
| dispatch evidence | cpu legs gpu_calls == 0; gpu legs gpu_calls == calls > 0 with no_device == gpu_failed == other == 0; the library's own AB_MODE echo equals the requested mode; R4 dgemm only, R6 dgemm + zgemm; AB_REPORT build string == strings ab_build_info; dyld-loaded dylib == the frozen one |
| profiler dimensions | R4: dgemm calls with M = 2 x PWs and K = 2 x PWs (regterg npw2) exist, dgemm max_m == max_k == 2 x G-vectors (addusdens / newd); R6: zgemm max_m == max_k == max PWs over k, dgemm max_k == 2 x G-vectors, zgemm max_n <= max(2 nbnd, nkb) (a loose upper bound) |

Walls are reported from each run's `summary.json` (`wall_qe_s`, `timing_grade` untimed under lock share with
other sessions active) as observations only; the reference legs run at 4 ranks and the frozen legs at 1, so
the two are never comparable, and `gpu` mode is expected to be slower than `cpu` on these shapes because every
small GEMM is forced to the GPU. No wall ratio is formed anywhere in this example. The GUI's "GPU" toggle buys
parity and visibility, not speed.

## Results

Measured 2026-10-16 00:39-01:07 (`results/run-first-pass.log`: driver wall 1645 s with another session's si64
campaign running on the machine, lock `share`, load1 3-10). Reference legs: `/usr/local/bin/pw.x` (QE 7.4.1,
sha256 e70ae3ba77a7, 0 `_ab_` symbols), `mpirun -np 4`, `-nk 4` for R3/R6. apple-bottom legs:
`~/qe-test/ab-frozen/33fbaa8c8944/pw.x` (sha256 2f0f029a5607, dylib 33fbaa8c8944,
`apple-bottom 1.4.0 (c5a1fa6a8166)`, QE develop 7.5 -O3), `mpirun -np 1`, `OMP_NUM_THREADS=1`.
`results/RESULTS.md` is regenerated from those run records with `--reuse --no-launch` (log
`results/run-reuse.log`; every leg reused on its sha-checked record, none relaunched); its header states the
regeneration time, the run-record range and the sha256 of `run.py` and of `tools/abrun/*.py`, because the git
commit it names does not contain them. **77 gated checks pass, 0 fail, 6 reported** (`Checks.summary()`).

How the gates got here. The first pass (`results/run-first-pass.log`) had 16 red rows: 3 were a parser defect
(the `(no inversion)` note kept its parentheses) and 13 were assumptions the controls below refuted. The second
pass (`results/run.log`) had 15 red rows, among them the first Ewald control (gated at 2e-8), which itself
failed on the 64-atom cells and led to controls a/b1/b2/b3, and the frozen build's final-iteration c_bands
row. Each gate was then restated to the identity it tests and the original comparison kept as a REPORTED row
with its number. By gate:

- The apple-bottom parity and dispatch gates are unchanged: every tolerance is identical between
  `run-first-pass.log` and `run-reuse.log`, and the criterion text differs only by the added AB_MODE-echo
  clause on the four counter rows, the campaign-count caveat (conv_thr, diago_full_acc) on the two R6
  routine rows, "energies only" on the two cross-build rows and "loose upper bound" on the zgemm max_n row.
- Three solver-identity gates were loosened, each with a control: Gamma trick vs complex 1e-8 -> 2e-8 Ry
  (two printed units; the estimated scf accuracy is below 1e-9 on both runs); the eight per-term identities
  2e-8 -> 1e-6 Ry/atom (empirical, 2.5x the R4/R5 same-cell scatter control, which is a REPORTED row); the
  total-energy fold 2e-8 -> REPORTED (the Ewald controls a/b1/b2/b3 gate its explanation).
- Restated rather than loosened: the fold is now gated on the density-dependent energy at 3e-9 Ry/atom
  (derived above; measured 4.7e-10 and 6.3e-10); the forces row became two rows (Td-site identity, exact
  zero; convergence check against QE's own SCF-correction estimate) instead of `< 1e-6` on every atom, which
  the 24-operation group does not impose; the 156-state unfolding window is gated (it was informational);
  the Ewald spread, fcc-vs-cubic, iteration-count and scatter-control rows are REPORTED instead of counted
  as PASS.

**apple-bottom parity and dispatch (all green).** On the same frozen pw.x, `ab_cpu` and `ab_gpu_all` give
identical printed total energies, all 160 (R4) and 640 (R6) eigenvalues and stress (0.00 kbar): R4
-2989.94060157 Ry, R6 -2990.44093548 Ry, 12 SCF iterations each; term by term R4 is identical and R6 agrees
to the printed digit (terms within 2e-8 Ry per cell, forces within 1e-8 Ry/bohr). Both also equal the
reference build to the last printed digit (cross-build dE = 0 on R4 and R6 against the 1e-6 Ry gate).
Counters: R4 dgemm 21062 calls, 0/21062 on the GPU in cpu mode, 21062/21062 in gpu_all, no zgemm row (Gamma
trick); R6 dgemm 3741 + zgemm 77067, 0 and 80808/80808 on the GPU; no fallback of any kind; AB_REPORT echoes
the requested mode, its build string equals the dylib's `strings` ab_build_info and dyld loaded the frozen
dylib in every leg. Dimension pins: R4 dgemm max_m = max_k = 143666 = 2 x 71833 G-vectors (addusdens /
newd), regterg's npw2 = 2 x 8923 = 17846 appears as M in 585 and as K in 416 of the per-call lines; R6 zgemm
max_m = max_k = 18016 = the largest printed PWs over the 4 k-points, dgemm max_k = 287330 = 2 x 143665, zgemm
max_n 320 <= max(2 nbnd, nkb) = 1152. QE walls, observations only (1 rank, nice 10, shared machine; in
parentheses the contention events recorded during the run and load1 at acquisition): R4 cpu 110 s (5, 6.6),
gpu_all 178 s (8, 3.2); R6 cpu 318 s (15, 4.5), gpu_all 676 s (33, 3.6). The reference legs at 4 ranks (R4
35 s (1, 9.9), R5 75 s (3, 6.8), R6 220 s (10, 8.0)) are not comparable with them; no ratio is formed from
any of these numbers.

**Bloch-theorem identities.** Energies (Ry): R1 -93.43575106, R1' -93.43558250, R2 -373.74257485, R2'
-373.74241014, R3 -373.80511658, R4 -2989.94060157, R5 -2989.94060158, R6 -2990.44093548 (R4 and R6 within
9.7e-7 and 1.3e-7 Ry of the campaign's recorded values at conv_thr 1e-6). The unfolding is exact to the
printed 0.1 meV: 128 = 128 states below E* = 7.1054 eV (16 parent levels, every degeneracy count = weight x
parent degeneracy; HOMO/LUMO 6.4521/7.1054 eV in both cells), and 156 = 156 states below the last computed
parent band (9.0009 eV). E* = min_k eps_21 coincides with the LUMO here because the sixfold X1c manifold folds
onto Gamma. Pressure 46.14 kbar in R2 and R4, 36.88 in R3 and R6; stress isotropic to 0.00 kbar. Forces: the
Td-site atoms are exactly zero in every leg; the other atoms carry residuals of 3.0e-8 (R3) to 3.4e-7 Ry/bohr
(R6 on the frozen build) against QE's `Total SCF correction` of 1e-6 to 5.2e-5, and QE prints
`SCF correction compared to forces is large` on every cubic-cell leg. R1 keeps 48 ops with inversion and its
32 points as given, the cubic cells are supercells with 24 ops (no inversion) and 4 irreducible points of
weights 1,3,3,1 (R3: 10). The Gamma-only sampling error folds too: [E(R4) - E(R6)]/64 and [E(R2) - E(R3)]/8
agree to 1.1e-9 Ry/atom (7.8177 mRy/atom; campaign 7.8).

**Findings behind the restated gates (numbers kept, controls in the check text).**

1. *The total-energy fold is 4.33e-8 (R4/R2) and 4.44e-8 Ry/atom (R6/R3), and the whole of it is the Ewald
   term (-4.37e-8).* The gated identity is the density-dependent energy, which folds to 4.7e-10 and
   6.3e-10 Ry/atom; the total is a REPORTED row with the cause in its text. What `PW/src/ewald.f90` does: it
   lowers alpha from 2.9 in steps of 0.1 until its G-sum error bound 2 Q^2 sqrt(2 alpha/2pi)
   erfc(sqrt(ecutrho/4 alpha)) is <= 1e-7 Ry (Q = total ionic charge, so 1.1 on the 8-atom cell and 0.9 on
   the 64-atom cells), sums G over the ecutrho sphere, and sums real space over pairs with
   r <= rmax = 4/sqrt(alpha) (`rgen`; the source comment says "terms up to ZiZj*erfc(4) are counted"; the
   stress routine `stres_ewa.f90` keeps terms to erfc(5)). That cutoff is a documented tolerance of the routine,
   not a defect: it is what the example quantifies and closes, and the gated identity avoids it. On the
   64-atom cells rmax = 4.216 bohr is below the nearest-neighbour distance 4.417 bohr, so the four nearest
   neighbours of every atom are dropped: erfc(sqrt(0.9) x 4.417) x 16/4.417 x 4 = 4.5e-8 Ry/atom, 2.9e-6 Ry
   per cell. The 8-atom cells also have rmax (3.81 bohr) below d_nn, but at alpha 1.1 the dropped shell is
   8.3e-10 Ry/atom (erfc is 5.7e-11 there against 3.1e-9), which is why the campaign and the 8-atom cells
   never showed it. Verified three ways: a replica of the routine reproduces every printed Ewald to
   7.4e-10 Ry/atom (worst R1', where the printed half-ulp per atom is 2.5e-9); a converged lattice sum (G to
   300 Ry, R to 20 bohr) is one number, -8.449879284928, for all five cells and alpha-independent to 3.6e-15;
   and printed == converged - (shell beyond rmax) - (G tail beyond ecutrho), both summed explicitly, closes
   every cell to < 8e-10 Ry/atom (64-atom cells to 1e-12). On the 2-atom fcc cells the rule's larger alpha
   (1.3 and 2.6) makes the shell negligible and the G-sphere truncation the whole deviation (1.5e-8 and
   2.0e-8, under the routine's own 2.3e-8 and 3.3e-8 bound), so the shell alone closes only the fold cells
   R2-R6; the full identity closes all.
2. *Per-term identities (one-electron, Hartree, xc, one-centre PAW) differ by 3e-8 to 4.6e-7 Ry/atom.*
   Individual terms are first order in the density residual; the total and the sum of the four terms are
   variational (second order), and that sum is the 3e-9 gate. Control: the same 64-atom cell on the two
   Davidson algorithms (R4 vs R5) scatters by up to 4.0e-7 Ry/atom term by term while the totals agree to
   1.6e-10; the two QE builds (R4 vs R4_cpu) by 5.5e-8 with dE = 0; cpu vs gpu on one binary: R4 exactly, R6
   by 3.1e-10 (2e-8 Ry per cell) with dE = 0. The per-term gates are 1e-6, an empirical 2.5x the measured
   scatter, labelled as such in the criterion; the gate cannot catch a per-term defect smaller than that
   scatter.
3. *fcc primitive vs cubic cell -5.37e-5 Ry/atom at ecutrho 100, +1.00e-5 at 200 (the sign flips).* The
   fcc 24^3 and cubic 36^3 xc grids are not commensurate, so the GGA xc energy density is sampled on different
   point sets. This is consistent with xc-grid aliasing and the attribution is by elimination: the ecutrho
   200 control changes the density G-sphere and the FFT grids together, and the grid-only control (R2 at
   nr = 48 with ecutrho 100) was not run. The pressure signature points the same way: R1 47.09 vs R2 46.14
   kbar (0.95 kbar apart) at ecutrho 100, R1' 46.52 vs R2' 46.65 (0.13 kbar) at 200, a 7x shrink against
   5.4x in the energy; the energy ratio is gated (>= 2), the pressure pair reported.
4. *Gamma trick vs complex algorithm: 1.0e-8 Ry.* One unit in the last printed digit (final estimated scf
   accuracy 7.5e-10 and 2.2e-10 Ry); gate <= 2e-8 (two printed units); all 160 eigenvalues agree to 0.1 meV
   and both took 12 iterations.
5. *Final-iteration `c_bands` warning in the frozen build's R6 legs: a finding about that build, with its cause
   not isolated.* 4 of 640 eigenvalues did not reach ethr = 9.55e-13 in the last Davidson call on the frozen
   build (QE develop 7.5 + patched cegterg/regterg, Accelerate GEMMs) at 1 rank; the warning sequence
   (iteration, count, ethr) over iterations 4-12 is identical in `AB_MODE=cpu` and `AB_MODE=gpu`, so GPU
   numerics are ruled out; the 7.4.1 / OpenBLAS / 4-rank reference warned only in iterations 4-5 and not at
   its final ethr 8.92e-13. The reference differs from the frozen build in QE version, Davidson patch, rank
   layout and BLAS vendor at once, so none of those is isolated (an unpatched QE-develop leg, or a 1-rank
   7.4.1 leg, is the missing control). Every eigenvalue equals the reference's to the printed 0.1 meV, and
   energy, forces and stress are identical (gated in the parity rows).
6. *Forces on the ideal lattice are not zero on every atom, and QE says so.* Under the 24 point operations
   QE keeps on a supercell (fractional translations disabled), only the atoms at (i/2, j/2, k/2) have Td site
   symmetry; the sublattice-2 atoms have C3v and the face-centre atoms Cs, so [111]-type and in-plane
   residuals are allowed. The records show exactly that: the 8 Td-site atoms are 0.00000000 in every 64-atom
   leg, the other 56 carry 1.8e-7 to 3.4e-7 Ry/bohr, and QE prints `Total SCF correction` 1.5e-5 to 5.2e-5
   with `SCF correction compared to forces is large: reduce conv_thr` on every cubic leg (R2 and R3 too, at
   0.000000 / 0.000001). The first pass gated every component < 1e-6 as a symmetry identity; now the Td-site
   zero is the identity and the residuals are a convergence check against QE's own estimate.

![energies per cell and band folding](results/si_supercell_folding.png)
