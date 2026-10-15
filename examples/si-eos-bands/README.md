# Example 4: silicon equation of state and band structure against published PBE results

The question this example answers: does the QE install on this machine reproduce textbook PBE silicon,
and does the apple-bottom-linked pw.x reproduce the same physics along a whole energy-volume curve, not
just at one point?

```
QE pw.x SCF at 9 lattice constants (PBE, ultrasoft Si, ecutwfc 45 / ecutrho 360 Ry, 6x6x6 shifted mesh)
  -> Birch-Murnaghan fit -> a0, B0, B0'         vs Haas, Tran, Blaha PRB 79, 085104 (2009), Csonka et al. PRB 79, 155107 (2009),
                                                the all-electron Delta-test reference (Lejaeghere et al. 2016) and experiment
  -> the same 9 SCFs on the frozen apple-bottom pw.x, AB_MODE=gpu, AB_MIN_GPU_DIM=0
     (every energy within 1e-7 Ry of the reference binary; same fitted a0 and B0; profiler says 100 % GPU)
  -> SCF at the experimental lattice constant + bands run on Gamma, L and the Delta line (x = 0.70-1.00 of Gamma-X)
     -> indirect gap, CBM position, direct Gamma gap, L1c   vs the standard PBE values
```

## Run

```bash
~/miniconda3_arm64/envs/meep_arm/bin/python run.py          # ~3-5 min; --skip-ab drops the apple-bottom leg
```

## References and gates

| quantity | reference | gate |
|---|---|---|
| a0 (PBE) | 5.469 A (HTB 2009, Table I); all-electron WIEN2k 5.4695 A (Delta-test V0 = 20.4530 A^3/atom) | 5.449-5.489 A |
| B0 (PBE) | 89.2 GPa (Csonka 2009, Table V); all-electron 88.5 GPa | 82-96 GPa |
| B0' | 4.31 (all-electron Delta-test) | 3.5-5.0 |
| a0 (experiment) | 5.431 A (room temperature), B0 99.2 GPa (Csonka 2009) | PBE must overestimate a0 (known) |
| indirect gap at a_exp | ~0.61 eV (PBE); CBM at ~0.85 of Gamma-X; experiment 1.17 eV | 0.50-0.72 eV; 0.78-0.92 |
| direct gap at Gamma | ~2.56 eV (PBE); experiment 3.4 eV | 2.45-2.67 eV |
| L1c - VBM | ~1.5 eV (PBE); experiment ~2.0 eV | reported |
| X-point degeneracy | bands 5 and 6 (X1c) are degenerate by symmetry | split < 2 meV (a wrong k convention breaks this) |
| eigenvalue convergence | no `c_bands: eigenvalues not converged` | required |
| apple-bottom parity | same energies, same fit | max dE < 1e-7 Ry (printed precision 1e-8), da0 < 1e-4 A, dB0 < 0.1 GPa |

Birch-Murnaghan (third order) is fitted to E(V) with scipy; 1 Ry/bohr^3 = 14710.507 GPa. Both legs run with
OMP_NUM_THREADS=1 (the reference is MPI+OpenBLAS; apple-bottom's leg is mpirun -np 1). The reference
binary is `/usr/local/bin/pw.x` (nm-verified apple-bottom-free); the parity leg uses
`~/qe-test/ab-frozen/<hash>/pw.x`.

## Results

All checks pass (2026-10-15, 314 s). Reference pw.x: a0 = 5.4695 A, B0 = 88.7 GPa, B0' = 4.31 (fit rms
8.7e-7 Ry) against the all-electron Delta-test reference 5.4695 A, 88.5 GPa, 4.31. apple-bottom gpu leg:
every energy identical to the reference to the printed 1e-8 Ry, same a0 and B0 to 5 digits, 115,101 of
115,101 BLAS calls dispatched to the GPU. Bands at a_exp: Gamma25' threefold degenerate, X1c pair
degenerate to < 0.01 meV, indirect gap 0.575 eV with the CBM at 0.85 of Gamma-X, direct gap 2.563 eV,
L1c 1.537 eV above the VBM. See `results/RESULTS.md`.
