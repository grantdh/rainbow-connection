# Quantum ESPRESSO Configuration

- **Install:** ~/qe-test/q-e-qe-7.4.1
- **Conda env:** qe-src (for building only)
- **Acceleration:** apple-bottom DD-BLAS (ZGEMM +23-43%, DGEMM +14-20%)
- **Validated energy:** Si 64-atom: -2990.44276157 Ry

## Precision routing
- DGEMM ≥ 1024: Metal GPU (DD-BLAS)
- ZGEMM ≥ 1024: Metal GPU (Gauss 3-multiply)
- DSYRK/ZHERK: AMX (conversion overhead kills GPU)
- Everything else: Accelerate AMX
