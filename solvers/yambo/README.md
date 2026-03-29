# Yambo Configuration

- **Install:** ~/yambo-build/yambo
- **Version:** 5.3.0, MPI+OpenMP+SLK+HDF5_IO
- **Conda env:** none (Homebrew deps, subshell isolation)
- **Key flags:** OMPI_CC=gcc-15, OMPI_FC=gfortran-15

## MPI note
Yambo's MPI exit warnings are cosmetic — calculations complete correctly.
The OMPI_CC/OMPI_FC fix prevents clang/gfortran ABI mismatch.

## Workflow: QE → Yambo
1. Run pw.x SCF + NSCF (force_symmorphic = .true.)
2. cd tmp/prefix.save && p2y
3. cp -r SAVE to working directory
4. yambo (setup) → yambo -d -g -p p (GW input) → yambo -F yambo.in
