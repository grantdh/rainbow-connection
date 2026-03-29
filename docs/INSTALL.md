# Installation Guide

Setting up the full MUPPET pipeline on Apple Silicon. Each solver is independent — install only what you need.

## Prerequisites

- macOS 13+ (Ventura or later) on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- ARM64 Miniconda: download from https://docs.conda.io/en/latest/miniconda.html (Apple Silicon version)

Verify you're running native ARM64:

```bash
arch
# Should print: arm64

file $(which python)
# Should include: arm64
```

## MEEP (Electromagnetic FDTD)

The easiest solver to install. Takes 2 minutes.

### FP64 (recommended for production)

```bash
conda create -n meep_arm -c conda-forge pymeep python=3.11 -y
conda activate meep_arm
python -c "import meep; print(f'MEEP {meep.__version__}, single={meep.is_single_precision()}')"
```

### FP32 (for benchmarking only)

Requires building from source. See docs/LESSONS_LEARNED_MEEP.md Section 2 for the full procedure. Summary:

```bash
conda create -n meep-single -c conda-forge pymeep python=3.11 -y
conda activate meep-single

# Download release tarball (NOT git clone)
curl -L https://github.com/NanoComp/meep/releases/download/v1.31.0/meep-1.31.0.tar.gz -o meep-1.31.0.tar.gz
tar xzf meep-1.31.0.tar.gz && cd meep-1.31.0

# Patch configure for conda's ctl.h location
sed -i '' "s|for ctl_h_path in /usr/include/ctl /usr/local/include/ctl|for ctl_h_path in /usr/include/ctl /usr/local/include/ctl $CONDA_PREFIX/include|" configure

# Create stub guile/gh.h (removed in Guile 2.2+)
mkdir -p $CONDA_PREFIX/include/guile
echo -e '#ifndef GH_H_STUB\n#define GH_H_STUB\n#include <libguile.h>\n#endif' > $CONDA_PREFIX/include/guile/gh.h

# Configure with --enable-single
CP=$CONDA_PREFIX
CPPFLAGS="-I$CP/include" CFLAGS="-O2 -I$CP/include" CXXFLAGS="-O2 -I$CP/include" \
LDFLAGS="-L$CP/lib -Wl,-rpath,$CP/lib" HDF5_CPPFLAGS="-I$CP/include" HDF5_LIBS="-L$CP/lib -lhdf5" \
./configure --prefix=$CP --enable-single --with-python --with-libctl=$CP --with-hdf5 PYTHON=python CC=gcc CXX=g++

# Build piecewise (scheme dir will fail — that's okay)
make -j8
cd src && make install && cd ../libpympb && make -j8 && cd ../python && make -j8 && make install
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ctl.h: no such file` | Patch configure search path (see above) |
| `guile/gh.h: file not found` | Create stub gh.h (see above) |
| `gen-ctl-io: -l: command not found` | Skip scheme dir, build src + python only |
| `meep_wrap.cxx: need --with-maintainer-mode` | Use release tarball, not git clone |

## Quantum ESPRESSO

### From source (with apple-bottom acceleration)

```bash
brew install gcc openblas fftw open-mpi scalapack

cd ~/qe-test
wget https://github.com/QEF/q-e/releases/download/qe-7.4.1/qe-7.4.1-ReleasePack.tgz
tar xzf qe-7.4.1-ReleasePack.tgz
cd q-e-qe-7.4.1

./configure
# Edit make.inc to link against OpenBLAS:
#   BLAS_LIBS = -L/opt/homebrew/opt/openblas/lib -lopenblas
#   LAPACK_LIBS = -L/opt/homebrew/opt/openblas/lib -lopenblas

make -j8 pw
```

For apple-bottom DD-BLAS acceleration, see the apple-bottom repo's integration guide. The key steps are replacing ZGEMM calls in `cegterg.f90` with the GPU-accelerated variant and linking against `libapple_bottom.a`.

### Validation

```bash
cd ~/qe-test/q-e-qe-7.4.1
bin/pw.x < test-suite/pw_silicon/si_scf.in | grep "total energy"
# Should match known value for your pseudopotential
```

## Yambo

See docs/LESSONS_LEARNED_YAMBO.md for the full procedure. Summary:

```bash
# Pin Homebrew packages first
brew pin gcc openblas fftw hdf5 netcdf netcdf-fortran open-mpi scalapack libxc

cd ~/yambo-build
git clone https://github.com/yambo-code/yambo.git
cd yambo && git checkout 5.3.0

export HB=/opt/homebrew
./configure FC=gfortran CC=gcc \
  FCFLAGS="-O2 -fallow-argument-mismatch -ffree-line-length-none" \
  --enable-open-mp --disable-mpi \
  --with-blas-libs="-L${HB}/opt/openblas/lib -lopenblas" \
  --with-lapack-libs="-L${HB}/opt/openblas/lib -lopenblas" \
  --with-fft-path="${HB}" --with-hdf5-path="${HB}" \
  --with-netcdf-path="${HB}" --with-netcdff-path="${HB}/opt/netcdf-fortran" \
  --with-libxc-path="${HB}/opt/libxc"

make core -j8
make interfaces -j8
```

### QE → Yambo pipeline test

```bash
# After QE NSCF run:
cd tmp/prefix.save
~/yambo-build/yambo/bin/p2y
cp -r SAVE ~/yambo-build/test/
cd ~/yambo-build/test
~/yambo-build/yambo/bin/yambo   # generates default input
```

### Critical warnings

- `brew upgrade` can break Yambo's dylib linkage. Pin packages before building.
- MPI runs may hang on startup. Use `mpirun -np 1` or run serial for small systems.
- Set `OMPI_CC=/opt/homebrew/bin/gcc-15` and `OMPI_FC=/opt/homebrew/bin/gfortran-15`.

## Elmer FEM (Future)

Not yet integrated. Planned approach:

```bash
brew install elmer
# OR build from source:
# git clone https://github.com/ElmerCSC/elmerfem.git
# mkdir build && cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/elmer
# make -j8 && make install
```

Will need: heat equation solver, mesh import from MEEP grid, boundary conditions from device geometry.

## Environment Setup

After installing solvers, create the unified environment:

```bash
cd ~/Dev/silicon-forge  # or wherever you cloned muppet-rainbow-connection
source env.sh

# Verify all solvers are found
which pw.x    # QE
which yambo   # Yambo
conda activate meep_arm && python -c "import meep"  # MEEP
```

## Dependency Isolation Rules

| Solver | Dep Source | Why |
|--------|-----------|-----|
| QE | Homebrew (gcc, openblas, fftw, mpi) | Needs Fortran compiler, MPI |
| Yambo | Homebrew (same + hdf5, netcdf, libxc) | Shares compiler stack with QE |
| MEEP | Conda (meep_arm env) | Brings its own FFTW, HDF5, MPI — incompatible with Homebrew's |
| Elmer | Homebrew or source | TBD |

Never mix: don't link MEEP against Homebrew FFTW, don't link QE against conda's MPI. The env.sh script keeps these separated.
