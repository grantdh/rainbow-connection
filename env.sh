#!/bin/bash
# silicon-forge multiphysics environment
# Usage: bash → source ~/silicon-forge/env.sh → work → exit

export SF_ROOT="$HOME/Dev/silicon-forge"

# Solver locations (existing installs — DO NOT MODIFY)
export SF_QE_ROOT="$HOME/qe-test/q-e-qe-7.4.1"
export SF_QE_BIN="$SF_QE_ROOT/bin"
export SF_YAMBO_ROOT="$HOME/yambo-build/yambo"
export SF_YAMBO_BIN="$SF_YAMBO_ROOT/bin"
export SF_MEEP_ENV="meep_arm"                    # conda env name
export SF_MEEP_SINGLE_ENV="meep-single"          # FP32 conda env
export SF_APPLE_BOTTOM="$HOME/apple-bottom"      # DD-BLAS library
export SF_ACME_MEEP="$HOME/acme-meep"            # Metal FDTD kernels

# MPI compiler fix (from Yambo lesson)
export OMPI_CC=/opt/homebrew/bin/gcc-15
export OMPI_FC=/opt/homebrew/bin/gfortran-15

# Convenience: add solver bins to PATH
export PATH="$SF_QE_BIN:$SF_YAMBO_BIN:$SF_ROOT/bin:$PATH"

# Shared pseudopotentials
export PSEUDO_DIR="$SF_ROOT/shared/pseudo"

echo "silicon-forge environment loaded."
echo "  QE:     $SF_QE_BIN/pw.x"
echo "  Yambo:  $SF_YAMBO_BIN/yambo"
echo "  MEEP:   conda activate $SF_MEEP_ENV"
echo "  Metal:  $SF_ACME_MEEP/src/metal/"
echo ""
echo "  Workflows: ls $SF_ROOT/workflows/"
