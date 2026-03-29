#!/bin/bash
# 01_run_qe.sh - Run QE SCF and NSCF calculations for Si
# Part of MUPPET: Rainbow Connection

set -e  # Exit on error

# Load environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"
source ~/Dev/silicon-forge/env.sh

echo "========================================="
echo "MUPPET: Si Photovoltaic Workflow"
echo "Step 1: QE DFT Calculations"
echo "========================================="
echo ""

cd "$WORKFLOW_DIR/inputs"

# Check prerequisites
if [ ! -f "si_scf.in" ] || [ ! -f "si_nscf.in" ]; then
    echo "Error: Input files not found in $WORKFLOW_DIR/inputs/"
    exit 1
fi

if [ ! -f "pseudo/Si.pbe-n-rrkjus_psl.1.0.0.UPF" ]; then
    echo "Error: Pseudopotential not found in pseudo/"
    exit 1
fi

# Create temporary directory
mkdir -p tmp

# Run SCF
echo "Running SCF calculation..."
echo "  Input: si_scf.in"
echo "  Output: si_scf.out"
echo ""

start_time=$(date +%s)
pw.x < si_scf.in > si_scf.out 2>&1

if grep -q "JOB DONE" si_scf.out; then
    scf_time=$(($(date +%s) - start_time))
    echo "✓ SCF completed successfully in ${scf_time}s"

    # Extract key results
    total_energy=$(grep "!" si_scf.out | tail -1 | awk '{print $5, $6}')
    echo "  Total energy: $total_energy"
else
    echo "✗ SCF failed. Check si_scf.out"
    exit 1
fi

echo ""

# Run NSCF
echo "Running NSCF calculation (with more bands for GW/BSE)..."
echo "  Input: si_nscf.in"
echo "  Output: si_nscf.out"
echo ""

start_time=$(date +%s)
pw.x < si_nscf.in > si_nscf.out 2>&1

if grep -q "JOB DONE" si_nscf.out; then
    nscf_time=$(($(date +%s) - start_time))
    echo "✓ NSCF completed successfully in ${nscf_time}s"

    # Extract band info
    nbnd=$(grep "number of Kohn-Sham states" si_nscf.out | tail -1 | awk '{print $5}')
    nks=$(grep "number of k points" si_nscf.out | tail -1 | awk '{print $5}')
    echo "  Bands: $nbnd"
    echo "  K-points: $nks"
else
    echo "✗ NSCF failed. Check si_nscf.out"
    exit 1
fi

echo ""
echo "========================================="
echo "QE calculations complete!"
echo "Total time: $((scf_time + nscf_time))s"
echo ""
echo "Next step: 02_convert_p2y.sh"
echo "========================================="
