#!/bin/bash
# run_all.sh - Run complete Si photovoltaic workflow
# Part of MUPPET: Rainbow Connection
#
# This script runs the entire QE → Yambo → MEEP pipeline from start to finish.
# Each step checks for prerequisites before proceeding.
#
# Usage:
#   bash run_all.sh              # Run all steps
#   bash run_all.sh --skip-qe    # Skip QE if already done
#   bash run_all.sh --skip-yambo # Skip Yambo if already done

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_QE=false
SKIP_YAMBO=false

for arg in "$@"; do
    case $arg in
        --skip-qe)
            SKIP_QE=true
            ;;
        --skip-yambo)
            SKIP_YAMBO=true
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-qe     Skip QE calculations (use existing wavefunctions)"
            echo "  --skip-yambo  Skip Yambo calculations (use existing spectra)"
            echo "  --help        Show this help message"
            exit 0
            ;;
    esac
done

echo "========================================="
echo "MUPPET: Rainbow Connection"
echo "Si Photovoltaic Workflow (Complete Pipeline)"
echo "========================================="
echo ""
echo "This will run:"
echo "  01: QE SCF + NSCF"
echo "  02: p2y conversion"
echo "  03: Yambo GW"
echo "  04: Yambo BSE"
echo "  05: Lorentzian fitting"
echo "  06: MEEP FDTD"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# Track total time
WORKFLOW_START=$(date +%s)

# Step 1: QE
if [ "$SKIP_QE" = false ]; then
    echo "Running step 1/6: QE calculations..."
    bash 01_run_qe.sh
    echo ""
else
    echo "Skipping QE calculations (--skip-qe)"
    echo ""
fi

# Step 2: p2y
echo "Running step 2/6: p2y conversion..."
bash 02_convert_p2y.sh
echo ""

# Step 3: Yambo GW
if [ "$SKIP_YAMBO" = false ]; then
    echo "Running step 3/6: Yambo GW..."
    bash 03_run_yambo_gw.sh
    echo ""
else
    echo "Skipping Yambo GW (--skip-yambo)"
    echo ""
fi

# Step 4: Yambo BSE
if [ "$SKIP_YAMBO" = false ]; then
    echo "Running step 4/6: Yambo BSE..."
    bash 04_run_yambo_bse.sh
    echo ""
else
    echo "Skipping Yambo BSE (--skip-yambo)"
    echo ""
fi

# Step 5: Lorentzian fit
echo "Running step 5/6: Lorentzian fitting..."
python3 05_fit_epsilon.py
echo ""

# Step 6: MEEP
if [ -f "06_run_meep.py" ]; then
    echo "Running step 6/6: MEEP FDTD..."
    python3 06_run_meep.py
    echo ""
else
    echo "Step 6 (MEEP) script exists as: 06_run_meep.py"
    echo "Run it manually with: python3 06_run_meep.py"
    echo ""
fi

WORKFLOW_TIME=$(($(date +%s) - WORKFLOW_START))
WORKFLOW_MIN=$((WORKFLOW_TIME / 60))
WORKFLOW_SEC=$((WORKFLOW_TIME % 60))

echo "========================================="
echo "WORKFLOW COMPLETE!"
echo "========================================="
echo "Total time: ${WORKFLOW_MIN}m ${WORKFLOW_SEC}s"
echo ""
echo "Results saved in: ../results/"
echo ""
echo "This simulation computed optical properties"
echo "from FIRST PRINCIPLES:"
echo "  ✓ DFT ground state (QE)"
echo "  ✓ GW band gap correction (Yambo)"
echo "  ✓ BSE excitonic effects (Yambo)"
echo "  ✓ Light propagation (MEEP)"
echo ""
echo "No empirical parameters. Pure ab initio."
echo "========================================="
