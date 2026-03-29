#!/bin/bash
# 03_run_yambo_gw.sh - Run Yambo GW quasiparticle corrections
# Part of MUPPET: Rainbow Connection

set -e

# Load environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"
source ~/Dev/silicon-forge/env.sh

echo "========================================="
echo "MUPPET: Si Photovoltaic Workflow"
echo "Step 3: Yambo GW Calculation"
echo "========================================="
echo ""

cd "$WORKFLOW_DIR/inputs"

# Check prerequisites
if [ ! -d "SAVE" ]; then
    echo "Error: Yambo database not found (SAVE/ missing)"
    echo "Run 02_convert_p2y.sh first"
    exit 1
fi

# Initialize Yambo (required before generating GW input)
if [ ! -f "r_setup" ]; then
    echo "Initializing Yambo database..."
    yambo > yambo_init.log 2>&1
    echo "✓ Yambo initialized"
    echo ""
fi

# Generate GW input file
if [ ! -f "yambo_gw.in" ]; then
    echo "Generating GW input file..."
    yambo -d -g -p p -V all > yambo_gw_template.log 2>&1
    mv yambo.in yambo_gw.in

    # Edit for quick test: restrict to band gap (bands 4-5)
    sed -i '' 's/%QPkrange/%QPkrange                        # [GW] QP generalized Kpoint\/Band indices/' yambo_gw.in
    sed -i '' 's/1|10|1|20|/1|10|4|5|/' yambo_gw.in

    echo "✓ GW input created: yambo_gw.in (bands 4-5, all k-points)"
    echo ""
fi

# Run GW calculation
echo "Running GW calculation..."
echo "  Input: yambo_gw.in"
echo "  Job name: si_gw"
echo "  Output: o-si_gw.qp"
echo ""

start_time=$(date +%s)
yambo -F yambo_gw.in -J si_gw 2>&1 | tee yambo_gw.log

gw_time=$(($(date +%s) - start_time))

# Check for output
if [ -f "o-si_gw.qp" ]; then
    echo ""
    echo "✓ GW calculation completed in ${gw_time}s"
    echo ""

    # Extract band gap correction
    echo "GW quasiparticle corrections:"
    echo "  K-point  Band  E_DFT(eV)  E_GW-E_DFT(eV)  Z"
    echo "  -----------------------------------------------"
    grep -A 20 "# K-point" o-si_gw.qp | grep -v "^#" | awk '{if(NF>3) print "  " $1 "        " $2 "     " $3 "      " $4 "         " $7}'

    # Calculate band gap
    vbm=$(grep -A 20 "# K-point" o-si_gw.qp | grep -v "^#" | awk '$2==4 {print $3+$4}' | head -1)
    cbm=$(grep -A 20 "# K-point" o-si_gw.qp | grep -v "^#" | awk '$2==5 {print $3+$4}' | tail -1)

    echo ""
    echo "Estimated GW band gap: $(echo "$cbm - $vbm" | bc -l | xargs printf "%.3f") eV"
    echo ""
    echo "Note: This is NOT converged! Full convergence requires:"
    echo "  - 8×8×8 k-grid"
    echo "  - Larger NGsBlkXp (4-8 Ry)"
    echo "  - More bands (100+)"

else
    echo "✗ GW calculation failed. Check yambo_gw.log"
    tail -50 yambo_gw.log
    exit 1
fi

echo ""
echo "========================================="
echo "GW calculation complete!"
echo "Results saved in: si_gw/ directory"
echo ""
echo "Next step: 04_run_yambo_bse.sh"
echo "========================================="
