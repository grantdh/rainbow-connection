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
    echo "  K-point  Band  E_DFT(eV)  E_GW-E_DFT(eV)"
    echo "  --------------------------------------------"
    awk '!/^#/ && NF>=4 {printf "  %3s      %3s     %8.3f      %+8.3f\n", $1, $2, $3, $4}' o-si_gw.qp

    # Calculate band gap (simple awk, no bc needed)
    echo ""
    echo "Band gap estimate:"
    awk '
    BEGIN {vbm_found=0; cbm_found=0}
    !/^#/ && NF>=4 {
        e_qp = $3 + $4
        if ($2 == 4 && !vbm_found) {vbm = e_qp; vbm_k=$1; vbm_found=1}
        if ($2 == 5 && !cbm_found) {cbm = e_qp; cbm_k=$1; cbm_found=1}
    }
    END {
        if (vbm_found && cbm_found) {
            gap = cbm - vbm
            printf "  VBM (K%s, band 4): %.3f eV\n", vbm_k, vbm
            printf "  CBM (K%s, band 5): %.3f eV\n", cbm_k, cbm
            printf "  GW gap: %.3f eV\n", gap
        } else {
            print "  Could not extract band gap"
        }
    }
    ' o-si_gw.qp
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
