#!/bin/bash
# 04_run_yambo_bse.sh - Run Yambo BSE for optical absorption spectrum
# Part of MUPPET: Rainbow Connection

set -e

# Load environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"
source ~/Dev/silicon-forge/env.sh

echo "========================================="
echo "MUPPET: Si Photovoltaic Workflow"
echo "Step 4: Yambo BSE Optical Spectrum"
echo "========================================="
echo ""

cd "$WORKFLOW_DIR/inputs"

# Check prerequisites
if [ ! -d "SAVE" ]; then
    echo "Error: Yambo database not found (SAVE/ missing)"
    echo "Run 02_convert_p2y.sh first"
    exit 1
fi

if [ ! -f "r_setup" ]; then
    echo "Error: Yambo not initialized. Run 03_run_yambo_gw.sh first"
    exit 1
fi

# First, compute screening (W) - required for BSE kernel
if [ ! -f "ndb.em1s" ] && [ ! -d "SAVE/ndb.em1s*" ]; then
    echo "Computing screened interaction (W) for BSE kernel..."
    echo "  This is required before BSE can run"
    echo ""

    # Generate screening input
    yambo -o c -V all > yambo_screening_template.log 2>&1
    mv yambo.in yambo_screening.in

    # Edit for quick calculation
    sed -i '' 's/%BndsRnXs/%BndsRnXs                        # [Xs] Polarization function bands/' yambo_screening.in
    sed -i '' 's/1 | 20 |/3 | 6 |/' yambo_screening.in

    echo "Running screening calculation..."
    yambo -F yambo_screening.in -J screening 2>&1 | tee yambo_screening.log

    # Copy screening database to SAVE
    if [ -d "screening/ndb.em1s" ]; then
        cp -r screening/ndb.em1s* SAVE/ 2>/dev/null || true
        echo "✓ Screening database created"
        echo ""
    else
        echo "✗ Screening calculation failed"
        exit 1
    fi
fi

# Generate BSE input file
if [ ! -f "yambo_bse.in" ]; then
    echo "Generating BSE input file..."
    yambo -o b -k sex -y d -V all > yambo_bse_template.log 2>&1
    mv yambo.in yambo_bse.in

    # Edit for quick test: restrict bands to valence-conduction region
    # For 8-electron Si: bands 1-4 valence, 5+ conduction
    # Use bands 3-6 for faster test
    sed -i '' 's/%BndsRnXs/%BndsRnXs                        # [Xs] Polarization function bands/' yambo_bse.in
    sed -i '' 's/1 | 20 |/3 | 6 |/' yambo_bse.in

    # Set reasonable BSE kernel cutoff
    sed -i '' 's/BSEBands= 1 | 20 |/BSEBands= 3 | 6 |/' yambo_bse.in || true

    # Energy range for optical spectrum (0-6 eV covers solar spectrum)
    sed -i '' 's/%EnRngeXs/%EnRngeXs                         # [Xs] Energy range/' yambo_bse.in
    sed -i '' 's/ 0.000000 | 10.00000 |/ 0.000000 | 6.000000 |/' yambo_bse.in

    echo "✓ BSE input created: yambo_bse.in (bands 3-6, 0-6 eV)"
    echo ""
fi

# Run BSE calculation
echo "Running BSE calculation for optical absorption..."
echo "  Input: yambo_bse.in"
echo "  Job name: si_bse"
echo "  Output: o-si_bse.eps_q1_diago_bse"
echo ""
echo "Note: BSE with bands 3-6 takes ~5-10 minutes"
echo ""

start_time=$(date +%s)
yambo -F yambo_bse.in -J si_bse 2>&1 | tee yambo_bse.log

bse_time=$(($(date +%s) - start_time))

# Check for output
if [ -f "o-si_bse.eps_q1_diago_bse" ]; then
    echo ""
    echo "✓ BSE calculation completed in ${bse_time}s"
    echo ""

    # Extract key features of the spectrum
    echo "Optical absorption spectrum computed:"
    npts=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | wc -l)
    echo "  Energy points: $npts"

    emin=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | head -1 | awk '{print $1}')
    emax=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | tail -1 | awk '{print $1}')
    echo "  Energy range: $emin - $emax eV"

    # Find absorption onset (first point where Im[eps] > 0.1)
    onset=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | awk '$2 > 0.1 {print $1; exit}')
    echo "  Absorption onset: ~$onset eV"

    # Find peak absorption
    peak_e=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | awk 'BEGIN{max=0} {if($2>max){max=$2; e=$1}} END{print e}')
    peak_val=$(grep -v "^#" o-si_bse.eps_q1_diago_bse | awk 'BEGIN{max=0} {if($2>max){max=$2}} END{print max}')
    echo "  Peak Im[ε]: $peak_val at $peak_e eV"

    # Copy to results directory
    mkdir -p "$WORKFLOW_DIR/results"
    cp o-si_bse.eps_q1_diago_bse "$WORKFLOW_DIR/results/"
    echo ""
    echo "Spectrum saved to: results/o-si_bse.eps_q1_diago_bse"

else
    echo "✗ BSE calculation failed. Check yambo_bse.log"
    tail -100 yambo_bse.log
    exit 1
fi

echo ""
echo "========================================="
echo "BSE calculation complete!"
echo "Optical spectrum ready for MEEP"
echo ""
echo "Next step: 05_fit_epsilon.py"
echo "========================================="
