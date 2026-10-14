#!/bin/bash
# 02_convert_p2y.sh - Convert QE wavefunctions to Yambo database
# Part of MUPPET: Rainbow Connection

set -e

# Load environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"
source ~/Dev/Claude/rainbow-connection/env.sh

echo "========================================="
echo "MUPPET: Si Photovoltaic Workflow"
echo "Step 2: QE → Yambo Conversion (p2y)"
echo "========================================="
echo ""

cd "$WORKFLOW_DIR/inputs"

# Check prerequisites
if [ ! -d "tmp/si.save" ]; then
    echo "Error: QE wavefunctions not found (tmp/si.save missing)"
    echo "Run 01_run_qe.sh first"
    exit 1
fi

# Run p2y from within the QE save directory
echo "Converting QE wavefunctions to Yambo format..."
echo "  Input: tmp/si.save/"
echo "  Output: tmp/si.save/SAVE/"
echo ""

cd tmp/si.save

start_time=$(date +%s)
p2y 2>&1 | tee p2y.log

p2y_time=$(($(date +%s) - start_time))

# Check if SAVE directory was created
if [ -d "SAVE" ]; then
    echo "✓ p2y conversion completed in ${p2y_time}s"
    echo ""

    # List key database files
    echo "Yambo database files created:"
    ls -lh SAVE/ns.db1 SAVE/ns.wf 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'

    # Count wavefunction fragments
    nwf=$(ls SAVE/ns.wf_fragments_* 2>/dev/null | wc -l)
    echo "  Wavefunction fragments: $nwf"

    # Move SAVE to workflow directory for easy access
    echo ""
    echo "Copying SAVE to $WORKFLOW_DIR/inputs/SAVE for Yambo runs..."
    cp -r SAVE "$WORKFLOW_DIR/inputs/"

else
    echo "✗ p2y conversion failed. Check p2y.log"
    cat p2y.log
    exit 1
fi

cd "$WORKFLOW_DIR"

echo ""
echo "========================================="
echo "Conversion complete!"
echo "Yambo database ready at: inputs/SAVE/"
echo ""
echo "Next step: 03_run_yambo_gw.sh"
echo "========================================="
