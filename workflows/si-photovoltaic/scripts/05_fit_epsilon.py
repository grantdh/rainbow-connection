#!/usr/bin/env python3
"""
05_fit_epsilon.py - Fit Yambo BSE spectrum to Lorentzian model for MEEP
Part of MUPPET: Rainbow Connection

This script is a thin wrapper around the yambo_eps_to_meep.py converter.
It uses workflow-specific paths and saves the output for MEEP simulations.
"""

import os
import sys
import subprocess
import json

# Workflow directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.dirname(SCRIPT_DIR)
INPUTS_DIR = os.path.join(WORKFLOW_DIR, "inputs")
RESULTS_DIR = os.path.join(WORKFLOW_DIR, "results")
CONVERTER_DIR = os.path.expanduser("~/Dev/silicon-forge/lib/converters")

print("=" * 60)
print("MUPPET: Si Photovoltaic Workflow")
print("Step 5: Fit ε(ω) to Lorentzian Model")
print("=" * 60)
print()

# Check prerequisites
bse_file = os.path.join(RESULTS_DIR, "o-si_bse.eps_q1_diago_bse")
if not os.path.exists(bse_file):
    # Try inputs directory
    bse_file = os.path.join(INPUTS_DIR, "o-si_bse.eps_q1_diago_bse")
    if not os.path.exists(bse_file):
        print(f"Error: BSE spectrum not found")
        print(f"Expected: {RESULTS_DIR}/o-si_bse.eps_q1_diago_bse")
        print("Run 04_run_yambo_bse.sh first")
        sys.exit(1)

converter = os.path.join(CONVERTER_DIR, "yambo_eps_to_meep.py")
if not os.path.exists(converter):
    print(f"Error: Converter not found at {converter}")
    sys.exit(1)

print(f"Input: {bse_file}")
print(f"Converter: {converter}")
print()

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Run the converter
# Use 4 poles for better fit quality (3 poles sometimes underfit Si)
output_json = os.path.join(RESULTS_DIR, "si_bse_material.json")

cmd = [
    "python3",
    converter,
    bse_file,
    "--poles", "4",
    "--output", output_json,
    "--name", "si_bse",
]

print("Running converter with 4 Lorentzian poles...")
print(f"Command: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd=RESULTS_DIR)

if result.returncode != 0:
    print()
    print("✗ Conversion failed")
    sys.exit(1)

print()

# Verify output
if os.path.exists(output_json):
    with open(output_json) as f:
        mat_data = json.load(f)

    print("=" * 60)
    print("Fit complete!")
    print("=" * 60)
    print(f"Material saved: {output_json}")
    print(f"ε_∞ = {mat_data['eps_inf']:.4f}")
    print(f"Poles: {len(mat_data['poles'])}")
    print(f"RMS error: ε₁={mat_data['rms_eps1']:.4f}, ε₂={mat_data['rms_eps2']:.4f}")
    print()

    # Copy also to lib/converters for MEEP script compatibility
    lib_output = os.path.join(CONVERTER_DIR, "yambo_material.json")
    import shutil
    shutil.copy(output_json, lib_output)
    print(f"Also saved to: {lib_output} (for 06_run_meep.py)")
    print()

    print("=" * 60)
    print("Si ab initio material ready for MEEP!")
    print()
    print("Next step: ../scripts/06_run_meep.py")
    print("=" * 60)

else:
    print("✗ Output file not created")
    sys.exit(1)
