#!/usr/bin/env python3
"""
yambo_eps_to_meep.py — Convert Yambo BSE dielectric function to MEEP material

Reads Yambo's ε(ω) output (o-*.eps_q1_diago_bse) and fits it to a sum of
Lorentzian oscillators compatible with MEEP's LorentzianSusceptibility model.

MEEP's dispersive material model:
  ε(ω) = ε_∞ + Σ_n  σ_n × ω_n² / (ω_n² - ω² - i×γ_n×ω)

where:
  ε_∞  = high-frequency dielectric constant
  ω_n  = resonance frequency
  γ_n  = damping rate (linewidth)
  σ_n  = oscillator strength (dimensionless weight)

Usage:
  python yambo_eps_to_meep.py /path/to/o-*.eps_q1_diago_bse [--poles N] [--plot]

Output:
  Prints MEEP Python code defining the material.
  Writes fitted parameters to yambo_material.json.

Grant Heileman, UNM ECE — March 2026
Part of silicon-forge: multiphysics on Apple Silicon
"""

import numpy as np
import json
import sys
import os

def read_yambo_eps(filename):
    """Read Yambo's dielectric function output file."""
    energies = []
    eps2_bse = []  # Im[eps] with BSE
    eps1_bse = []  # Re[eps] with BSE
    eps2_ip = []   # Im[eps] independent particle
    eps1_ip = []   # Re[eps] independent particle

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) >= 5:
                energies.append(float(vals[0]))
                eps2_bse.append(float(vals[1]))
                eps1_bse.append(float(vals[2]))
                eps2_ip.append(float(vals[3]))
                eps1_ip.append(float(vals[4]))

    return {
        'energy_eV': np.array(energies),
        'eps2_bse': np.array(eps2_bse),
        'eps1_bse': np.array(eps1_bse),
        'eps2_ip': np.array(eps2_ip),
        'eps1_ip': np.array(eps1_ip),
    }


def lorentzian_eps(omega_eV, eps_inf, poles):
    """Compute ε(ω) from Lorentzian pole parameters.
    
    poles: list of (omega_n, gamma_n, sigma_n) in eV
    Returns complex ε(ω)
    """
    eps = np.full_like(omega_eV, eps_inf, dtype=complex)
    for omega_n, gamma_n, sigma_n in poles:
        eps += sigma_n * omega_n**2 / (omega_n**2 - omega_eV**2 - 1j * gamma_n * omega_eV)
    return eps


def fit_lorentzian(energy_eV, eps1, eps2, n_poles=3, eps_inf_guess=1.0):
    """Fit ε(ω) = ε₁ + iε₂ to a sum of Lorentzian oscillators.
    
    Uses least-squares on both real and imaginary parts simultaneously.
    """
    from scipy.optimize import least_squares

    # Build target complex epsilon
    eps_target = eps1 + 1j * eps2

    # Initial guesses: distribute poles across the energy range
    e_min = energy_eV[energy_eV > 0.1].min()
    e_max = energy_eV.max()
    
    # Find peaks in Im[eps] for initial pole positions
    peaks = []
    for i in range(1, len(eps2) - 1):
        if eps2[i] > eps2[i-1] and eps2[i] > eps2[i+1] and eps2[i] > 0.5:
            peaks.append(energy_eV[i])
    
    # If not enough peaks found, distribute evenly
    while len(peaks) < n_poles:
        peaks.append(e_min + (e_max - e_min) * (len(peaks) + 1) / (n_poles + 1))
    peaks = sorted(peaks[:n_poles])

    # Parameter vector: [eps_inf, omega_1, gamma_1, sigma_1, omega_2, ...]
    x0 = [eps_inf_guess]
    for i, peak in enumerate(peaks):
        x0.extend([
            peak,           # omega_n (eV)
            0.5,            # gamma_n (eV) — moderate damping
            2.0,            # sigma_n — oscillator strength
        ])
    x0 = np.array(x0)

    # Bounds: all physical (positive frequencies, damping, strengths)
    lb = [1.0]  # eps_inf >= 1.0 (vacuum is the physical floor)
    ub = [20.0]
    for _ in range(n_poles):
        lb.extend([0.01, 0.01, 0.001])   # omega, gamma, sigma > 0
        ub.extend([e_max * 1.5, 10.0, 50.0])
    lb = np.array(lb)
    ub = np.array(ub)

    def residual(x):
        eps_inf = x[0]
        poles = []
        for i in range(n_poles):
            idx = 1 + i * 3
            poles.append((x[idx], x[idx+1], x[idx+2]))
        
        eps_model = lorentzian_eps(energy_eV, eps_inf, poles)
        
        # Residual on both real and imaginary parts
        r_real = (eps_model.real - eps1) / (np.abs(eps1).max() + 1e-10)
        r_imag = (eps_model.imag - eps2) / (np.abs(eps2).max() + 1e-10)
        return np.concatenate([r_real, r_imag])

    result = least_squares(residual, x0, bounds=(lb, ub), 
                          method='trf', max_nfev=10000)

    # Extract fitted parameters
    eps_inf = result.x[0]
    poles = []
    for i in range(n_poles):
        idx = 1 + i * 3
        poles.append({
            'omega_eV': result.x[idx],
            'gamma_eV': result.x[idx + 1],
            'sigma': result.x[idx + 2],
        })

    return eps_inf, poles, result


def eV_to_meep_freq(eV):
    """Convert eV to MEEP frequency units (c/a where a=1μm).
    
    MEEP uses units where c=1, and if a=1μm:
      f_meep = E_eV / (hc in eV·μm) = E_eV / 1.23984
    """
    hc_eV_um = 1.23984198  # eV × μm
    return eV / hc_eV_um


def generate_meep_code(eps_inf, poles, material_name="yambo_material"):
    """Generate MEEP Python code for the fitted material."""
    
    lines = [
        f"import meep as mp",
        f"",
        f"# Material fitted from Yambo BSE: {material_name}",
        f"# ε(ω) = ε_∞ + Σ σ_n ω_n² / (ω_n² - ω² - iγ_n ω)",
        f"#",
        f"# ε_∞ = {eps_inf:.4f}",
        f"#",
        f"susceptibilities = [",
    ]
    
    for i, pole in enumerate(poles):
        omega_meep = eV_to_meep_freq(pole['omega_eV'])
        gamma_meep = eV_to_meep_freq(pole['gamma_eV'])
        sigma = pole['sigma']
        lines.append(
            f"    mp.LorentzianSusceptibility("
            f"frequency={omega_meep:.6f}, "
            f"gamma={gamma_meep:.6f}, "
            f"sigma={sigma:.6f}"
            f"),  # pole {i+1}: ω={pole['omega_eV']:.3f} eV, γ={pole['gamma_eV']:.3f} eV"
        )
    
    lines.extend([
        f"]",
        f"",
        f"{material_name} = mp.Medium(",
        f"    epsilon={eps_inf:.4f},",
        f"    E_susceptibilities=susceptibilities,",
        f")",
        f"",
        f"# Usage in simulation:",
        f"# geometry = [mp.Block(material={material_name}, ...)]",
    ])
    
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert Yambo BSE ε(ω) to MEEP Lorentzian material")
    parser.add_argument("input", help="Yambo epsilon file (o-*.eps_q1_diago_bse)")
    parser.add_argument("--poles", type=int, default=3, help="Number of Lorentzian poles (default: 3)")
    parser.add_argument("--use-ip", action="store_true", help="Use IP spectrum instead of BSE")
    parser.add_argument("--plot", action="store_true", help="Show comparison plot")
    parser.add_argument("--output", default=None, help="Output JSON file (default: yambo_material.json)")
    parser.add_argument("--name", default="yambo_si", help="Material variable name in MEEP code")
    args = parser.parse_args()

    # Read Yambo data
    print(f"Reading: {args.input}")
    data = read_yambo_eps(args.input)
    
    energy = data['energy_eV']
    if args.use_ip:
        eps1, eps2 = data['eps1_ip'], data['eps2_ip']
        print("Using independent-particle (IP) spectrum")
    else:
        eps1, eps2 = data['eps1_bse'], data['eps2_bse']
        print("Using BSE spectrum (includes excitonic effects)")
    
    print(f"  Energy range: {energy.min():.2f} - {energy.max():.2f} eV ({len(energy)} points)")
    print(f"  ε₁(0) = {eps1[0]:.2f} (static dielectric constant)")
    print(f"  ε₂ peak = {eps2.max():.2f} at {energy[np.argmax(eps2)]:.2f} eV")

    # Fit
    print(f"\nFitting {args.poles} Lorentzian poles...")
    eps_inf, poles, result = fit_lorentzian(energy, eps1, eps2, n_poles=args.poles)
    
    print(f"  ε_∞ = {eps_inf:.4f}")
    for i, p in enumerate(poles):
        print(f"  Pole {i+1}: ω = {p['omega_eV']:.3f} eV, "
              f"γ = {p['gamma_eV']:.3f} eV, σ = {p['sigma']:.4f}")
    print(f"  Fit residual: {result.cost:.6f}")

    # Evaluate fit quality
    eps_fit = lorentzian_eps(energy, eps_inf, 
                            [(p['omega_eV'], p['gamma_eV'], p['sigma']) for p in poles])
    rms_eps1 = np.sqrt(np.mean((eps_fit.real - eps1)**2))
    rms_eps2 = np.sqrt(np.mean((eps_fit.imag - eps2)**2))
    print(f"  RMS error: ε₁ = {rms_eps1:.4f}, ε₂ = {rms_eps2:.4f}")

    # Generate MEEP code
    meep_code = generate_meep_code(eps_inf, poles, material_name=args.name)
    print(f"\n{'='*60}")
    print("MEEP material definition:")
    print('='*60)
    print(meep_code)

    # Save JSON
    output_json = args.output or "yambo_material.json"
    result_dict = {
        'source_file': os.path.basename(args.input),
        'spectrum_type': 'IP' if args.use_ip else 'BSE',
        'eps_inf': float(eps_inf),
        'poles': poles,
        'fit_residual': float(result.cost),
        'rms_eps1': float(rms_eps1),
        'rms_eps2': float(rms_eps2),
        'energy_range_eV': [float(energy.min()), float(energy.max())],
        'meep_code': meep_code,
    }
    with open(output_json, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved: {output_json}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            ax1.plot(energy, eps1, 'b-', label='Yambo BSE', linewidth=2)
            ax1.plot(energy, eps_fit.real, 'r--', label=f'Lorentzian fit ({args.poles} poles)')
            ax1.set_ylabel('Re[ε(ω)]')
            ax1.legend()
            ax1.set_title('Yambo → MEEP material fitting')
            
            ax2.plot(energy, eps2, 'b-', label='Yambo BSE', linewidth=2)
            ax2.plot(energy, eps_fit.imag, 'r--', label=f'Lorentzian fit ({args.poles} poles)')
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('Im[ε(ω)]')
            ax2.legend()
            
            plt.tight_layout()
            plotfile = output_json.replace('.json', '.png')
            plt.savefig(plotfile, dpi=150)
            print(f"Saved plot: {plotfile}")
            plt.show()
        except ImportError:
            print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
