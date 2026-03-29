#!/usr/bin/env python3
"""
Compute the Quantum Geometric Tensor from QE/Wannier90 overlap matrices.
Grant Heileman, UNM ECE - March 2026
"""

import numpy as np
import sys
import json


def read_mmn(seedname):
    fname = seedname + '.mmn'
    print(f"Reading {fname}...")
    with open(fname) as f:
        header = f.readline().strip()
        nbnd, nkpt, nntot = map(int, f.readline().split())
        print(f"  {nbnd} bands, {nkpt} k-points, {nntot} neighbors/k-point")
        overlaps = np.zeros((nkpt, nntot, nbnd, nbnd), dtype=complex)
        neighbors = np.zeros((nkpt, nntot), dtype=int)
        G_vectors = np.zeros((nkpt, nntot, 3), dtype=int)
        for ik in range(nkpt):
            for inn in range(nntot):
                line = f.readline().split()
                ik2 = int(line[1]) - 1
                g1, g2, g3 = int(line[2]), int(line[3]), int(line[4])
                neighbors[ik, inn] = ik2
                G_vectors[ik, inn] = [g1, g2, g3]
                for n in range(nbnd):
                    for m in range(nbnd):
                        line = f.readline().split()
                        overlaps[ik, inn, m, n] = float(line[0]) + 1j * float(line[1])
    return {'nbnd': nbnd, 'nkpt': nkpt, 'nntot': nntot,
            'overlaps': overlaps, 'neighbors': neighbors, 'G_vectors': G_vectors}


def read_eig(seedname):
    fname = seedname + '.eig'
    data = np.loadtxt(fname)
    nbnd = int(data[:, 0].max())
    nkpt = int(data[:, 1].max())
    eig = np.zeros((nkpt, nbnd))
    for row in data:
        eig[int(row[1])-1, int(row[0])-1] = row[2]
    return eig


def read_nnkp(seedname):
    fname = seedname + '.nnkp'
    recip = np.zeros((3, 3))
    with open(fname) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'begin recip_lattice' in line:
            for j in range(3):
                recip[j] = [float(v) for v in lines[i+1+j].split()]
    return recip


def compute_quantum_metric(mmn_data, eig, nbnd_occ, recip_lattice):
    nkpt = mmn_data['nkpt']
    nntot = mmn_data['nntot']
    M = mmn_data['overlaps']

    T_kb = np.zeros((nkpt, nntot))
    for ik in range(nkpt):
        for ib in range(nntot):
            M_occ = M[ik, ib, :nbnd_occ, :nbnd_occ]
            T_kb[ik, ib] = np.sum(np.abs(M_occ)**2)

    one_minus_T = nbnd_occ - T_kb
    omega_I_raw = np.mean(one_minus_T)

    nk_per_dim = round(nkpt**(1.0/3.0))
    b1_len = np.linalg.norm(recip_lattice[0])
    bohr_to_ang = 0.529177
    dk_invA = (b1_len / nk_per_dim) / bohr_to_ang

    w_b = 3.0 / (nntot * dk_invA**2)
    omega_I = w_b * np.sum(one_minus_T) / nkpt

    return {'T_kb': T_kb, 'one_minus_T': one_minus_T,
            'omega_I': omega_I, 'omega_I_raw': omega_I_raw,
            'w_b': w_b, 'dk_invA': dk_invA, 'nk_per_dim': nk_per_dim}


def compute_optical_conductivity(mmn_data, eig, nbnd_occ, omega_array, broadening=0.15):
    nkpt = mmn_data['nkpt']
    nntot = mmn_data['nntot']
    M = mmn_data['overlaps']
    nbnd = mmn_data['nbnd']
    sigma = np.zeros_like(omega_array)

    for ik in range(nkpt):
        for ib in range(nntot):
            for n in range(nbnd_occ):
                for m in range(nbnd_occ, nbnd):
                    weight = np.abs(M[ik, ib, n, m])**2
                    dE = eig[ik, m] - eig[ik, n]
                    if dE < 0.01:
                        continue
                    spectral = (broadening / np.pi) / ((omega_array - dE)**2 + broadening**2)
                    sigma += weight * dE * spectral

    sigma /= (nkpt * nntot)
    return sigma


def main():
    if len(sys.argv) < 2:
        print("Usage: python qgt_from_dft.py /path/to/seedname [--nbnd_occ N]")
        sys.exit(1)

    seedname = sys.argv[1]
    nbnd_occ = 4
    for i, arg in enumerate(sys.argv):
        if arg == '--nbnd_occ' and i+1 < len(sys.argv):
            nbnd_occ = int(sys.argv[i+1])

    print("MUPPET QGT Module - DFT Quantum Metric Calculation")
    print("=" * 60)

    mmn_data = read_mmn(seedname)
    eig = read_eig(seedname)
    recip = read_nnkp(seedname)

    print(f"  Occupied bands: {nbnd_occ}")
    gap = eig[:, nbnd_occ].min() - eig[:, nbnd_occ-1].max()
    print(f"  Band gap: {gap:.4f} eV")
    print(f"  VBM: {eig[:, nbnd_occ-1].max():.4f} eV")
    print(f"  CBM: {eig[:, nbnd_occ].min():.4f} eV")
    print()

    print("Computing quantum metric from overlap matrices...")
    result = compute_quantum_metric(mmn_data, eig, nbnd_occ, recip)

    print(f"  Finite-difference dk = {result['dk_invA']:.4f} 1/A")
    print(f"  FD weight w_b = {result['w_b']:.4f} A^2")
    print(f"  <N_occ - T(k,b)> = {result['omega_I_raw']:.6f}")
    print(f"  Omega_I (gauge-invariant spread) = {result['omega_I']:.2f} A^2")
    print()

    print("Computing interband sigma(w) from quantum metric...")
    omega = np.linspace(0.01, 10.0, 500)
    sigma = compute_optical_conductivity(mmn_data, eig, nbnd_occ, omega, broadening=0.15)

    threshold = sigma.max() * 0.01 if sigma.max() > 0 else 1e-10
    onset_idx = np.argmax(sigma > threshold)
    peak_idx = np.argmax(sigma)

    print(f"  Absorption onset: {omega[onset_idx]:.2f} eV")
    print(f"  Peak sigma at: {omega[peak_idx]:.2f} eV")
    print(f"  Peak value: {sigma.max():.6f} (arb. units)")
    print()

    output = {
        'material': 'GaAs (DFT-PBE)',
        'nbnd_occ': nbnd_occ,
        'band_gap_eV': float(gap),
        'omega_I_angstrom2': float(result['omega_I']),
        'energy_eV': omega.tolist(),
        'sigma_inter': sigma.tolist(),
    }
    outfile = os.path.basename(seedname) + '_qgt.json'
    with open(outfile, 'w') as f:
        json.dump(output, f)
    print(f"Saved: {outfile}")
    print()
    print("=" * 60)
    print("This is the quantum metric of GaAs computed from DFT.")
    print("Next: add Drude (Queiroz), Kramers-Kronig -> full eps(w),")
    print("then feed into MEEP to reproduce your EMI shielding paper.")


if __name__ == "__main__":
    import os
    main()
