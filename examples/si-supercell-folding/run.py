#!/usr/bin/env python
"""Example: Bloch-theorem identities of silicon supercells, with apple-bottom parity and dispatch evidence.

An unrelaxed 2x2x2 supercell of an 8-atom cubic silicon cell, sampled at Gamma, is the same calculation as
the 8-atom cell on its Gamma-centred 2x2x2 mesh: same density, same total energy per atom, and the Gamma
eigenvalues of the supercell are the union of the parent's eigenvalues over the mesh (band folding). With
commensurate FFT grids (72^3 = 2 x 36^3) the identity holds to the printed precision of pw.x, so it is a
sharp test of the whole SCF, of the Gamma-trick (real) Davidson against the complex one, and of the
apple-bottom GEMM paths that the two Davidsons exercise (regterg/DGEMM on the Gamma cell, cegterg/ZGEMM
on the k-point cell).

  reference /usr/local/bin/pw.x (QE 7.4.1, OpenBLAS, nm-verified apple-bottom-free), mpirun -np 4:
    R1  2-atom fcc primitive, explicit 32-point tpiba list (the 2x2x2 conventional mesh unfolded onto the bcc reciprocal lattice)
    R2  8-atom cubic cell, Gamma-centred 2x2x2 mesh, FFT 36^3 pinned
    R3  8-atom cubic cell, 4x4x4 mesh (-nk 4)
    R4  64-atom 2x2x2 supercell, K_POINTS gamma (Gamma trick: regterg, DGEMM)
    R5  64-atom, automatic 1 1 1 0 0 0 (same physics on the complex cegterg/ZGEMM path)
    R6  64-atom, 2x2x2 mesh (-nk 4; = the apple-bottom campaign's input B)
    R1', R2'  the two small cells again at ecutrho 200 Ry (R2' FFT 48^3) for the primitive-vs-cubic grid demonstration
  frozen apple-bottom pw.x (~/qe-test/ab-frozen/33fbaa8c8944, apple-bottom 1.4.0, QE develop 7.5 -O3), mpirun -np 1:
    R4 and R6 in AB_MODE=cpu and in AB_MODE=gpu AB_MIN_GPU_DIM=0 with the library's profiler on
    (--auto-leg adds R4 in AB_MODE=auto; --trace adds R4 at the library floor on 3ae144a9cadd with AB_TRACE)

Every launch goes through tools/abrun (runner.run_sync): timing lock honoured, MUPPET_SHARE semantics by
default (lock=share, nice 10, load recorded), exlib-format line in examples/common/solver-runs.log. All
walls are untimed observations (other sessions run on this machine); no speedup is computed anywhere.

Gates (exlib.Checks; gated rows PASS/FAIL, reported rows REPORTED and never counted): exact fold of the
density-dependent energy per atom (3e-9 Ry: two printed ulps per cell on both cells; the total-energy fold is
reported because ewald.f90 truncates its real-space sum at rmax = 4/sqrt(alpha), which is below the
nearest-neighbour distance on the 64-atom cells), the Ewald term against a replica of ewald.f90 and against a
converged lattice sum with the dropped real-space shell and G tail summed explicitly (tolerances 5e-9/nat plus
the replica's floor), per-term identities at an empirical 1e-6 (first-order terms; their sum is the 3e-9
gate), eigenvalue unfolding below E* and below the last computed parent band (1 meV), degeneracy counts,
HOMO/LUMO, Gamma trick vs complex algorithm (2e-8 Ry), symmetry of the ideal lattice (Td-site forces exactly
zero, other residuals below QE's own SCF-correction estimate, symmetrized stress, symmetry-op and k counts),
recorded campaign energies, apple-bottom parity (same pw.x: 1e-8 Ry; cross-build: 1e-6 Ry), dispatch evidence
from the library's own counters, and the profiler's GEMM dimensions pinned to QE's own printed PWs and
G-vector counts.

Run:  ~/miniconda3_arm64/envs/meep_arm/bin/python run.py [--ranks 4] [--lock share] [--fast] [--auto-leg] [--trace]
      (27 min measured on 2026-10-16 with another session's campaign running, lock share, load1 3-10; not measured
      on a quiet machine. --fast skips R5 and the R6 gpu leg; --reuse keeps finished legs in results/ whose input,
      mode, binary id and sha256 (dylib sha256 for frozen ids), pseudopotential sha256, ranks and nk match;
      --reuse --no-launch re-evaluates the gates from the run records and refuses to launch anything)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import exlib  # noqa: E402
from abrun import binaries, compare, launch, parse, presets, runner  # noqa: E402

RES = HERE / "results"
FROZEN_DEFAULT = "33fbaa8c8944"                     # apple-bottom 1.4.0 c5a1fa6, -O3 QE develop, OMP=1 only
TRACE_BINARY = "3ae144a9cadd"                       # adds AB_TRACE; --trace audit only
AB_BUILD_EXPECTED = "apple-bottom 1.4.0 (c5a1fa6a8166)"
REF_ID = "ref"                                      # /usr/local/bin/pw.x in abrun's catalog
AB_FROZEN_ROOT = Path.home() / "qe-test/ab-frozen"
SI64_INPUTS = Path(os.environ.get("MUPPET_SI64_INPUTS", str(Path.home() / "qe-test/xp/inputs")))
UPF_NAME = "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
UPF_SHA256 = "8f630aadb55097e3b456a4a7c49b0aabafb5c4b8ef586ff15fe247f98361d39e"
CAMPAIGN_C = SI64_INPUTS / "C_si64_160b_gamma.in"
# Recorded energies at conv_thr 1e-6 (the campaign inputs C = R4 and B = R6). Per cell, every recorded value and its build:
#   R4 (C_si64_160b_gamma): QE 7.4.1 (~/qe-test/baseline/si64.out, 22Mar2026, 8 iterations) and two builds of one QE develop commit
#      (5b7f8e1bc + the cegterg/regterg patches): -O0 (apple-bottom benchmarks/results/2026-10-14-1dbdb52-qe-si64/tables.md, 1 rank x 8
#      threads) and -O3 (2026-10-15-c5a1fa6-qe-si64-opt/tables.md: 1.4.0-np1 at 1 rank, 1.4.0-np8 at 8 ranks).
#   R6 (B_si64_160b_k222): the two develop builds only (-O0 2026-10-14-1dbdb52; -O3 2026-10-15-c5a1fa6 at np1 and np8nk4); no 7.4.1 value.
RECORDED = {"R4": -2989.94060060, "R6": -2990.44093535}             # the gated value: the 1-rank -O3 row (equal to 7.4.1 / -O0 where those exist)
RECORDED_ALL = {"R4": {"QE 7.4.1 si64.out 2026-03-22": -2989.94060060, "develop 5b7f8e1bc -O0 (2026-10-14-1dbdb52, np1 x 8 threads)": -2989.94060060,
                       "develop 5b7f8e1bc -O3 (2026-10-15-c5a1fa6, 1.4.0-np1)": -2989.94060060, "develop 5b7f8e1bc -O3 (2026-10-15-c5a1fa6, 1.4.0-np8)": -2989.94060129},
                "R6": {"develop 5b7f8e1bc -O0 (2026-10-14-1dbdb52, np1 x 8 threads)": -2990.44093535,
                       "develop 5b7f8e1bc -O3 (2026-10-15-c5a1fa6, 1.4.0-np1)": -2990.44093535, "develop 5b7f8e1bc -O3 (2026-10-15-c5a1fa6, 1.4.0-np8nk4)": -2990.44093534}}
RECORDED_SOURCE = {"R4": "QE 7.4.1 (~/qe-test/baseline/si64.out, 2026-03-22) and two builds (-O0, -O3) of QE develop 5b7f8e1bc at 1 and 8 ranks "
                         "(apple-bottom benchmarks/results/2026-10-14-1dbdb52-qe-si64 and 2026-10-15-c5a1fa6-qe-si64-opt tables.md)",
                   "R6": "the two develop 5b7f8e1bc builds only (-O0 2026-10-14-1dbdb52 at 1 rank; -O3 2026-10-15-c5a1fa6 at 1 and 8 ranks); no QE 7.4.1 value exists"}
REPLICA_FLOOR = 1e-11                               # Ry/atom: the Ewald replica's own float64 and truncation floor (b1's alpha spread 3.6e-15; the
                                                    # 64-atom closure residual 9e-13; see ewald_controls)
FOLD_TOL = 2 * 1e-8 / 8 + 2 * 1e-8 / 64             # 2.8e-9 Ry/atom: one printed ulp (1e-8 Ry per cell) on each of the two printed values (total,
FOLD_GATE = 3e-9                                    # Ewald) of both cells; gated at 3e-9. The SCF residual (conv_thr 1e-10 / 1e-9 per cell) adds < 2e-11
A_CONV = 10.2                                       # bohr, conventional cube (the campaign's celldm 20.4 / 2)
ECUT = 25.0
ECUTRHO = 100.0
NBND_SMALL, NBND_64 = 24, 160
NBND_CONTROL = 168                                  # R6 control: completes the near-degenerate group that nbnd = 160 cuts at k = (0,-1/2,-1/2)
UPSTREAM_BIN = Path(os.environ.get("MUPPET_QE_DEVELOP_UPSTREAM", str(Path.home() / "qe-test/q-e-develop-20261018/bin-variants/pw.x.upstream")))
UPSTREAM_DESC = "QE develop 4a626b7ce (v.8.0dev), upstream OpenBLAS, built from one fresh checkout with the mpi-gpu-ab-o3 configuration (bin-variants/PROVENANCE.txt)"
CONV_SMALL, CONV_64 = "1.0d-10", "1.0d-9"
EV_TOL = 1e-3                                       # eV, unfolding
PRINT_EV = 1e-4                                     # eV, printed precision of pw.x eigenvalues
UNFOLD_MARGIN = 2e-3                                # eV below E*: a 1 meV discrepancy cannot move a state across the cut
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]         # fixed categorical order: reference, apple-bottom, parent/other

CONV8 = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0), (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]


# --------------------------------------------------------------------------- inputs

def positions64():
    return sorted((x / 2 + i / 2, y / 2 + j / 2, z / 2 + k / 2) for (x, y, z) in CONV8 for i, j, k in product((0, 1), repeat=3))


def tpiba32():
    """The conventional cell's Gamma-centred 2x2x2 mesh unfolded onto the fcc primitive's bcc reciprocal lattice:
    (m/2 + c) for m in {0,1}^3 and c in the four cosets of the sc reciprocal lattice modulo the bcc one."""
    pts = []
    for m in product((0, 1), repeat=3):
        for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            pts.append(tuple(mi / 2 + ci for mi, ci in zip(m, c)))
    assert len(set(pts)) == 32
    return pts


def pw_input(prefix, ibrav, celldm, positions, nbnd, kpoints, conv_thr, ecutrho=ECUTRHO, nr=None):
    sysx = ""
    if nr:
        sysx = f"\n  nr1 = {nr}, nr2 = {nr}, nr3 = {nr}\n  nr1s = {nr}, nr2s = {nr}, nr3s = {nr}"
    pos = "\n".join(f"  Si {x:.6f} {y:.6f} {z:.6f}" for x, y, z in positions)
    return f"""&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  verbosity = 'high'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = {ibrav}
  celldm(1) = {celldm}
  nat = {len(positions)}
  ntyp = 1
  ecutwfc = {ECUT}
  ecutrho = {ecutrho}
  nbnd = {nbnd}
  occupations = 'fixed'
  nosym = .false.{sysx}
/
&ELECTRONS
  conv_thr = {conv_thr}
  mixing_beta = 0.5
  electron_maxstep = 200
  diago_full_acc = .true.
  diagonalization = 'david'
/
ATOMIC_SPECIES
  Si 28.086 {UPF_NAME}
ATOMIC_POSITIONS crystal
{pos}
{kpoints}
"""


def k_auto(n):
    return f"K_POINTS automatic\n  {n} {n} {n} 0 0 0\n"


def k_tpiba(pts):
    return "K_POINTS tpiba\n  %d\n" % len(pts) + "".join(f"  {x:.6f} {y:.6f} {z:.6f} 1.0\n" for x, y, z in pts)


def all_inputs():
    p2 = [(0, 0, 0), (.25, .25, .25)]
    p64 = positions64()
    return {
        "R1": pw_input("si2", 2, A_CONV, p2, NBND_SMALL, k_tpiba(tpiba32()), CONV_SMALL),
        "R1p": pw_input("si2", 2, A_CONV, p2, NBND_SMALL, k_tpiba(tpiba32()), CONV_SMALL, ecutrho=200.0),
        "R2": pw_input("si8", 1, A_CONV, CONV8, NBND_SMALL, k_auto(2), CONV_SMALL, nr=36),
        "R2p": pw_input("si8", 1, A_CONV, CONV8, NBND_SMALL, k_auto(2), CONV_SMALL, ecutrho=200.0, nr=48),
        "R3": pw_input("si8", 1, A_CONV, CONV8, NBND_SMALL, k_auto(4), CONV_SMALL, nr=36),
        "R4": pw_input("si64g", 1, 2 * A_CONV, p64, NBND_64, "K_POINTS gamma\n", CONV_64, nr=72),
        "R5": pw_input("si64c", 1, 2 * A_CONV, p64, NBND_64, k_auto(1), CONV_64, nr=72),
        "R6": pw_input("si64k", 1, 2 * A_CONV, p64, NBND_64, k_auto(2), CONV_64, nr=72),
    }


DESC = {"R1": "2-atom fcc, 32 explicit tpiba points", "R1p": "R1 at ecutrho 200", "R2": "8-atom cubic, 2x2x2, FFT 36^3",
        "R2p": "R2 at ecutrho 200, FFT 48^3", "R3": "8-atom cubic, 4x4x4, FFT 36^3", "R4": "64-atom, K_POINTS gamma, FFT 72^3",
        "R5": "64-atom, automatic 1 1 1 (complex), FFT 72^3", "R6": "64-atom, 2x2x2, FFT 72^3"}
NAT = {"R1": 2, "R1p": 2, "R2": 8, "R2p": 8, "R3": 8, "R4": 64, "R5": 64, "R6": 64}


# --------------------------------------------------------------------------- parsers (on top of abrun's parse.parse_out)

RE_TERMS = {"one_electron": r"one-electron contribution =\s+(-?[\d.]+) Ry", "hartree": r"hartree contribution\s+=\s+(-?[\d.]+) Ry",
            "xc": r"xc contribution\s+=\s+(-?[\d.]+) Ry", "ewald": r"ewald contribution\s+=\s+(-?[\d.]+) Ry",
            "paw": r"one-center paw contrib\.\s+=\s+(-?[\d.]+) Ry"}
RE_KBLOCK = re.compile(r"k =\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)\s*\(\s*(\d+) PWs\)\s+bands \(ev\):\s*\n\n"
                       r"((?:[ \t]*-?\d+\.\d+(?:[ \t]+-?\d+\.\d+)*[ \t]*\n)+)")
RE_KLIST = re.compile(r"k\(\s*(\d+)\) = \(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\), wk =\s*([\d.]+)")
RE_DENSE = re.compile(r"Dense\s+grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)")
RE_SMOOTH = re.compile(r"Smooth grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)")
RE_SYM = re.compile(r"(\d+) Sym\. Ops\.(, with inversion,| \(no inversion\))")
RE_PER_CALL = re.compile(r"^AB_PROFILE (\w+) M=(\d+) N=(\d+) K=(\d+)", re.M)
RE_CBANDS = re.compile(r"c_bands:\s+(\d+) eigenvalues not converged\s*\n\s*ethr =\s*([0-9.Ee+-]+)")


def parse_cbands(text):
    """[(scf iteration, eigenvalues not converged, ethr), ...]: every c_bands warning with the SCF iteration it occurred in."""
    blocks = re.split(r"iteration #\s*(\d+)", text)
    return [(int(it), int(n), ethr) for it, body in zip(blocks[1::2], blocks[2::2]) for n, ethr in RE_CBANDS.findall(body)]


def parse_terms(text):
    out = {}
    for k, pat in RE_TERMS.items():
        m = re.findall(pat, text)
        out[k] = float(m[-1]) if m else None
    return out


def parse_bands(text):
    """[{k: (kx,ky,kz) in 2pi/alat, npw, eig: [eV]}] from the final 'End of self-consistent calculation' block."""
    i = text.rfind("End of self-consistent calculation")
    body = text[i:] if i >= 0 else text
    return [{"k": (float(a), float(b), float(c)), "npw": int(n), "eig": [float(v) for v in vals.split()]}
            for a, b, c, n, vals in RE_KBLOCK.findall(body)]


def parse_klist(text, nks):
    """The first nks k(i) = (...), wk = ... lines (cartesian block)."""
    rows = RE_KLIST.findall(text)[:nks]
    return [{"i": int(i), "k": (float(a), float(b), float(c)), "wk": float(w)} for i, a, b, c, w in rows]


def parse_extra(text):
    out = {"terms": parse_terms(text), "bands": parse_bands(text)}
    m = RE_DENSE.search(text)
    out["ngm"], out["fft"] = (int(m.group(1)), [int(m.group(i)) for i in (2, 3, 4)]) if m else (None, None)
    m = RE_SMOOTH.search(text)
    out["ngms"], out["fft_smooth"] = (int(m.group(1)), [int(m.group(i)) for i in (2, 3, 4)]) if m else (None, None)
    m = RE_SYM.search(text)
    out["nsym"], out["sym_note"] = (int(m.group(1)), m.group(2).strip(" ,()")) if m else (None, None)
    out["supercell"] = "This is a supercell" in text
    out["gamma_only"] = "gamma-point specific algorithms are used" in text
    m = re.search(r"number of k points=\s*(\d+)", text)
    out["nks"] = int(m.group(1)) if m else None
    out["klist"] = parse_klist(text, out["nks"]) if out["nks"] else []
    out["npw_per_k"] = [b["npw"] for b in out["bands"]]
    out["c_bands_warning"] = "eigenvalues not converged" in text
    out["c_bands_seq"] = parse_cbands(text)
    out["scf_correction_warning"] = "SCF correction compared to forces is large" in text      # PW/src/forces.f90: sumscf > 0.1 * sumfor
    j = text.rfind("iteration #")
    out["c_bands_final_warning"] = "eigenvalues not converged" in text[j:] if j >= 0 else False
    m = re.findall(r"ethr =\s*([0-9.Ee+-]+)", text[j:]) if j >= 0 else []
    out["ethr_final"] = m[-1] if m else None
    return out


def per_call_dims(err_text):
    """{routine: [(M,N,K), ...]} from the AB_PROFILE per-call lines (one per GPU call)."""
    out: dict = {}
    for r, m, n, k in RE_PER_CALL.findall(err_text):
        out.setdefault(r, []).append((int(m), int(n), int(k)))
    return out


def nh_of_upf(path):
    """Number of beta projectors times (2l+1) per atom, from the UPF's PP_BETA angular momenta."""
    ls = [int(x) for x in re.findall(r'<PP_BETA\.\d+[^>]*angular_momentum="(\d+)"', Path(path).read_text(errors="ignore"))]
    return sum(2 * l + 1 for l in ls), len(ls)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_tracked(path):
    """True when git tracks path (or a file under it) at the checked-out commit; the provenance header says so when it does not."""
    r = subprocess.run(["git", "-C", str(exlib.REPO_ROOT), "ls-files", "--error-unmatch", str(path)], capture_output=True, text=True)
    return r.returncode == 0


# --------------------------------------------------------------------------- Ewald controls (the fold gate's second estimator)
#
# PW/src/ewald.f90 (QE 7.4.1 and develop): alpha starts at 2.9 and drops by 0.1 until the G-sum error bound
# 2 Q^2 sqrt(2 alpha / 2pi) erfc(sqrt(ecutrho / 4 alpha)) <= 1e-7 Ry (Q = total ionic charge); the G sum runs over the ecutrho
# sphere; the real-space sum runs over pairs with r <= rmax = 4/sqrt(alpha) (rgen keeps r^2 <= rmax^2; "terms up to ZiZj*erfc(4)
# are counted"). Both truncations are replicated here and the two dropped pieces are summed explicitly over the dropped regions.

E2 = 2.0                                            # Rydberg units: e^2 = 2
GMAX2_CONV, RMAX_CONV = 300.0, 20.0                 # converged sum: exp(-300/4alpha) and erfc(20 sqrt(alpha)) are < 1e-30 for alpha in [0.6, 1.5]
EWALD_CELL = {"R1": "R1", "R1p": "R1p", "R2": "R2", "R2p": "R2p", "R3": "R2", "R4": "R4", "R5": "R4", "R6": "R4"}   # same lattice, charge and ecutrho -> same alpha and rmax


def _lattice_points(vecs, nmax):
    rng = np.arange(-nmax, nmax + 1)
    return np.array(list(product(rng, rng, rng)), float) @ vecs


def ewald_gsum(at, pos, alpha, gmin2, gmax2, Z=4.0):
    """(4pi/omega) sum over gmin2 < |G|^2 <= gmax2 of |S(G)|^2 exp(-|G|^2/4alpha)/|G|^2 (ewald.f90's G loop without the constant
    terms, before the 0.5 e^2 factor) and the number of G-vectors summed. |G|^2 in bohr^-2 is the kinetic energy in Ry."""
    omega = abs(np.linalg.det(at)); bg = 2 * np.pi * np.linalg.inv(at).T
    G = _lattice_points(bg, int(np.sqrt(gmax2) * max(np.linalg.norm(at, axis=1)) / (2 * np.pi)) + 2)   # |m_i| <= |G| |a_i| / 2pi
    gg = (G ** 2).sum(1); m = (gg > max(gmin2, 1e-12)) & (gg <= gmax2); G, gg = G[m], gg[m]
    S = np.zeros(len(gg), complex)
    for p in pos:                                   # atom by atom: the 64-atom sphere has ~0.5M G-vectors
        S += Z * np.exp(1j * (G @ p))
    return 4 * np.pi / omega * ((np.abs(S) ** 2) * np.exp(-gg / (4 * alpha)) / gg).sum(), int(m.sum())


def ewald_rsum(at, pos, alpha, rmin, rmax, Z=4.0):
    """sum over ordered pairs (i, j + R) with rmin < r <= rmax of Z^2 erfc(sqrt(alpha) r)/r (ewald.f90's R loop, before the
    0.5 e^2 factor) and the number of pairs summed."""
    from scipy.special import erfc
    bg = 2 * np.pi * np.linalg.inv(at).T
    R = _lattice_points(at, int(rmax * max(np.linalg.norm(bg, axis=1)) / (2 * np.pi)) + 2)             # |n_i| <= |R| |b_i| / 2pi
    tot, npairs = 0.0, 0
    for i in range(len(pos)):
        d = pos[i][None, None, :] - pos[None, :, :] + R[:, None, :]
        rr = np.sqrt((d ** 2).sum(2)); mm = (rr > max(rmin, 1e-8)) & (rr <= rmax)
        tot += (Z * Z * erfc(np.sqrt(alpha) * rr[mm]) / rr[mm]).sum(); npairs += int(mm.sum())
    return tot, npairs


def ewald_sum(at, pos, alpha, gmax2, rmax, Z=4.0):
    """Ewald energy (Ry) in ewald.f90's convention for lattice rows at (bohr), positions pos (bohr): G-sphere |G|^2 <= gmax2
    (bohr^-2), real-space pairs with |r| <= rmax (bohr), splitting alpha (bohr^-2)."""
    omega = abs(np.linalg.det(at)); nat = len(pos); charge = nat * Z
    ewaldg = -4 * np.pi / omega * charge ** 2 / alpha / 4.0 + ewald_gsum(at, pos, alpha, 0.0, gmax2, Z)[0] - nat * Z ** 2 * np.sqrt(4 * alpha / np.pi)
    return 0.5 * E2 * (ewaldg + ewald_rsum(at, pos, alpha, 0.0, rmax, Z)[0])


def qe_ewald_gbound(charge, alpha, ecutrho):
    """ewald.f90's 'upperbound' on the G-sum truncation error (Ry per cell)."""
    from scipy.special import erfc
    return 2 * charge ** 2 * np.sqrt(2 * alpha / (2 * np.pi)) * erfc(np.sqrt(ecutrho / 4 / alpha))


def qe_ewald_alpha(charge, ecutrho):
    """PW/src/ewald.f90: alpha from 2.9 down in steps of 0.1 until the G-sum error bound is <= 1e-7 Ry."""
    alpha = 2.9
    while True:
        alpha = round(alpha - 0.1, 10)
        if alpha <= 0:
            raise ValueError("optimal alpha not found")
        if qe_ewald_gbound(charge, alpha, ecutrho) <= 1e-7:
            return alpha


def ewald_controls():
    """Per distinct cell: ewald.f90's rule (alpha search, G-sphere of ecutrho, rmax = 4/sqrt(alpha)), a converged sum, and the
    two pieces the rule drops, each a direct sum over the dropped region: the real-space shell r > rmax (to 20 bohr) and the G
    tail ecutrho < |G|^2 <= 300 Ry. Legs on the same lattice with the same charge and ecutrho share a cell (EWALD_CELL)."""
    from scipy.special import erfc
    fcc = np.array([(-1, 0, 1), (0, 1, 1), (-1, 1, 0)]) * A_CONV / 2            # ibrav 2
    p2 = np.array([(0, 0, 0), (.25, .25, .25)]) @ fcc
    cells = {"R1": (fcc, p2, ECUTRHO), "R1p": (fcc, p2, 200.0),
             "R2": (np.eye(3) * A_CONV, np.array(CONV8) * A_CONV, ECUTRHO), "R2p": (np.eye(3) * A_CONV, np.array(CONV8) * A_CONV, 200.0),
             "R4": (np.eye(3) * 2 * A_CONV, np.array(positions64()) * 2 * A_CONV, ECUTRHO)}
    out = {}
    for name, (at, pos, ecutrho) in cells.items():
        nat = len(pos); charge = nat * 4.0; alpha = qe_ewald_alpha(charge, ecutrho); rmax = 4 / np.sqrt(alpha)
        d_nn = min(np.linalg.norm(pos[i] - pos[j]) for i in range(nat) for j in range(nat) if i != j)
        shell, npairs = ewald_rsum(at, pos, alpha, rmax, RMAX_CONV)
        gtail, ng = ewald_gsum(at, pos, alpha, ecutrho, GMAX2_CONV)
        out[name] = {"nat": nat, "alpha": alpha, "rmax_bohr": rmax, "d_nn_bohr": float(d_nn), "erfc_sqrt_alpha_d_nn": float(erfc(np.sqrt(alpha) * d_nn)),
                     "qe_rule_per_atom": ewald_sum(at, pos, alpha, ecutrho, rmax) / nat,
                     "converged_per_atom": ewald_sum(at, pos, 1.0, GMAX2_CONV, RMAX_CONV) / nat,
                     "shell_per_atom": 0.5 * E2 * shell / nat, "shell_pairs": npairs,           # dropped real-space shell r > rmax (positive, Ry/atom)
                     "gtail_per_atom": 0.5 * E2 * gtail / nat, "gtail_nG": ng,                  # dropped G tail |G|^2 > ecutrho (positive, Ry/atom)
                     "gsum_bound_per_atom": qe_ewald_gbound(charge, alpha, ecutrho) / nat,
                     "gtail_beyond_bound_per_atom": qe_ewald_gbound(charge, alpha, GMAX2_CONV) / nat}      # what the explicit G tail (to 300 Ry) itself leaves out, by ewald.f90's bound
    out["R2"]["converged_by_alpha"] = {str(a): ewald_sum(*cells["R2"][:2], a, GMAX2_CONV, RMAX_CONV) / 8 for a in (0.6, 1.0, 1.5)}   # alpha independence
    return out


def stage(t):
    print(f"\n=== {t} ===", flush=True)


def rnd(x, nd=8):
    """Differences of printed values, free of float noise: |a - b| rounded to the printed precision."""
    return abs(round(x, nd))


# --------------------------------------------------------------------------- running through abrun

def run_nbnd_control(args, inputs_dir):
    """R6 with nbnd = 168 on the fresh-develop upstream pw.x (the one-tree control build), 1 rank, OMP/OpenBLAS/vecLib threads 1,
    under machlock bench: the closure of the c_bands attribution (the lagging states are the group that nbnd = 160 cuts).
    Launched through exlib.run because the binary is not in abrun's catalog; under --reuse the record is reloaded only when the
    binary sha256 and the rewritten input sha256 match. Returns a dict (with the output text under 'out'), or {'skipped': why}."""
    d = RES / "ctl" / "R6n168"
    d.mkdir(parents=True, exist_ok=True)
    src = pw_input("si64k", 1, 2 * A_CONV, positions64(), NBND_CONTROL, k_auto(2), CONV_64, nr=72)
    (inputs_dir / "R6n168.in").write_text(src)
    text = presets.rewrite_input(src)                 # outdir ./tmp, pseudo_dir ./pseudo, as abrun does for the other legs
    in_sha = presets.sha256_text(text)
    meta_p = d / "control.json"
    if not UPSTREAM_BIN.is_file():
        return {"skipped": f"control binary not present: {UPSTREAM_BIN} (set MUPPET_QE_DEVELOP_UPSTREAM)"}
    bsha = sha256(UPSTREAM_BIN)
    if args.reuse and meta_p.exists() and (d / "out.txt").exists():
        meta = json.loads(meta_p.read_text()); out = (d / "out.txt").read_text(errors="replace")
        if meta.get("binary_sha256") == bsha and meta.get("input_sha256") == in_sha and "JOB DONE" in out:
            print(f"  [reuse] R6n168: E = {meta.get('energy_Ry')} Ry; binary sha256 {bsha[:12]}, input sha256 {in_sha[:12]} (results/ctl/R6n168)")
            meta["out"] = out
            return meta
    if args.no_launch:
        return {"skipped": "--no-launch: no reusable control record in results/ctl/R6n168"}
    (d / "input.in").write_text(text)
    pd = d / "pseudo"; pd.mkdir(exist_ok=True)
    upf = (RES / "pseudo" / UPF_NAME) if (RES / "pseudo" / UPF_NAME).exists() else (SI64_INPUTS / UPF_NAME)
    link = pd / UPF_NAME
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(upf.resolve())
    shutil.rmtree(d / "tmp", ignore_errors=True)
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    cmd = [str(launch.MPIRUN), "-np", "1"] + [x for k, v in env.items() for x in ("-x", f"{k}={v}")] + [str(UPSTREAM_BIN), "-in", "input.in"]
    share = os.environ.pop("MUPPET_SHARE", None)       # the control takes machlock bench (exlib.run's default), as the audit session asked
    load0 = exlib.load1()
    t0 = time.time()
    try:
        rc, wall, out = exlib.run(cmd, d, d / "out.txt", env=env, label="si-supercell-folding R6n168 control (develop upstream, 1 rank, bench)", timeout=7200)
    finally:
        if share is not None:
            os.environ["MUPPET_SHARE"] = share
    shutil.rmtree(d / "tmp", ignore_errors=True)
    pw = exlib.parse_pw(out)
    meta = {"binary": str(UPSTREAM_BIN), "binary_desc": UPSTREAM_DESC, "binary_sha256": bsha, "input_sha256": in_sha, "nbnd": NBND_CONTROL, "rc": rc,
            "wall_s": wall, "driver_wall_s": time.time() - t0, "load1_at_start": load0, "lock": "machlock bench", "ranks": 1,
            "energy_Ry": pw.get("total_energy_Ry"), "iterations": pw.get("scf_iterations"), "qe_wall": pw.get("wall_time")}
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"  [R6n168] rc {rc}, E = {meta['energy_Ry']} Ry, {meta['iterations']} iterations, wall {wall:.0f} s (observation; machlock bench, load1 {load0:.2f} at start)")
    meta["out"] = out
    return meta


class Legs:
    """All solver launches of this example; each returns abrun's run record (run.json as a dict)."""

    def __init__(self, args, inputs_dir):
        self.args = args
        self.inputs_dir = inputs_dir
        self.records: dict[str, dict] = {}
        self.reused: set = set()

    def _reusable(self, run_dir, name, mode, binary_id, ranks, nk):
        """(record, None) when the finished record in run_dir is the run this leg would launch, else (None, why).

        The key is the whole request plus the identity of what ran: input sha256 (after abrun's rewrite), mode, binary id AND the
        binary's sha256 as the catalog has it now (a rebuilt /usr/local/bin/pw.x keeps its id), the dylib sha256 for frozen ids, the
        pseudopotential sha256, ranks and nk, and that the record's run_dir is the directory being checked."""
        p = run_dir / "run.json"
        if not self.args.reuse:
            return None, "not --reuse"
        if not p.exists():
            return None, "no run.json"
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            return None, f"unreadable run.json ({exc})"
        cat = binaries.get(binary_id)
        want = presets.sha256_text(presets.rewrite_input((self.inputs_dir / f"{name}.in").read_text()))
        rb, rq, ps = rec.get("binary") or {}, rec.get("request") or {}, (rec.get("pseudos") or [{}])[0]
        tests = [("state done", rec.get("state") == "done"), ("input sha256", rec.get("input_sha256") == want), ("mode", rq.get("mode") == mode),
                 ("binary id", rb.get("id") == binary_id), ("binary sha256 == catalog", cat is not None and rb.get("sha256") == cat.sha256),
                 ("dylib sha256 == catalog", cat is not None and (cat.dylib_sha256 is None or rb.get("dylib_sha256") == cat.dylib_sha256)),
                 ("pseudo sha256 pinned", ps.get("file") == UPF_NAME and ps.get("sha256") == UPF_SHA256),
                 ("ranks", int(rq.get("ranks", -1)) == ranks), ("nk", int(rq.get("nk", -1)) == nk),
                 ("run_dir", Path(rec.get("run_dir", "")).resolve() == run_dir.resolve()), ("out.txt", (run_dir / "out.txt").exists())]
        bad = [t for t, okk in tests if not okk]
        return (rec, None) if not bad else (None, "mismatch: " + ", ".join(bad))

    def run(self, leg, name, mode, binary_id, ranks=1, nk=1, trace=False, run_dir=None):
        run_dir = run_dir or (RES / ("ref" if mode == "reference" else "ab") / leg)
        rec, why = self._reusable(run_dir, name, mode, binary_id, ranks, nk)
        if rec is not None:
            b = rec["binary"]
            print(f"  [reuse] {leg}: {rec['id']} E = {rec['result']['total_energy_Ry']} Ry; binary {b['id']} sha256 {b['sha256'][:12]}"
                  + (f", dylib {b['dylib_sha256'][:12]}" if b.get("dylib_sha256") else "") + f", pseudo {rec['pseudos'][0]['sha256'][:12]}, "
                  f"ranks {rec['request']['ranks']} nk {rec['request']['nk']} (results/{run_dir.relative_to(RES)})")
            self.reused.add(leg)
        else:
            if self.args.no_launch:
                raise SystemExit(f"--no-launch: leg {leg} is not reusable from results/{run_dir.relative_to(RES)} ({why}); refusing to launch pw.x")
            req = launch.Request(preset="custom", input_path=str(self.inputs_dir / f"{name}.in"), binary_id=binary_id, mode=mode,
                                 ranks=ranks, nk=nk, omp=1, trace=trace, lock=self.args.lock,
                                 label=f"si-supercell-folding {leg}")
            t0 = time.time()
            try:
                rec = runner.run_sync(req, run_dir=run_dir, log=lambda s: print(s, flush=True))
            except runner.PlanError as exc:
                raise SystemExit(f"abrun refused leg {leg}: {exc.violations}") from exc
            print(f"  [{leg}] {rec['state']} in {time.time() - t0:.0f} s of driver time (QE wall {rec['result'].get('wall_qe_s')} s, observation)")
        rec["_leg"], rec["_name"] = leg, name
        rec["_extra"] = parse_extra((run_dir / "out.txt").read_text(errors="replace"))       # the directory that was checked, not rec['run_dir']
        err = run_dir / "err.txt"
        rec["_dims"] = per_call_dims(err.read_text(errors="replace")) if (err.exists() and mode != "reference") else {}
        self.records[leg] = rec
        return rec


def td_sites(name):
    """Atoms whose site symmetry under QE's symmetry group of the run is the full Td, so their force vanishes identically.

    R1 (fcc primitive): QE keeps the 48 operations with inversion and fractional translations, every atom has Td site symmetry.
    Cubic cells (R2-R6): 'This is a supercell, fractional translations are disabled' leaves the 24 point operations of Td about the
    origin; only the atoms at (i/2, j/2, k/2) are fixed by all of them. The others sit at C3v sites (sublattice 2: residual along
    [111]) or Cs sites (face-centre type: residual in a mirror plane), where a symmetrized residual force is allowed."""
    pos = {"R1": [(0, 0, 0), (.25, .25, .25)], "R2": CONV8, "R3": CONV8, "R4": positions64(), "R5": positions64(), "R6": positions64()}
    if name not in pos:
        return []
    if name == "R1":
        return list(range(2))
    return [i for i, p in enumerate(pos[name]) if all(abs(c - round(2 * c) / 2) < 1e-9 for c in p)]


def force_split(name, forces):
    """{td_max, other_max, n_td, n_nonzero} from the printed force block: the Td-site atoms versus the rest."""
    if not forces:
        return None
    F = np.array(forces, float); td = td_sites(name); other = [i for i in range(len(F)) if i not in td]
    return {"n_td": len(td), "td_sites": td, "td_max": float(np.abs(F[td]).max()) if td else 0.0,
            "other_max": float(np.abs(F[other]).max()) if other else 0.0, "n_nonzero": int((np.abs(F).sum(1) > 0).sum())}


def rec_row(rec):
    r, x, e, b = rec["result"], rec["_extra"], rec["etiquette"], rec["binary"]
    m = e.get("machine_at_acquisition") or {}
    return {"leg": rec["_leg"], "input": rec["_name"], "id": rec["id"], "state": rec["state"], "binary": b["id"], "binary_sha256": b["sha256"],
            "started": rec.get("started"), "finished": rec.get("finished"),
            "force_split": force_split(rec["_name"], r.get("forces")), "total_scf_correction": r.get("total_scf_correction"),
            "scf_correction_warning": x["scf_correction_warning"],
            "mode": rec["request"]["mode"], "ranks": rec["request"]["ranks"], "nk": rec["request"]["nk"], "qe_version": r.get("qe_version"),
            "E_Ry": r.get("total_energy_Ry"), "E_per_atom_Ry": (r["total_energy_Ry"] / NAT[rec["_name"]]) if r.get("total_energy_Ry") is not None else None,
            "iters": r.get("scf_iterations"), "converged": r.get("converged"), "nks": x["nks"], "nsym": x["nsym"], "sym_note": x["sym_note"],
            "supercell": x["supercell"], "gamma_only": x["gamma_only"], "fft": x["fft"], "fft_smooth": x["fft_smooth"], "ngm": x["ngm"],
            "npw_per_k": x["npw_per_k"], "terms": x["terms"], "pressure_kbar": r.get("pressure_kbar"), "stress_kbar": r.get("stress_kbar"),
            "max_force": (float(np.abs(np.array(r["forces"])).max()) if r.get("forces") else None),
            "wall_qe_s": r.get("wall_qe_s"), "wall_runner_s": rec.get("wall_runner_s"), "timing_grade": e.get("timing_grade"),
            "lock_effective": e.get("lock_effective"), "load1_at_acquisition": m.get("load1"), "foreign_at_launch": len(e.get("foreign_at_launch") or []),
            "contention_events": len(e.get("contention") or []), "c_bands_warning": x["c_bands_warning"], "c_bands_final_warning": x["c_bands_final_warning"],
            "homo_eV": r.get("homo_eV"), "lumo_eV": r.get("lumo_eV"),
            "run_dir": rec["run_dir"]}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4, help="MPI ranks of the reference legs (frozen legs always -np 1)")
    ap.add_argument("--lock", default="share", choices=list(launch.LOCKS), help="abrun lock mode (share = MUPPET_SHARE semantics)")
    ap.add_argument("--binary", default=FROZEN_DEFAULT, help="frozen apple-bottom build id for the parity legs")
    ap.add_argument("--fast", action="store_true", help="skip R5 and the R6 gpu leg")
    ap.add_argument("--auto-leg", action="store_true", help="add R4 in AB_MODE=auto")
    ap.add_argument("--trace", action="store_true", help=f"add R4 at the library floor on {TRACE_BINARY} with AB_TRACE")
    ap.add_argument("--skip-ab", action="store_true", help="reference legs only")
    ap.add_argument("--reuse", action="store_true", help="keep finished legs whose input, mode, binary id + sha256 (+ dylib sha256), pseudo sha256, ranks and nk match")
    ap.add_argument("--no-launch", action="store_true", help="refuse to launch any solver: every leg must be reusable (re-evaluate the gates from the run records)")
    ap.add_argument("--skip-nbnd-control", action="store_true", help="skip the R6 nbnd = 168 control on the fresh-develop upstream pw.x")
    args = ap.parse_args()
    if args.no_launch and not args.reuse:
        ap.error("--no-launch needs --reuse")
    if not args.reuse:
        exlib.reset_results(RES)
    RES.mkdir(exist_ok=True)
    checks = exlib.Checks()
    t0 = time.time()
    S: dict = {"provenance": exlib.provenance(), "args": vars(args), "ecutwfc": ECUT, "ecutrho": ECUTRHO, "a_conv_bohr": A_CONV,
               "recorded": RECORDED, "recorded_all": RECORDED_ALL, "recorded_source": RECORDED_SOURCE, "lock": args.lock,
               "walls_note": "untimed observations; other sessions run on this machine; no speedup computed"}
    # the code that produced the table, by content: the git commit in provenance may not contain it
    abrun_dir = HERE.parents[1] / "tools" / "abrun"
    S["provenance"].update({"driver": str(HERE / "run.py"), "driver_sha256": sha256(HERE / "run.py"),
                            "abrun_sha256": {p.name: sha256(p) for p in sorted(abrun_dir.glob("*.py"))},
                            "tracked_in_commit": {"examples/si-supercell-folding/run.py": git_tracked(HERE / "run.py"), "tools/abrun": git_tracked(abrun_dir)},
                            "regenerated_with_reuse": bool(args.reuse)})

    # ---------------------------------------------------------------- S0 provenance
    stage("S0. Provenance: reference pw.x, frozen apple-bottom build, pseudopotential, inputs")
    ref = binaries.get(REF_ID)
    frz = binaries.get(args.binary)
    if ref is None or frz is None:
        raise SystemExit(f"abrun catalog lacks {REF_ID!r} or {args.binary!r}: run bin/abrun binaries --refresh")
    n_ab = exlib.apple_bottom_symbols(Path(ref.path))
    checks.add("reference pw.x is apple-bottom-free (nm)", n_ab == 0 and ref.ab_linkage == "none", f"{n_ab} _ab_ symbols in {ref.path} (QE {ref.qe_version})", "0 _ab_* exports, no libapplebottom link")
    sums = {}
    sums_file = AB_FROZEN_ROOT / args.binary / "SHA256SUMS"
    for line in sums_file.read_text().splitlines():
        v = line.split()
        if len(v) == 2:
            sums[v[1]] = v[0]
    sha_pw, sha_dy = sha256(frz.path), sha256(frz.dylib_path)
    checks.add("frozen build SHA256SUMS verified (pw.x and libapplebottom.dylib)", sha_pw == sums.get("pw.x") and sha_dy == sums.get("libapplebottom.dylib") and frz.sha_verified,
               f"pw.x {sha_pw[:12]}, dylib {sha_dy[:12]} in {sums_file}", "both equal SHA256SUMS; abrun re-verifies at every launch")
    if args.binary == FROZEN_DEFAULT:
        checks.add("frozen dylib ab_build_info (strings)", frz.ab_build_info == AB_BUILD_EXPECTED, f"{frz.ab_build_info!r}", f"== {AB_BUILD_EXPECTED!r} for {FROZEN_DEFAULT}")
    else:
        checks.report("frozen dylib ab_build_info (strings)", f"{frz.ab_build_info!r}", f"--binary {args.binary} is not the default {FROZEN_DEFAULT}: no expected string, reported")
    prov_txt = (AB_FROZEN_ROOT / args.binary / "PROVENANCE.txt").read_text(errors="replace")
    S["provenance"].update({"ab_frozen": str(AB_FROZEN_ROOT / args.binary), "ab_pw_x_sha256": sha_pw, "ab_dylib_sha256": sha_dy, "ab_build_info": frz.ab_build_info,
                            "ab_provenance_head": "\n".join(prov_txt.splitlines()[:4]), "ab_omp1_only": frz.omp1_only, "ab_class": frz.cls, "ab_qe_build": frz.qe_build})
    # pseudopotential: copied into results/pseudo/ with its sha pinned here and in abrun (presets.UPF_SHA256). abrun resolves the UPF by
    # name from its own search list (SI64_INPUTS first, so the records point at ~/qe-test/xp/inputs), and refuses a file whose sha differs.
    src_upf = SI64_INPUTS / UPF_NAME
    pdir = RES / "pseudo"; pdir.mkdir(exist_ok=True)
    shutil.copy2(src_upf, pdir / UPF_NAME)
    exlib.check_upf(pdir / UPF_NAME)
    upf_sha = sha256(pdir / UPF_NAME)
    checks.add("pseudopotential pinned", upf_sha == UPF_SHA256 and presets.UPF_SHA256.get(UPF_NAME) == UPF_SHA256, f"{UPF_NAME} sha256 {upf_sha[:12]} (copied from {src_upf.parent})",
               f"== {UPF_SHA256[:12]} here and in abrun's presets.UPF_SHA256 (PAW PBE, z=4, 6 projectors)")
    nh, nbeta = nh_of_upf(pdir / UPF_NAME)
    S["pseudo"] = {"file": UPF_NAME, "sha256": upf_sha, "source": str(src_upf), "nbeta": nbeta, "nh": nh}
    inputs_dir = RES / "inputs"; inputs_dir.mkdir(exist_ok=True)
    link = inputs_dir / UPF_NAME
    if link.is_symlink() and not link.exists():
        link.unlink()                                   # dangling symlink from a moved results/ tree
    if not link.exists():
        os.symlink(pdir / UPF_NAME, link)
    inputs = all_inputs()
    for name, text in inputs.items():
        (inputs_dir / f"{name}.in").write_text(text)
    S["inputs"] = {n: {"sha256": presets.sha256_text(t), "desc": DESC[n], "nat": NAT[n]} for n, t in inputs.items()}
    # the 64 generated positions are the campaign's (so its recorded energies apply)
    camp = CAMPAIGN_C.read_text().split("ATOMIC_POSITIONS crystal")[1].split("K_POINTS")[0].split()
    camp_pos = sorted(tuple(float(v) for v in camp[i + 1:i + 4]) for i in range(0, len(camp), 4))
    dpos = max(abs(a - b) for p, q in zip(positions64(), camp_pos) for a, b in zip(p, q)) if len(camp_pos) == 64 else float("inf")
    checks.add("64-atom positions equal the campaign input C", len(camp_pos) == 64 and dpos < 1e-6, f"{len(camp_pos)} positions, max |dr| = {dpos:.1e}", f"sorted sets equal to 1e-6 ({CAMPAIGN_C.name})")
    print(f"  reference {ref.path} sha {ref.sha256[:12]}; frozen {frz.path} ({frz.ab_build_info}); UPF nh = {nh} ({nbeta} betas); lock {args.lock}; ranks {args.ranks}")
    print(f"  foreign solvers at start: {exlib.external_solver_pids()[:2]}; load1 {exlib.load1():.2f}")

    legs = Legs(args, inputs_dir)
    nk4 = min(4, args.ranks)

    # ---------------------------------------------------------------- S1 reference legs
    stage(f"S1. Reference legs on {ref.path} (mpirun -np {args.ranks}; -nk {nk4} for R3/R6)")
    order = ["R2", "R1", "R4", "R3", "R6", "R1p", "R2p"] + ([] if args.fast else ["R5"])
    for name in order:
        legs.run(name, name, "reference", REF_ID, ranks=args.ranks, nk=(nk4 if name in ("R3", "R6") else 1))

    # ---------------------------------------------------------------- S2 apple-bottom legs
    ab_legs = []
    if not args.skip_ab:
        stage(f"S2. apple-bottom legs on {frz.path} (mpirun -np 1, OMP_NUM_THREADS=1, profiler on)")
        ab_legs = [("R4_cpu", "R4", "ab_cpu"), ("R4_gpu", "R4", "ab_gpu_all"), ("R6_cpu", "R6", "ab_cpu")] + ([] if args.fast else [("R6_gpu", "R6", "ab_gpu_all")])
        if args.auto_leg:
            ab_legs.append(("R4_auto", "R4", "ab_auto"))
        for leg, name, mode in ab_legs:
            legs.run(leg, name, mode, args.binary, ranks=1, nk=1)
        if args.trace:
            legs.run("R4_trace", "R4", "ab_gpu", TRACE_BINARY, ranks=1, nk=1, trace=True)
    else:
        stage("S2. apple-bottom legs skipped (--skip-ab)")

    # ---------------------------------------------------------------- S2b nbnd control
    if args.skip_nbnd_control:
        stage("S2b. nbnd = 168 control skipped (--skip-nbnd-control)"); ctl168 = None
    else:
        stage(f"S2b. nbnd = {NBND_CONTROL} control: R6 on the fresh-develop upstream pw.x ({UPSTREAM_BIN.name}, 1 rank, machlock bench)")
        ctl168 = run_nbnd_control(args, inputs_dir)
        if ctl168.get("skipped"):
            print(f"  skipped: {ctl168['skipped']}")
    S["control_nbnd168"] = {k: v for k, v in (ctl168 or {}).items() if k != "out"} or None

    # ---------------------------------------------------------------- S3/S4 gates
    stage("S3. Gates")
    R = legs.records
    rows = {leg: rec_row(rec) for leg, rec in R.items()}
    S["runs"] = rows
    starts = [r["started"] for r in rows.values() if r.get("started")]; ends = [r["finished"] for r in rows.values() if r.get("finished")]
    S["provenance"]["run_records"] = {"first_started": min(starts) if starts else None, "last_finished": max(ends) if ends else None,
                                      "reused": sorted(legs.reused), "launched": sorted(set(rows) - legs.reused)}

    def E(leg):
        v = rows.get(leg, {}).get("E_Ry")
        return float("nan") if v is None else v

    def term(leg, key):
        v = (rows.get(leg, {}).get("terms") or {}).get(key)
        return float("nan") if v is None else v

    def ok(x):
        return bool(np.isfinite(x))

    for leg in ["R1", "R1p", "R2", "R2p", "R3", "R4", "R5", "R6"]:
        if leg in rows:
            r = rows[leg]
            checks.add(f"{leg} converged on the reference pw.x", r["state"] == "done" and bool(r["converged"]) and r["E_Ry"] is not None,
                       f"E = {r['E_Ry']} Ry, {r['iters']} iterations, {r['nks']} k, FFT {r['fft']}, QE wall {r['wall_qe_s']} s (observation)", "state done, SCF converged")
    cbw = {leg: rows[leg]["c_bands_final_warning"] for leg in rows}
    ref_legs = [leg for leg in rows if rows[leg]["mode"] == "reference"]
    frz_legs = [leg for leg in rows if rows[leg]["mode"] != "reference"]
    checks.add("eigenvalues converged in the final SCF iteration (reference legs: no c_bands warning there)", not any(cbw[l] for l in ref_legs),
               f"final-iteration warnings in {[l for l in ref_legs if cbw[l]] or 'none'}; mid-SCF warnings in {[l for l in ref_legs if rows[l]['c_bands_warning']]}; final ethr "
               + ", ".join(f"{l} {R[l]['_extra']['ethr_final']}" for l in ("R4", "R6") if l in rows),
               "none in the last iteration (mid-SCF ones are benign: Davidson tightens ethr as the density converges and reports the states that lag at that iteration)")
    if frz_legs:
        # The warning is a band-count and rank-layout artefact, not a build property. Attribution (apple-bottom audit session, 2026-10-18,
        # one fresh develop checkout 4a626b7ce built three ways: upstream/OpenBLAS, + ab-frozen 33fbaa8c8944, + the cegterg/regterg patch;
        # R6 at 1 rank in cpu and gpu): all four agree in every SCF iteration, warning sequence, eigenvalue, force and stress, so the patch,
        # the BLAS library and GPU numerics have no effect. At k = (0,-1/2,-1/2) bands 157-168 sit at 9.5080-9.5081 eV, so nbnd = 160 cuts a
        # near-degenerate group; with -nk 4 only pool 0 (Gamma) prints (Modules/environment.f90), which hides the warning; nbnd = 168 gives a
        # clean final iteration. Here the gate is the control: eigenvalues equal the reference leg's and cpu == gpu warn identically.
        warned, ctl, ctl_ok = [l for l in frz_legs if cbw[l]], [], True
        for leg in warned:
            name = R[leg]["_name"]
            bn, bl = R[name]["_extra"]["bands"], R[leg]["_extra"]["bands"]
            d = max((abs(a - b) for x, y in zip(bn, bl) for a, b in zip(x["eig"], y["eig"])), default=float("inf")) if bn and len(bn) == len(bl) else float("inf")
            ctl_ok = ctl_ok and d <= PRINT_EV + 1e-9
            seq = R[leg]["_extra"]["c_bands_seq"]; last = [s for s in seq if s[0] == rows[leg]["iters"]]
            ctl.append(f"{leg}: {last[0][1] if last else '?'} of {sum(len(x['eig']) for x in bl)} eigenvalues not converged at ethr {last[0][2] if last else '?'} in the final iteration {rows[leg]['iters']} "
                       f"(warned in iterations {sorted(set(s[0] for s in seq))}); max |delta eig| vs {name} {d*1e3:.1f} meV")
        same = {f"{a}/{b}": R[a]["_extra"]["c_bands_seq"] == R[b]["_extra"]["c_bands_seq"] for a, b in (("R4_cpu", "R4_gpu"), ("R6_cpu", "R6_gpu")) if a in rows and b in rows}
        ctl_ok = ctl_ok and all(same.values())
        checks.add("final-iteration c_bands warning at 1 rank (nbnd = 160 cuts a near-degenerate group; hidden at 4 ranks by pool-0-only printing); gated on its control", ctl_ok,
                   (("; ".join(ctl) + f"; warning sequence (iteration, count, ethr) identical cpu vs gpu: {same}; reference R6 (QE 7.4.1, OpenBLAS, 4 ranks) final ethr {R['R6']['_extra']['ethr_final']}, "
                     f"warned in iterations {sorted(set(s[0] for s in R['R6']['_extra']['c_bands_seq']))} only") if warned else f"no final-iteration warning in {frz_legs}; cpu vs gpu sequences identical: {same}"),
                   "control (the gate): every printed eigenvalue equals the reference's to 1e-4 eV and cpu and gpu modes warn identically (same iterations, counts and ethr). Cause "
                   "(attributed on 2026-10-18 from one fresh QE develop checkout 4a626b7ce built upstream / + apple-bottom / + cegterg patch, R6 at 1 rank, cpu and gpu: all identical): at "
                   "k = (0,-1/2,-1/2) bands 157-168 lie within 0.1 meV at 9.508 eV, so nbnd = 160 splits a near-degenerate group and a few of its states lag at ethr ~1e-12; with -nk 4 "
                   "only pool 0 (Gamma) prints, which hides it (reordering the k-points brings it back; nbnd = 168 gives a clean final iteration). The patch, the BLAS library and GPU "
                   "numerics have no effect. Energy, forces and stress parity are gated separately below")
        S["c_bands"] = {leg: {"final_warning": cbw[leg], "sequence": R[leg]["_extra"]["c_bands_seq"], "ethr_final": R[leg]["_extra"]["ethr_final"]} for leg in rows}
    # nbnd = 168 control (closure of the attribution): the same R6 at 1 rank with the near-degenerate group complete
    if ctl168 and not ctl168.get("skipped"):
        ex = parse_extra(ctl168["out"]); seq = ex["c_bands_seq"]; last_it = ctl168["iterations"]
        final = [sq for sq in seq if sq[0] == last_it]
        e_ref = R["R6"]["result"]["total_energy_Ry"]; e_ctl = ctl168["energy_Ry"]
        checks.add(f"nbnd = {NBND_CONTROL} control (fresh-develop upstream pw.x, 1 rank, machlock bench): no c_bands warning in the final SCF iteration",
                   ctl168["rc"] == 0 and last_it is not None and not final and not ex["c_bands_final_warning"],
                   f"rc {ctl168['rc']}; {len(seq)} warnings in iterations {sorted(set(sq[0] for sq in seq))}; final iteration {last_it}: {'none' if not final else final}; final ethr {ex['ethr_final']}",
                   "none in the last iteration (mid-SCF warnings are benign). The expectation came from the audit session's 4-rank reordered nbnd = 168 run, where only "
                   "pool 0 prints, so it covered one k-point; at 1 rank the warning aggregates all four k-points and names none. Kept red when it fails: the band-count "
                   "explanation is then incomplete for the 1-rank layout (which states lag is not identified by this run)")
        checks.add(f"nbnd = {NBND_CONTROL} control: same total energy as the 4-rank R6 at printed precision", e_ctl is not None and e_ref is not None and abs(e_ctl - e_ref) <= 1e-8 + 1e-12,
                   f"{e_ctl} vs R6 {e_ref} Ry (dE {(e_ctl - e_ref) if (e_ctl is not None and e_ref is not None) else float('nan'):+.1e})", "|dE| <= 1e-8 Ry (one printed unit; the eight extra bands are empty)")
        r6 = R["R6_cpu"]["_extra"] if "R6_cpu" in R else None
        f160 = [sq for sq in r6["c_bands_seq"] if sq[0] == rows["R6_cpu"]["iters"]] if r6 else []
        checks.report(f"nbnd = {NBND_CONTROL} vs nbnd = 160 at 1 rank: states still unconverged in the final SCF iteration",
                      f"nbnd 160 (frozen build, cpu == gpu): {f160[0][1] if f160 else '?'} of 640 at ethr {f160[0][2] if f160 else '?'}; nbnd {NBND_CONTROL} (upstream develop): "
                      f"{final[0][1] if final else 0} of {4 * NBND_CONTROL} at ethr {final[0][2] if final else ex['ethr_final']}; energies identical",
                      "the count drops when the (0,-1/2,-1/2) group is completed but does not vanish at 1 rank; ethr ~1e-12 is where Davidson's residual test sits near the float64 "
                      "floor for a 64-atom cell at conv_thr 1e-9, so the remaining states may be a threshold-floor effect rather than another cut group (not tested here)")
        blk = [bb for bb in ex["bands"] if sorted(abs(x) for x in bb["k"]) == [0.0, 0.5, 0.5]]
        if blk and len(blk[0]["eig"]) >= NBND_CONTROL:
            e = blk[0]["eig"]; grp = e[156:168]
            checks.report(f"nbnd = {NBND_CONTROL} control: the near-degenerate group that nbnd = 160 cut, at k = {blk[0]['k']} (2pi/alat)",
                          f"bands 157-168 span {(max(grp) - min(grp)) * 1e3:.2f} meV around {float(np.mean(grp)):.4f} eV; gap to band 156 below: {(e[156] - e[155]) * 1e3:.1f} meV",
                          "bands 157-160 were the last four of nbnd = 160 and sat inside this group; with the group complete Davidson converges it (audit session, 2026-10-18)")
        checks.report(f"nbnd = {NBND_CONTROL} control provenance",
                      f"{ctl168['binary']} sha256 {ctl168['binary_sha256'][:12]} ({ctl168.get('binary_desc')}); rc {ctl168['rc']}, {ctl168['iterations']} iterations, "
                      f"wall {ctl168['wall_s']:.0f} s (observation; lock {ctl168['lock']}, load1 {ctl168['load1_at_start']:.2f} at start)",
                      "launched by this driver through exlib.run under machlock bench (the binary is not in abrun's catalog); 7.4.1 and the frozen build are context only")
        S.setdefault("c_bands", {})["R6n168_control"] = {"final_warning": bool(final), "sequence": seq, "ethr_final": ex["ethr_final"], "energy_Ry": e_ctl}
    elif ctl168:
        checks.report(f"nbnd = {NBND_CONTROL} control", ctl168["skipped"], "needs ~/qe-test/q-e-develop-20261018/bin-variants/pw.x.upstream (or MUPPET_QE_DEVELOP_UPSTREAM)")
    # FFT grids (the pitfall: the fold needs the supercell grid to be exactly twice the parent's)
    fft_ok = rows["R2"]["fft"] == [36] * 3 and rows["R3"]["fft"] == [36] * 3 and rows["R4"]["fft"] == [72] * 3 and rows["R6"]["fft"] == [72] * 3 and rows["R2p"]["fft"] == [48] * 3
    checks.add("FFT grids pinned and commensurate", fft_ok, f"R2 {rows['R2']['fft']}, R3 {rows['R3']['fft']}, R4 {rows['R4']['fft']}, R6 {rows['R6']['fft']}, R2' {rows['R2p']['fft']}, R1 {rows['R1']['fft']}, R1' {rows['R1p']['fft']}",
               "36^3 (8 atoms), 72^3 (64 atoms), 48^3 (R2'); R1 is never commensurate")

    # 1. exact fold. The Bloch identity is gated on the density-dependent energy (total minus Ewald == one-electron + Hartree + xc +
    # one-centre PAW); the total-energy fold carries ewald.f90's real-space truncation (see the Ewald controls) and is reported.
    ec = ewald_controls()
    S["ewald_controls"] = ec
    c2, c4 = ec["R2"], ec["R4"]
    pairs = (("R2", "R4"), ("R3", "R6"))

    def tsum(leg):
        return sum(term(leg, k) for k in ("one_electron", "hartree", "xc", "paw")) / NAT[leg]

    dsc = {f"{a}/{b}": abs((E(b) - term(b, "ewald")) / NAT[b] - (E(a) - term(a, "ewald")) / NAT[a]) for a, b in pairs}
    dsum = {f"{a}/{b}": abs(tsum(b) - tsum(a)) for a, b in pairs}
    # total - Ewald - (one-electron + Hartree + xc + one-centre PAW) is descf (the SCF correction term of the PAW total energy) plus print rounding
    descf = {leg: E(leg) - term(leg, "ewald") - sum(term(leg, k) for k in ("one_electron", "hartree", "xc", "paw")) for leg in rows}      # Ry per cell
    for a, b in pairs:
        k = f"{a}/{b}"
        checks.add(f"exact fold {b}/64 == {a}/8 (density-dependent energy: total minus Ewald)", ok(dsc[k]) and dsc[k] < FOLD_GATE,
                   f"{dsc[k]:.2e} Ry/atom (E({b}) = {E(b):.8f}, E({a}) = {E(a):.8f} Ry); the sum of the four printed terms one-electron + Hartree + xc + one-centre PAW folds to {dsum[k]:.2e} Ry/atom "
                   f"(total - Ewald - those four = descf + print rounding: {descf[a]:+.1e} Ry per cell on {a}, {descf[b]:+.1e} on {b})",
                   f"< {FOLD_GATE:.0e} Ry/atom: Bloch theorem on a commensurate 2x2x2 supercell (same density, same SCF energy per atom). Tolerance: one printed ulp (1e-8 Ry per cell) on each of the "
                   f"two printed values (total, Ewald) of both cells, 2 x 1e-8/8 + 2 x 1e-8/64 = {FOLD_TOL:.1e} Ry/atom, rounded up; the SCF residual (conv_thr 1e-10 and 1e-9 per cell) adds < 2e-11")
    d24 = abs(E("R4") / 64 - E("R2") / 8)
    d36 = abs(E("R6") / 64 - E("R3") / 8)
    ew24 = term("R4", "ewald") / 64 - term("R2", "ewald") / 8
    ew36 = term("R6", "ewald") / 64 - term("R3", "ewald") / 8
    dshell = c4["shell_per_atom"] - c2["shell_per_atom"]
    ewald_note = (f"ewald.f90 picks alpha from its G-sum bound (<= 1e-7 Ry per cell: {c2['alpha']:.1f} on the 8-atom cell, {c4['alpha']:.1f} on 64 atoms) and truncates the real-space sum at "
                  f"rmax = 4/sqrt(alpha); on the 64-atom cells at ecutrho {ECUTRHO:.0f} that rmax ({c4['rmax_bohr']:.3f} bohr) is below the nearest-neighbour distance ({c4['d_nn_bohr']:.3f} bohr), so the printed "
                  f"Ewald misses the erfc shell by {c4['shell_per_atom']:.1e} Ry/atom ({c4['shell_per_atom'] * 64:.1e} Ry per cell); the 8-atom cells (rmax {c2['rmax_bohr']:.2f} bohr, also below d_nn) miss only "
                  f"{c2['shell_per_atom']:.1e} Ry/atom because erfc(sqrt(alpha) d_nn) is {c2['erfc_sqrt_alpha_d_nn']:.1e} there against {c4['erfc_sqrt_alpha_d_nn']:.1e}")
    checks.report("total-energy fold R4/64 vs R2/8 (carries the Ewald rule's truncation; its explanation is gated by the Ewald controls a, b2, b3)",
                  f"{d24:.2e} Ry/atom = density part {dsc['R2/R4']:.2e} + Ewald term {ew24:+.2e}; the Ewald part is -(shell dropped on 64 atoms {c4['shell_per_atom']:.2e} - shell dropped on 8 atoms "
                  f"{c2['shell_per_atom']:.2e}) = {-dshell:+.2e}, remainder {ew24 + dshell:+.1e} Ry/atom",
                  f"gated < 2e-8 in the first pass (results/run-first-pass.log), then < 1e-7; now reported. {ewald_note}")
    checks.report("total-energy fold R6/64 vs R3/8 (carries the Ewald rule's truncation; same cells as R4/R2)",
                  f"{d36:.2e} Ry/atom = density part {dsc['R3/R6']:.2e} + Ewald term {ew36:+.2e}; Ewald part -(shell64 - shell8) = {-dshell:+.2e}, remainder {ew36 + dshell:+.1e} Ry/atom",
                  "same Ewald rule as R4/R2 (R6 is the same cell as R4 and R3 the same cell as R2: same alpha, same rmax, same dropped shells)")
    dd = abs((E("R4") - E("R6")) / 64 - (E("R2") - E("R3")) / 8)
    checks.add("difference of differences (Gamma-only sampling error folds too)", ok(dd) and dd < FOLD_GATE,
               f"{dd:.2e} Ry/atom; sampling error {(E('R4') - E('R6')) / 64 * 1e3:+.4f} mRy/atom (campaign: B-C = 0.5003 Ry = 7.8 mRy/atom)",
               f"< {FOLD_GATE:.0e} Ry/atom: the Ewald truncation is the same on R4 and R6 (one cell: same alpha and rmax) and on R2 and R3, so it cancels; tolerance one printed ulp on each of the "
               f"four totals, 2 x 1e-8/8 + 2 x 1e-8/64 = {FOLD_TOL:.1e}, rounded up")
    S["fold"] = {"d_R4_R2_per_atom": d24, "d_R6_R3_per_atom": d36, "diff_of_diffs": dd, "gamma_sampling_error_Ry_per_atom": (E("R4") - E("R6")) / 64, "density_only": dsc, "sum_of_terms": dsum,
                 "descf_per_cell": descf, "gate": FOLD_GATE, "print_bound": FOLD_TOL}

    # 2. eigenvalue unfolding R4 (Gamma) vs R2 (4 irreducible points with multiplicities)
    x2, x4 = R["R2"]["_extra"], R["R4"]["_extra"]
    mult = [int(round(4 * k["wk"])) for k in x2["klist"]]
    checks.add("R2 reports 4 irreducible k-points with multiplicities 1,3,3,1", x2["nks"] == 4 and sorted(mult) == [1, 1, 3, 3] and sum(mult) == 8,
               f"nks {x2['nks']}, wk {[k['wk'] for k in x2['klist']]} -> {mult}", "24 ops + time reversal on the Gamma-centred 2x2x2 mesh")
    unf = {}
    if x2["bands"] and x4["bands"] and len(x2["bands"]) == len(mult) and all(len(b["eig"]) >= 21 for b in x2["bands"]):
        estar = min(b["eig"][20] for b in x2["bands"])
        ecut = estar - UNFOLD_MARGIN
        P = sorted(e for b, m in zip(x2["bands"], mult) for e in b["eig"] if e < ecut for _ in range(m))
        Sg = sorted(e for e in x4["bands"][0]["eig"] if e < ecut)
        n = min(len(P), len(Sg))
        deltas = np.abs(np.array(P[:n]) - np.array(Sg[:n])) if n else np.array([np.inf])
        worst = int(np.argmax(deltas))
        checks.add("unfolding: R4 Gamma spectrum == R2 mesh eigenvalues with multiplicity below E*", len(P) == len(Sg) and n > 0 and float(deltas.max()) < EV_TOL,
                   f"{len(Sg)} supercell vs {len(P)} parent states below E* = {estar:.4f} eV (cut {UNFOLD_MARGIN*1e3:.0f} meV lower); max |delta| = {deltas.max()*1e3:.3f} meV at sorted index {worst} ({Sg[worst] if worst < len(Sg) else float('nan'):.4f} vs {P[worst] if worst < len(P) else float('nan'):.4f} eV)",
                   "same count and max |delta| < 1 meV (E* = min_k eps_21(k); printed precision 0.1 meV)")
        homo2, lumo2 = max(b["eig"][15] for b in x2["bands"]), min(b["eig"][16] for b in x2["bands"])
        homo4, lumo4 = x4["bands"][0]["eig"][127], x4["bands"][0]["eig"][128]
        checks.add("HOMO/LUMO of R4 equal max_k eps_16 / min_k eps_17 of R2", abs(homo4 - homo2) < EV_TOL and abs(lumo4 - lumo2) < EV_TOL,
                   f"R4 {homo4:.4f}/{lumo4:.4f} eV (printed {rows['R4']['homo_eV']}/{rows['R4']['lumo_eV']}); R2 {homo2:.4f}/{lumo2:.4f} eV; gap {lumo4 - homo4:.4f} eV", "< 1 meV (128 occupied bands = 8 x 16)")
        levels = sorted(set(round(e, 4) for e in P))
        bad = []
        for lv in levels:
            cP = sum(1 for e in P if abs(e - lv) <= EV_TOL)
            cS = sum(1 for e in Sg if abs(e - lv) <= EV_TOL)
            if cP != cS:
                bad.append((lv, cP, cS))
        checks.add("degeneracy count: R4 states within 1 meV of each parent level == weight x parent degeneracy", not bad and len(levels) > 0,
                   f"{len(levels)} distinct parent levels; mismatches {bad[:3]}", "every level matches")
        # the same comparison up to the last computed parent band: complete below min_k eps_24(k) as long as the supercell's nbnd covers it
        e24 = min(b["eig"][-1] for b in x2["bands"]) - UNFOLD_MARGIN
        P24 = sorted(e for b, m in zip(x2["bands"], mult) for e in b["eig"] if e < e24 for _ in range(m))
        S24 = sorted(e for e in x4["bands"][0]["eig"] if e < e24)
        n24 = min(len(P24), len(S24))
        d24s = float(np.abs(np.array(P24[:n24]) - np.array(S24[:n24])).max()) if n24 else float("nan")
        top4 = x4["bands"][0]["eig"][-1]
        checks.add("unfolding up to the last computed parent band: R4 Gamma spectrum == R2 mesh eigenvalues below min_k eps_24 (conduction states included)",
                   len(P24) == len(S24) and n24 > 0 and top4 > e24 and d24s < EV_TOL,
                   f"{len(S24)} supercell vs {len(P24)} parent states below min_k eps_24 - {UNFOLD_MARGIN*1e3:.0f} meV = {e24:.4f} eV; max |delta| = {d24s*1e3:.3f} meV; highest computed supercell state "
                   f"{top4:.4f} eV (above the cut: the window is complete on both sides)",
                   "same count and max |delta| < 1 meV; E* = min_k eps_21 equals the LUMO here (the sixfold X1c manifold folds onto Gamma), so the E* window holds the valence manifold only and "
                   "this one adds the conduction states (complete while the supercell's nbnd = 160 exceeds the count)")
        unf = {"E_star_eV": estar, "cut_eV": ecut, "n_parent": len(P), "n_supercell": len(Sg), "max_delta_meV": float(deltas.max()) * 1e3, "worst_index": worst,
               "homo_lumo_R2": [homo2, lumo2], "homo_lumo_R4": [homo4, lumo4], "parent_sorted": P, "supercell_sorted": Sg, "multiplicities": mult, "n_levels": len(levels),
               "window24": {"cut_eV": e24, "n_parent": len(P24), "n_supercell": len(S24), "max_delta_meV": d24s * 1e3},
               "parent_per_k": [{"k": b["k"], "mult": m, "eig": b["eig"]} for b, m in zip(x2["bands"], mult)], "supercell_eig": x4["bands"][0]["eig"]}
    else:
        checks.add("unfolding: eigenvalue blocks parsed", False, f"R2 blocks {len(x2['bands'])}, R4 blocks {len(x4['bands'])}", "4 x 24 and 1 x 160 eigenvalues")
    S["unfolding"] = unf

    # 3. the Ewald term. (a) a replica of ewald.f90 reproduces every printed value; (b1) the converged lattice sum is one number;
    # (b2) printed == converged minus the two explicitly summed truncations, every cell; (b3) on the fold cells the dropped
    # real-space shell alone accounts for the deviation. The original 'identical across R2-R6' comparison is kept as a reported finding.
    ew = {leg: term(leg, "ewald") / NAT[leg] for leg in ("R2", "R3", "R4", "R5", "R6") if leg in rows}
    rep = {n: abs(ec[n]["qe_rule_per_atom"] - term(n, "ewald") / NAT[n]) for n in ec if n in rows}
    tol_a = {n: 5e-9 / ec[n]["nat"] + REPLICA_FLOOR for n in rep}
    floor_note = (f"{REPLICA_FLOOR:.0e} Ry/atom is the replica's own floor: float64 summation (b1's alpha spread {max(ec['R2']['converged_by_alpha'].values()) - min(ec['R2']['converged_by_alpha'].values()):.1e}) "
                  f"and the pieces beyond its own limits (the G tail beyond 300 Ry is bounded by ewald.f90's formula at <= {max(ec[n]['gtail_beyond_bound_per_atom'] for n in ec):.1e} Ry/atom over these cells, "
                  f"the real-space shell beyond 20 bohr by erfc(20 sqrt(alpha)) < 1e-150)")
    checks.add("Ewald control (a): a replica of ewald.f90's rule (alpha search, G-sphere of ecutrho, rmax = 4/sqrt(alpha)) reproduces every printed Ewald", all(ok(v) and v < tol_a[n] for n, v in rep.items()),
               "; ".join(f"{n} alpha {ec[n]['alpha']:.1f} rmax {ec[n]['rmax_bohr']:.2f} bohr (d_nn {ec[n]['d_nn_bohr']:.2f}): {ec[n]['qe_rule_per_atom']:.10f} vs printed {term(n, 'ewald') / NAT[n]:.10f} "
                         f"(|dev| {rep[n]:.1e}, tol {tol_a[n]:.1e})" for n in rep),
               f"|replica - printed| < 5e-9/nat + {REPLICA_FLOOR:.0e} Ry/atom on every cell: 5e-9/nat is the half-ulp of the printed per-cell value (1e-8 Ry) per atom, so a replica equal to QE's internal "
               f"value can miss the printed one by up to that; {floor_note}")
    cba = ec["R2"]["converged_by_alpha"]; conv = ec["R2"]["converged_per_atom"]; dalpha = max(cba.values()) - min(cba.values())
    checks.add("Ewald control (b1): the converged lattice sum (G to 300 Ry, R to 20 bohr) is one number, independent of alpha", ok(dalpha) and dalpha < 1e-14,
               f"{conv:.12f} Ry/atom on the 8-atom cell; alpha 0.6 / 1.0 / 1.5 spread {dalpha:.1e}; converged per cell: " + ", ".join(f"{n} {ec[n]['converged_per_atom']:.10f}" for n in ec),
               "< 1e-14 Ry/atom (the Ewald energy does not depend on the splitting parameter; float64 ulp of 8.45 is 1.8e-15)")
    clos = {}
    for leg in EWALD_CELL:
        if leg in rows:
            c = ec[EWALD_CELL[leg]]; p = term(leg, "ewald") / NAT[leg]
            clos[leg] = {"cell": EWALD_CELL[leg], "printed": p, "dev": p - conv, "shell": c["shell_per_atom"], "gtail": c["gtail_per_atom"], "resid_shell_only": p - conv + c["shell_per_atom"],
                         "resid_full": p - conv + c["shell_per_atom"] + c["gtail_per_atom"], "tol": 5e-9 / NAT[leg] + REPLICA_FLOOR,
                         "tol_shell_only": c["gtail_per_atom"] + 5e-9 / NAT[leg] + REPLICA_FLOOR}
    checks.add("Ewald control (b2): printed == converged - dropped real-space shell (r > rmax) - dropped G tail (|G|^2 > ecutrho), every cell",
               bool(clos) and all(ok(v["resid_full"]) and abs(v["resid_full"]) < v["tol"] for v in clos.values()),
               "; ".join(f"{n} ({ec[n]['nat']} atoms): dev {clos[n]['dev']:+.2e} = -(shell {clos[n]['shell']:.1e} [{ec[n]['shell_pairs']} pairs] + G tail {clos[n]['gtail']:.1e} [{ec[n]['gtail_nG']} G; ewald.f90 bound {ec[n]['gsum_bound_per_atom']:.1e}]) "
                         f"+ {clos[n]['resid_full']:+.1e} (tol {clos[n]['tol']:.1e})" for n in ec if n in clos)
               + "; residual " + ", ".join(f"{l} {clos[l]['resid_full']:+.1e}" for l in ("R3", "R5", "R6") if l in clos) + " (same cells as R2, R4, R4)",
               f"|residual| < 5e-9/nat + {REPLICA_FLOOR:.0e} Ry/atom (both dropped pieces are direct sums over the dropped regions, Ry with e^2 = 2; 5e-9/nat is the printed half-ulp of the per-cell value per atom; {floor_note})")
    fold_cells = [l for l in ("R2", "R3", "R4", "R5", "R6") if l in clos]
    checks.add("Ewald control (b3): on the fold cells the dropped real-space shell alone accounts for the deviation from the converged sum",
               bool(fold_cells) and all(abs(clos[l]["resid_shell_only"]) < clos[l]["tol_shell_only"] for l in fold_cells),
               "; ".join(f"{l} dev {clos[l]['dev']:+.2e} + shell {clos[l]['shell']:.2e} = {clos[l]['resid_shell_only']:+.1e} (tol {clos[l]['tol_shell_only']:.1e})" for l in fold_cells)
               + "; shell alone on the other cells: " + ", ".join(f"{l} {clos[l]['resid_shell_only']:+.1e} (= the G tail {clos[l]['gtail']:.1e}, below ewald.f90's bound {ec[l]['gsum_bound_per_atom']:.1e})" for l in ("R1", "R1p", "R2p") if l in clos),
               f"|printed - converged + shell| < G tail + 5e-9/nat + {REPLICA_FLOOR:.0e} Ry/atom on R2-R6 (what is left after the shell is the G tail, <= 3.9e-10 there because alpha <= 1.1, plus the print half-ulp and the "
               "replica floor); on the 2-atom cells the rule's larger alpha (1.3, 2.6) makes the shell negligible and the G-sphere truncation the whole deviation, so the shell alone cannot close them (reported here, closed by b2)")
    sp = (max(ew.values()) - min(ew.values())) if ew and all(ok(v) for v in ew.values()) else float("nan")
    checks.report("Ewald per atom across R2-R6 (was gated < 1e-8 in the first pass)", f"spread {sp:.2e} Ry/atom ({', '.join(f'{k} {v:.9f}' for k, v in ew.items())})",
                  f"the spread is the dropped real-space shell, {c4['shell_per_atom']:.1e} on the 64-atom cells against {c2['shell_per_atom']:.1e} on the 8-atom cells (alpha {c4['alpha']:.1f} vs {c2['alpha']:.1f}, "
                  f"rmax {c4['rmax_bohr']:.2f} vs {c2['rmax_bohr']:.2f} bohr, both below d_nn {c4['d_nn_bohr']:.3f} bohr); the converged lattice sum is one number (b1, b2)")
    d1 = abs(term("R1", "ewald") / 2 - ew.get("R2", float("nan")))
    checks.add("Ewald per atom of the fcc primitive equals the cubic cell's", ok(d1) and d1 < 1e-7, f"{d1:.2e} Ry/atom (R1 {term('R1', 'ewald') / 2:.9f})", "< 1e-7 Ry/atom (ewald.f90's G-sum bound is 1e-7 Ry per cell)")
    S["ewald_closure"] = clos
    # per-term identities: an individual term is first order in the density residual while the total is second order (variational).
    # Control: the same cell at the same conv_thr on the two Davidson algorithms (R4 vs R5), on two QE builds (R4 vs R4_cpu),
    # and on one binary cpu vs gpu (R4_cpu vs R4_gpu).
    ctrl = {}
    for a, b in (("R4", "R5"), ("R4", "R4_cpu"), ("R4_cpu", "R4_gpu"), ("R6", "R6_cpu"), ("R6_cpu", "R6_gpu")):
        if a in rows and b in rows:
            ctrl[f"{a}/{b}"] = {"max_term": max(abs(term(a, k) - term(b, k)) / 64 for k in ("paw", "hartree", "xc", "one_electron")), "total": abs(E(a) - E(b)) / 64}
    scat = max((v["max_term"] for v in ctrl.values()), default=float("nan"))
    tt = {}
    for key in ("paw", "hartree", "xc", "one_electron"):
        for a, b in pairs:
            d = abs(term(a, key) / NAT[a] - term(b, key) / NAT[b])
            tt[f"{key}:{a}/{b}"] = d
            checks.add(f"{key} per atom {a}/{b} (first-order term)", ok(d) and d < 1e-6, f"{d:.2e} Ry/atom ({a} {term(a, key) / NAT[a]:.9f}, {b} {term(b, key) / NAT[b]:.9f})",
                       f"< 1e-6 Ry/atom, an empirical tolerance ({1e-6 / scat if scat else float('nan'):.1f}x the largest same-cell scatter in the control below, {scat:.1e} Ry/atom for R4 vs R5; not derived from "
                       f"conv_thr; was 2e-8 in the first pass). Physics: an individual term is first order in the density residual while the total and the sum of the four terms (gated at "
                       f"{FOLD_GATE:.0e} in the fold above) are variational, second order")
    checks.report("term scatter control: same cell, same conv_thr, other SCF path",
                  "; ".join(f"{k}: max term {v['max_term']:.1e}, total {v['total']:.1e} Ry/atom" for k, v in ctrl.items()),
                  "the per-term pair differences above are of the size of this SCF-convergence scatter of first-order terms (conv_thr 1e-9 per cell), not a fold defect; the totals agree to the printed digit")
    S["terms"] = {"ewald_per_atom": ew, "ewald_spread": sp, "ewald_R1_vs_R2": d1, "pair_diffs": tt, "scatter_control": ctrl, "per_run": {leg: rows[leg]["terms"] for leg in rows}}

    # 4. cubic vs primitive: the fcc 24^3 and cubic 36^3 real-space grids are not commensurate, so the GGA xc energy density is
    # sampled on different point sets (aliasing); the physics gate is that the aliasing shrinks with the denser grid at ecutrho 200
    dc = abs(E("R1") / 2 - E("R2") / 8)
    dc2 = abs(E("R1p") / 2 - E("R2p") / 8)
    pr = {l: rows[l]["pressure_kbar"] for l in ("R1", "R2", "R1p", "R2p")}
    dp100 = abs(pr["R1"] - pr["R2"]) if None not in (pr["R1"], pr["R2"]) else float("nan")
    dp200 = abs(pr["R1p"] - pr["R2p"]) if None not in (pr["R1p"], pr["R2p"]) else float("nan")
    checks.report("fcc primitive vs cubic cell per atom (non-commensurate xc grids; was gated < 1e-5 in the first pass)",
                  f"E(R1)/2 - E(R2)/8 = {E('R1') / 2 - E('R2') / 8:+.2e} Ry/atom at ecutrho 100 (R1 FFT {rows['R1']['fft']}, R2 {rows['R2']['fft']}), {E('R1p') / 2 - E('R2p') / 8:+.2e} at 200 (the sign flips); "
                  f"pressure signature: R1 {pr['R1']} vs R2 {pr['R2']} kbar ({dp100:.2f} kbar apart) at ecutrho 100, R1' {pr['R1p']} vs R2' {pr['R2p']} kbar ({dp200:.2f} apart) at 200",
                  "consistent with xc-grid aliasing: the fcc 24^3 and cubic 36^3 grids are not commensurate (the primitive cell's points are not a sub-lattice of the cubic cell's), so the GGA xc energy "
                  "density is sampled on different point sets; the attribution is by elimination (the ecutrho-200 control changes the density G-sphere and the FFT grids together; a grid-only control, R2 at "
                  "nr = 48 with ecutrho 100, was not run). The old < 1e-5 gate assumed commensurate grids")
    checks.add("consistent with xc-grid aliasing: the discrepancy at ecutrho 200 (R1' vs R2') is smaller by >= 2x", ok(dc2) and dc2 > 0 and dc / dc2 >= 2,
               f"{dc:.2e} -> {dc2:.2e} Ry/atom (ratio {dc / dc2 if dc2 else float('nan'):.1f}; R1' FFT {rows['R1p']['fft']}, R2' {rows['R2p']['fft']}); pressure difference {dp100:.2f} -> {dp200:.2f} kbar "
               f"(ratio {dp100 / dp200 if dp200 else float('nan'):.1f})",
               "ratio >= 2: both grids are 1.5x denser per direction at ecutrho 200, and the grid-sum error of a smooth integrand falls faster than the spacing, so 2x is a conservative floor; the pressure "
               "pair is reported beside the energy pair, not gated")
    S["cubic_vs_primitive"] = {"d100": dc, "d200": dc2, "signed_d100": E("R1") / 2 - E("R2") / 8, "signed_d200": E("R1p") / 2 - E("R2p") / 8, "pressure": pr, "dp100": dp100, "dp200": dp200}

    # 5. Gamma trick vs complex algorithm
    if "R5" in rows:
        d45 = rnd(E("R4") - E("R5"))
        acc = [(R[l]["result"].get("scf_accuracy") or [None])[-1] for l in ("R4", "R5")]
        checks.add("Gamma trick (R4) == complex algorithm (R5) total energy", ok(d45) and d45 <= 2e-8 + 1e-15,
                   f"{d45:.1e} Ry (R4 {E('R4'):.8f}, R5 {E('R5'):.8f}; final estimated scf accuracy {acc[0]} / {acc[1]} Ry, conv_thr 1e-9)",
                   "<= 2e-8 Ry: two units of the printed precision (1e-8 Ry each, one per printed total; was 1e-8 in the first pass); the SCF residual at conv_thr 1e-9 per cell (estimated scf accuracy "
                   "below 1e-9 on both) is under one unit; the real regterg and the complex cegterg solve the same Kohn-Sham problem")
        e4, e5 = x4["bands"][0]["eig"] if x4["bands"] else [], R["R5"]["_extra"]["bands"][0]["eig"] if R["R5"]["_extra"]["bands"] else []
        de = max((abs(a - b) for a, b in zip(e4, e5)), default=float("inf")) if len(e4) == len(e5) == 160 else float("inf")
        checks.add("all 160 Gamma eigenvalues equal (R4 vs R5)", de <= PRINT_EV + 1e-9, f"max |delta| = {de*1e3:.2f} meV over {len(e4)} states", "<= 1e-4 eV (printed precision)")
        checks.add("R4 uses the Gamma-point algorithms and R5 does not", x4["gamma_only"] and not R["R5"]["_extra"]["gamma_only"], f"R4 {x4['gamma_only']}, R5 {R['R5']['_extra']['gamma_only']}", "'gamma-point specific algorithms are used' only in R4")
        checks.report("SCF iteration counts R4 vs R5", f"{rows['R4']['iters']} vs {rows['R5']['iters']}", "expected equal within 1; informational")
        S["gamma_vs_complex"] = {"dE": d45, "max_deig_eV": de, "iters": [rows["R4"]["iters"], rows["R5"]["iters"]]}

    # 6. symmetry of the ideal lattice
    # Forces. Under the group QE keeps, only the Td-site atoms (td_sites) have a force that vanishes identically; the others carry a
    # symmetry-allowed residual that is a convergence quantity, gated against QE's own estimate of the SCF force error ('Total SCF correction').
    fmax = {leg: rows[leg]["max_force"] for leg in ("R1", "R2", "R3", "R4", "R5", "R6") if leg in rows}
    fs = {leg: rows[leg]["force_split"] for leg in ("R1", "R2", "R3", "R4", "R5", "R6") if leg in rows and rows[leg]["force_split"]}
    checks.add("forces: Td-site atoms exactly zero (R1 both atoms; R2/R3 the 4 and R4-R6 the 8 atoms at (i/2, j/2, k/2))", len(fs) >= 5 and all(v["td_max"] == 0.0 for v in fs.values()),
               ", ".join(f"{k} max |F| over its {v['n_td']} Td sites {v['td_max']:.1e}" for k, v in fs.items()) + " Ry/bohr",
               "exactly 0.0 as printed: these sites are fixed by every operation QE keeps (R1: 48 ops with inversion and fractional translations, so both atoms; cubic cells: 'This is a supercell, "
               "fractional translations are disabled' leaves the 24 point operations of Td about the origin, which fix only the atoms at (i/2, j/2, k/2)), so the symmetrized force vanishes identically")
    conv = {}
    for leg, v in fs.items():
        if leg == "R1":
            continue
        c = rows[leg]["total_scf_correction"]
        tol = max(c if c is not None else 0.0, 5e-7)                   # 'Total SCF correction' is printed to 1e-6 Ry/bohr: 0.000000 means < 5e-7
        conv[leg] = {"other_max": v["other_max"], "n_nonzero": v["n_nonzero"], "total_scf_correction": c, "tol": tol, "warning": rows[leg]["scf_correction_warning"], "ok": v["other_max"] <= tol}
    checks.add("forces: symmetry-allowed residuals on the other atoms are below QE's own SCF-correction estimate (convergence check, R2-R6)", bool(conv) and all(v["ok"] for v in conv.values()),
               "; ".join(f"{k} max |F| {v['other_max']:.1e} ({v['n_nonzero']} atoms with a nonzero printed force) vs Total SCF correction {v['total_scf_correction']:.1e}"
                         + (" [QE: 'SCF correction compared to forces is large: reduce conv_thr']" if v["warning"] else "") for k, v in conv.items()) + " Ry/bohr",
               "max |F| over the non-Td atoms <= max(Total SCF correction, 5e-7) Ry/bohr. Under the 24 point operations those sites have C3v (sublattice 2: [111]-type residual) or Cs (face-centre "
               "type: residual in a mirror plane) site symmetry, so a residual is symmetry-allowed; it is gated against QE's printed estimate of the force error from incomplete SCF convergence "
               "(PW/src/forces.f90, 'Total SCF correction', printed to 1e-6), and QE's warning that this estimate exceeds the forces themselves is reported. The first pass gated every component < 1e-6 "
               "as if forces vanished by symmetry on all atoms")
    aniso = {}
    for leg in ("R2", "R3", "R4", "R5", "R6"):
        s = rows.get(leg, {}).get("stress_kbar")
        if s:
            s = np.array(s)
            aniso[leg] = max(abs(s[0, 0] - s[1, 1]), abs(s[1, 1] - s[2, 2]), abs(s[0, 1]), abs(s[0, 2]), abs(s[1, 2]), abs(s[1, 0]), abs(s[2, 0]), abs(s[2, 1]))
        else:
            aniso[leg] = float("inf")
    checks.add("stress isotropic (R2-R6; symmetrization consistency)", all(v < 0.01 for v in aniso.values()), ", ".join(f"{k} {v:.3f}" for k, v in aniso.items()) + " kbar",
               "|sxx-syy|, |syy-szz|, off-diagonals < 0.01 kbar: QE's symmatrix under the cubic group makes the printed tensor isotropic by construction, so this row checks the parser and the "
               "symmetrization, not the physics; the physics is in the fold-pair pressure row")

    def P(leg):
        v = rows.get(leg, {}).get("pressure_kbar")
        return float("nan") if v is None else v

    pd = {f"{a}/{b}": abs(P(a) - P(b)) for a, b in (("R2", "R4"), ("R3", "R6")) if a in rows and b in rows}
    checks.add("pressure identical across each fold pair", all(ok(v) and v < 0.05 for v in pd.values()), ", ".join(f"{k} {v:.3f} kbar" for k, v in pd.items()) + f" (P: R2 {rows['R2']['pressure_kbar']}, R4 {rows['R4']['pressure_kbar']}, R3 {rows['R3']['pressure_kbar']}, R6 {rows['R6']['pressure_kbar']} kbar)",
               "< 0.05 kbar (same wavefunctions and density on a fold pair give the same stress; printed to 0.01 kbar)")
    checks.add("R1: 48 Sym. Ops., with inversion, 32 k-points as given", rows["R1"]["nsym"] == 48 and rows["R1"]["sym_note"] == "with inversion" and rows["R1"]["nks"] == 32,
               f"{rows['R1']['nsym']} ({rows['R1']['sym_note']}), {rows['R1']['nks']} k-points", "fcc primitive keeps the full diamond group; the explicit tpiba list is not reduced (nrot = nsym = 48)")
    for leg in ("R2", "R4", "R6"):
        checks.add(f"{leg}: supercell, 24 Sym. Ops. (no inversion)", rows[leg]["supercell"] and rows[leg]["nsym"] == 24 and rows[leg]["sym_note"] == "no inversion",
                   f"supercell {rows[leg]['supercell']}, {rows[leg]['nsym']} ({rows[leg]['sym_note']})", "fractional translations disabled by QE")
    checks.add("R2 and R6 report exactly 4 k-points", rows["R2"]["nks"] == 4 and rows["R6"]["nks"] == 4, f"R2 {rows['R2']['nks']}, R6 {rows['R6']['nks']} (R3 {rows['R3']['nks']}, R1 {rows['R1']['nks']})", "4 and 4")
    S["symmetry"] = {"max_force": fmax, "force_split": fs, "force_convergence": conv, "stress_anisotropy": aniso, "pressure_pair_diffs": pd}

    # 7. recorded reference energies
    for leg in ("R4", "R6"):
        d = abs(E(leg) - RECORDED[leg])
        vals = RECORDED_ALL[leg]; spread = max(vals.values()) - min(vals.values())
        checks.add(f"{leg} reproduces the recorded campaign energy", ok(d) and d < 2e-6,
                   f"{E(leg):.8f} vs {RECORDED[leg]:.8f} Ry: |dE| = {d:.1e} Ry (conv_thr here 1e-9, there 1e-6); every recorded value: " + "; ".join(f"{k} {v:.8f}" for k, v in vals.items())
                   + f" (spread {spread:.1e} Ry)",
                   f"< 2e-6 Ry; recorded by {RECORDED_SOURCE[leg]}; all at conv_thr 1e-6")

    # 8. apple-bottom parity and 9. dispatch evidence, 10. dimensions
    if ab_legs:
        parity = {}
        for name, cpu_leg, gpu_leg in (("R4", "R4_cpu", "R4_gpu"), ("R6", "R6_cpu", "R6_gpu")):
            if cpu_leg not in rows:
                continue
            rc = rows[cpu_leg]
            dref = abs(E(cpu_leg) - E(name))
            checks.add(f"{name}: ab_cpu vs reference (cross-build; two meanings of 'CPU only')", ok(dref) and dref < 1e-6,
                       f"{dref:.1e} Ry (ab_cpu {E(cpu_leg):.8f} on QE {rc['qe_version']} + Accelerate GEMMs vs reference {E(name):.8f} on QE 7.4.1 OpenBLAS)", "< 1e-6 Ry (energies only; never a wall comparison)")
            parity[name] = {"dE_abcpu_vs_ref": dref}
            if gpu_leg in rows:
                rg = rows[gpu_leg]
                d = rnd(E(cpu_leg) - E(gpu_leg))
                checks.add(f"{name}: ab_cpu == ab_gpu_all total energy (same pw.x)", ok(d) and d < 1e-8, f"{d:.1e} Ry (cpu {E(cpu_leg):.8f}, gpu {E(gpu_leg):.8f}; iterations {rc['iters']}/{rg['iters']})", "< 1e-8 Ry on one frozen binary (printed digits identical)")
                bc, bg = R[cpu_leg]["_extra"]["bands"], R[gpu_leg]["_extra"]["bands"]
                de = max((abs(a - b) for x, y in zip(bc, bg) for a, b in zip(x["eig"], y["eig"])), default=float("inf")) if bc and len(bc) == len(bg) else float("inf")
                nev = sum(len(x["eig"]) for x in bc)
                checks.add(f"{name}: all printed eigenvalues equal (cpu vs gpu)", de <= PRINT_EV + 1e-9, f"max |delta| = {de*1e3:.2f} meV over {nev} values", "<= 1e-4 eV")
                Fc, Fg = R[cpu_leg]["result"].get("forces"), R[gpu_leg]["result"].get("forces")
                dF = rnd(float(np.abs(np.array(Fc) - np.array(Fg)).max())) if Fc and Fg else float("inf")
                Sc, Sgp = R[cpu_leg]["result"].get("stress_kbar"), R[gpu_leg]["result"].get("stress_kbar")
                dS = abs(round(float(np.abs(np.array(Sc) - np.array(Sgp)).max()), 2)) if Sc and Sgp else float("inf")
                checks.add(f"{name}: forces and stress identical to printed precision (cpu vs gpu)", dF <= 1e-8 and dS <= 0.01, f"max |dF| = {dF:.1e} Ry/bohr, max |dsigma| = {dS:.2f} kbar", "<= 1e-8 Ry/bohr, <= 0.01 kbar")
                parity[name].update({"dE_cpu_gpu": d, "max_deig_eV": de, "max_dF": dF, "max_dstress": dS})
        S["parity"] = parity
        # dispatch evidence from the library's own counters
        disp = {}
        for leg, name, mode in ab_legs:
            ab = R[leg]["ab"]; tot = ab.get("totals") or {}; fb = tot.get("fallback_breakdown") or {}
            routines = ab.get("routines") or {}
            disp[leg] = {"mode": mode, "evidence": ab.get("evidence_source"), "schema": ab.get("schema"), "build": ab.get("build"), "totals": tot,
                         "routines": {k: {kk: v.get(kk) for kk in ("calls", "gpu_calls", "cpu_calls", "fallback_total", "max_m", "max_n", "max_k", "total_ms", "gpu_ms")} for k, v in routines.items()},
                         "gpu_verified": ab.get("gpu_verified"), "lib_loaded": ab.get("lib_loaded"), "lib_ok": ab.get("lib_ok"), "per_call_lines": ab.get("per_call_lines")}
            counts = ", ".join(f"{k} {v.get('gpu_calls')}/{v.get('calls')}" for k, v in routines.items())
            echo_want = launch.MODES[mode]["ab_mode"]; echo_ok = ab.get("ab_mode_echo") == echo_want        # the library's own AB_REPORT echo, not the request
            if mode == "ab_cpu":
                checks.add(f"{leg}: no GEMM on the GPU (counters); AB_MODE echo == cpu", tot.get("calls", 0) > 0 and tot.get("gpu_calls") == 0 and echo_ok,
                           f"{counts} ({ab.get('evidence_source')}); AB_REPORT echoes AB_MODE={ab.get('ab_mode_echo')}", f"totals.gpu_calls == 0 and the library's AB_MODE echo == {echo_want!r}")
            elif mode == "ab_gpu_all":
                hard = sum(fb.get(k, 0) or 0 for k in ("no_device", "gpu_failed", "other"))
                checks.add(f"{leg}: every GEMM on the GPU, no hard fallback (counters); AB_MODE echo == gpu", tot.get("calls", 0) > 0 and tot.get("gpu_calls") == tot.get("calls") and hard == 0 and echo_ok,
                           f"{counts}; fallbacks {fb}" + (" [amber: too_large > 0]" if fb.get("too_large") else "") + f"; AB_REPORT echoes AB_MODE={ab.get('ab_mode_echo')}",
                           f"gpu_calls == calls > 0; no_device == gpu_failed == other == 0 (too_large amber); the library's AB_MODE echo == {echo_want!r}")
            elif mode == "ab_auto":
                has_info = bool(R[leg]["binary"].get("ab_build_info"))
                d = rnd(E(leg) - E("R4_cpu"))
                checks.add("R4_auto: 1.4.0 dispatch table keeps QE shapes on Accelerate; energy equals ab_cpu", (tot.get("gpu_calls") == 0 and d < 1e-8) or not has_info,
                           f"{counts}; |dE| vs ab_cpu {d:.1e} Ry" + ("" if has_info else " (pre-1.4.0 dylib: reported, not gated)"), "gpu_calls == 0 and |dE| < 1e-8 Ry")
            if name == "R4":
                checks.add(f"{leg}: profile has a dgemm row and no zgemm row (Gamma trick)", "dgemm" in routines and "zgemm" not in routines, f"routines {list(routines)}", "regterg is real arithmetic")
            if name == "R6":
                checks.add(f"{leg}: profile has both dgemm and zgemm rows", "dgemm" in routines and "zgemm" in routines, f"routines {list(routines)}: {counts}",
                           "cegterg ZGEMM + PAW/augmentation DGEMM. The campaign's B run counted 29048 ZGEMM + 2580 DGEMM at conv_thr 1e-6 without diago_full_acc, forces or stress (9 iterations); R6 here "
                           "runs conv_thr 1e-9 with diago_full_acc, tprnfor and tstress (12 iterations), so the counts are not comparable (diago_full_acc is the likely driver of the 2.65x; not isolated)")
            checks.add(f"{leg}: AB_REPORT build string and dyld-loaded library", ab.get("build") == frz.ab_build_info and ab.get("build_matches_dylib") and ab.get("lib_loaded") == frz.dylib_path and ab.get("lib_ok"),
                       f"{ab.get('build')!r}; loaded {ab.get('lib_loaded')}", f"== strings ab_build_info and == {frz.dylib_path}")
        S["dispatch"] = disp
        # profiler dimensions pinned to QE's own numbers (per-call lines exist for every GPU call: the gpu_all legs)
        dims = {}
        if "R4_gpu" in rows:
            x, ab = R["R4_gpu"]["_extra"], R["R4_gpu"]["ab"]
            npw, ngm = (x["npw_per_k"] or [None])[0], x["ngm"]
            dg = (ab.get("routines") or {}).get("dgemm") or {}
            calls = R["R4_gpu"]["_dims"].get("dgemm", [])
            has_m = any(m == 2 * npw for m, n, k in calls) if npw else False
            has_k = any(k == 2 * npw for m, n, k in calls) if npw else False
            checks.add("R4 dgemm: regterg's npw2 = 2 x printed PWs appears as M and as K of real calls", has_m and has_k and len(calls) == dg.get("calls"),
                       f"npw {npw} -> {2 * npw if npw else None}: M-match {has_m}, K-match {has_k} over {len(calls)} per-call lines ({dg.get('calls')} dgemm calls)", "both present (regterg 'N','N' npw2 x notcnv and 'T','N' ... x npw2)")
            checks.add("R4 dgemm max_k == 2 x ngm (PAW newd / addusforce) and max_m == 2 x ngm (addusdens/addusstress)", ngm is not None and dg.get("max_k") == 2 * ngm and dg.get("max_m") == 2 * ngm,
                       f"ngm {ngm} -> {2 * ngm if ngm else None}; profile max_m {dg.get('max_m')}, max_n {dg.get('max_n')}, max_k {dg.get('max_k')}", "max_k == max_m == 2*ngm (the plan's 2*npw pin for max_m is wrong on disk: addusdens has M = 2*ngm)")
            dims["R4_gpu"] = {"npw": npw, "ngm": ngm, "dgemm": {k: dg.get(k) for k in ("calls", "max_m", "max_n", "max_k")}, "per_call": len(calls), "M_eq_2npw": sum(1 for m, n, k in calls if m == 2 * npw), "K_eq_2npw": sum(1 for m, n, k in calls if k == 2 * npw)}
        if "R6_gpu" in rows:
            x, ab = R["R6_gpu"]["_extra"], R["R6_gpu"]["ab"]
            npwx, ngm = max(x["npw_per_k"] or [0]), x["ngm"]
            zg, dg = (ab.get("routines") or {}).get("zgemm") or {}, (ab.get("routines") or {}).get("dgemm") or {}
            nkb = 64 * nh
            checks.add("R6 zgemm max_m == max_k == max printed PWs over k (npwx)", npwx > 0 and zg.get("max_m") == npwx and zg.get("max_k") == npwx, f"PWs {x['npw_per_k']} -> npwx {npwx}; zgemm max_m {zg.get('max_m')}, max_k {zg.get('max_k')}", "cegterg kdim = npw")
            checks.add("R6 dgemm max_k == 2 x ngm", ngm is not None and dg.get("max_k") == 2 * ngm, f"ngm {ngm} -> {2 * ngm if ngm else None}; dgemm max_m {dg.get('max_m')}, max_n {dg.get('max_n')}, max_k {dg.get('max_k')}", "PAW newd DGEMM nij x nat x 2*ngm")
            checks.add("R6 zgemm max_n <= max(2 nbnd, nkb) (upper bound)", zg.get("max_n") is not None and zg.get("max_n") <= max(2 * NBND_64, nkb), f"max_n {zg.get('max_n')} vs bound {max(2 * NBND_64, nkb)} (nkb = 64 x {nh})", "a loose bound from cegterg nvecx and calbec, not a pin")
            dims["R6_gpu"] = {"npw_per_k": x["npw_per_k"], "npwx": npwx, "ngm": ngm, "nkb": nkb, "zgemm": {k: zg.get(k) for k in ("calls", "max_m", "max_n", "max_k")}, "dgemm": {k: dg.get(k) for k in ("calls", "max_m", "max_n", "max_k")}}
        S["dims"] = dims
        if "R4_trace" in rows:
            tr = parse.parse_trace_headers(R["R4_trace"]["run_dir"])
            n_above = 0
            for f in Path(R["R4_trace"]["run_dir"]).glob("ab_trace.*"):
                for line in f.read_text(errors="replace").splitlines():
                    m = re.search(r"\bM=(\d+)\s+N=(\d+)\s+K=(\d+)", line)
                    if m and not line.startswith("#") and min(int(v) for v in m.groups()) > 32:
                        n_above += 1
            h = tr[0] if tr else {}
            d = rnd(E("R4_trace") - E("R4_cpu"))
            checks.add("R4_trace (3ae1, library floor): path=gpu lines == calls with min(M,N,K) > 32; header floor 32; energy == ab_cpu",
                       bool(tr) and h.get("min_gpu_dim") == 32 and h.get("path_gpu") == n_above and h.get("min_dim_below_floor") == 0 and d < 1e-8,
                       f"header AB_MIN_GPU_DIM {h.get('min_gpu_dim')}, path=gpu {h.get('path_gpu')} vs {n_above} above the floor, {h.get('path_cpu')} cpu, |dE| {d:.1e} Ry", "all equal; < 1e-8 Ry")
            S["trace"] = {"headers": tr, "n_above_floor": n_above, "dE_vs_cpu": d}
        # parity tables through abrun's compare (same-pw.x rows first, then cross-build; walls untimed)
        S["compare_md"] = {}
        for name in ("R4", "R6"):
            group = [R[l] for l in (name, f"{name}_cpu", f"{name}_gpu", f"{name}_auto", f"{name}_trace") if l in R]
            if len(group) > 1:
                S["compare_md"][name] = compare.compare(group, ref_id=R[name]["id"], energy_tol=1e-6)["markdown"]      # the example's cross-build tolerance, not abrun's 1e-5 default
        print("  walls (QE WALL, s, untimed observations; lock %s): " % args.lock + ", ".join(f"{l} {rows[l]['wall_qe_s']}" for l in rows))

    # ---------------------------------------------------------------- S4 plots, RESULTS.md, summary.json
    stage("S4. Plots and RESULTS.md")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
        e0 = E("R2") / 8
        order_plot = [l for l in ("R1", "R1p", "R2", "R2p", "R3", "R4", "R5", "R6", "R4_cpu", "R4_gpu", "R4_auto", "R6_cpu", "R6_gpu") if l in rows]
        labels = {"R1": "R1 fcc/2 (32 k)", "R1p": "R1' rho 200", "R2": "R2 cubic/8 (2x2x2)", "R2p": "R2' rho 200", "R3": "R3 cubic/8 (4x4x4)", "R4": "R4 64/Gamma", "R5": "R5 64/Gamma cplx",
                  "R6": "R6 64/2x2x2", "R4_cpu": "R4 ab_cpu", "R4_gpu": "R4 ab_gpu_all", "R4_auto": "R4 ab_auto", "R6_cpu": "R6 ab_cpu", "R6_gpu": "R6 ab_gpu_all"}
        ys = [(rows[l]["E_per_atom_Ry"] - e0) * 1e3 if rows[l]["E_per_atom_Ry"] is not None else np.nan for l in order_plot]
        seen = set()
        for i, l in enumerate(order_plot):
            kind = "reference pw.x" if rows[l]["mode"] == "reference" else ("apple-bottom gpu" if "_gpu" in l else "apple-bottom cpu/auto")
            col = PALETTE[0] if kind.startswith("reference") else PALETTE[1]
            ax[0].plot([i], [ys[i]], "o", ms=7, color=col, mfc="white" if kind.endswith("gpu") else col, mew=1.8, label=None if kind in seen else kind)
            seen.add(kind)
        ax[0].axhline(0, color="#999", lw=0.8)
        ax[0].set_xticks(np.arange(len(order_plot))); ax[0].set_xticklabels([labels[l] for l in order_plot], rotation=55, ha="right", fontsize=8)
        ax[0].set_ylabel("E/atom - E(R2)/8  (mRy)"); ax[0].set_title("Energy per atom of each cell (4x4x4 cells sit 7.8 mRy lower)", fontsize=9); ax[0].legend(fontsize=8, loc="center right")
        if unf:
            for j, blk in enumerate(unf["parent_per_k"]):
                for e in blk["eig"]:
                    ax[1].plot([j - 0.1 * blk["mult"], j + 0.1 * blk["mult"]], [e, e], color=PALETTE[2], lw=1.0, label="8-atom mesh (width = multiplicity)" if (j == 0 and e == blk["eig"][0]) else None)
            for n_, e in enumerate(unf["supercell_eig"]):
                ax[1].plot([4.6, 5.4], [e, e], color=PALETTE[0], lw=0.8, alpha=0.9, label="64-atom Gamma" if n_ == 0 else None)
            ax[1].axhline(unf["E_star_eV"], color="#999", ls="--", lw=0.8)
            ax[1].text(5.45, unf["E_star_eV"], "E*", fontsize=7, va="bottom", color="#52514e")
            ax[1].axhspan(unf["homo_lumo_R4"][0], unf["homo_lumo_R4"][1], color="#eee", zorder=0)
            ax[1].set_xticks([0, 1, 2, 3, 5]); ax[1].set_xticklabels([f"k{j + 1}\n({k['k'][0]:.1f},{k['k'][1]:.1f},{k['k'][2]:.1f})\nx{m}" for j, (k, m) in enumerate(zip(x2["klist"], unf["multiplicities"]))] + ["64-atom\nGamma"], fontsize=7)
            ax[1].set_ylabel("eigenvalue (eV)"); ax[1].set_ylim(min(unf["supercell_eig"]) - 2.2, unf["E_star_eV"] + 1.5); ax[1].legend(fontsize=8, loc="lower center", ncol=2)
            ax[1].set_title(f"Band folding below E* = min_k eps_21: max |delta| {unf['max_delta_meV']:.2f} meV (gap shaded)", fontsize=9)
        fig.tight_layout(); fig.savefig(RES / "si_supercell_folding.png", dpi=140); print(f"  wrote {RES / 'si_supercell_folding.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  plot skipped: {exc}")
    S["checks"] = checks.rows; S["all_pass"] = checks.all_pass; S["summary"] = checks.summary()
    S["n_pass"], S["n_fail"], S["n_reported"] = checks.n_pass, checks.n_fail, checks.n_reported
    S["total_wall_s"] = time.time() - t0
    exlib.dump_json(RES / "summary.json", S)
    (RES / "RESULTS.md").write_text(results_md(S, checks))
    print(f"\n{'ALL GATED CHECKS PASSED' if checks.all_pass else 'SOME CHECKS FAILED'}: {checks.summary()} in {S['total_wall_s']:.0f}s -> {RES / 'RESULTS.md'}")
    return 0 if checks.all_pass else 1


def results_md(S, checks):
    p, rows = S["provenance"], S["runs"]
    rr, tracked = p.get("run_records") or {}, p.get("tracked_in_commit") or {}
    L = [f"# Si supercell folding: results ({p['date']})", "",
         f"rainbow-connection {p['rainbow_connection_commit']}, {p['hardware']}, macOS {p['macos']}. Reference pw.x {p['pw_x_reference']} "
         f"(sha256 {p['pw_x_reference_sha256'][:12]}, {p['pw_x_reference_ab_symbols']} apple-bottom symbols, QE 7.4.1 OpenBLAS); frozen apple-bottom {p['ab_frozen']} "
         f"(pw.x {p['ab_pw_x_sha256'][:12]}, dylib {p['ab_dylib_sha256'][:12]}, {p['ab_build_info']}, {p.get('ab_qe_build')}). "
         f"ecutwfc {S['ecutwfc']} / ecutrho {S['ecutrho']} Ry, a = {S['a_conv_bohr']} bohr, PAW {S['pseudo']['file']} (sha256 {S['pseudo']['sha256'][:12]}). "
         f"Lock {S['lock']}; reference legs mpirun -np {S['args']['ranks']}, frozen legs -np 1 OMP_NUM_THREADS=1. {S['walls_note']}.", "",
         (f"Regenerated with --reuse at {p['date']} from run records {rr.get('first_started')} .. {rr.get('last_finished')} ({len(rr.get('reused', []))} legs reused, "
          f"{len(rr.get('launched', []))} launched)" if p.get("regenerated_with_reuse") else
          f"Measured {rr.get('first_started')} .. {rr.get('last_finished')} ({len(rr.get('launched', []))} legs launched)")
         + f". Driver run.py sha256 {(p.get('driver_sha256') or '')[:12]}; tools/abrun/*.py sha256 in summary.json provenance.abrun_sha256 ({len(p.get('abrun_sha256') or {})} files)"
         + ("" if all(tracked.values()) else f"; commit {p['rainbow_connection_commit']} does not contain " + " or ".join(k for k, v in tracked.items() if not v) + " (untracked)") + ".", "",
         f"**{checks.summary()}** (REPORTED rows are findings and controls, never counted).", "",
         "## Checks", "", checks.markdown(), "", "## Runs", "",
         "| leg | input | binary | mode | ranks/nk | E (Ry) | E/atom (Ry) | iters | k | sym ops | FFT | PWs max | G-vectors | P (kbar) | max |F| | QE wall (s), observation | lock | load1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for leg, r in rows.items():
        L.append(f"| {leg} | {r['input']}: {DESC[r['input']]} | {r['binary']} | {r['mode']} | {r['ranks']}/{r['nk']} | {r['E_Ry']} | {r['E_per_atom_Ry']:.9f} | {r['iters']} | {r['nks']} | {r['nsym']} ({r['sym_note']}) | "
                 f"{'x'.join(str(v) for v in r['fft']) if r['fft'] else '-'} | {max(r['npw_per_k']) if r['npw_per_k'] else '-'} | {r['ngm']} | {r['pressure_kbar']} | {r['max_force']:.1e} | {r['wall_qe_s']:.2f} | {r['lock_effective']} | {r['load1_at_acquisition']:.2f} |"
                 if r["E_Ry"] is not None else f"| {leg} | {r['input']} | {r['binary']} | {r['mode']} | {r['ranks']}/{r['nk']} | {r['state']} | | | | | | | | | | | | |")
    L += ["", "## Energy terms per atom (Ry)", "", "| leg | one-electron | hartree | xc | ewald | one-centre PAW | total |", "|---|---|---|---|---|---|---|"]
    for leg, r in rows.items():
        t = r["terms"] or {}
        if r["E_Ry"] is not None and all(t.get(k) is not None for k in ("one_electron", "hartree", "xc", "ewald", "paw")):
            n = NAT[r["input"]]
            L.append(f"| {leg} | {t['one_electron']/n:.9f} | {t['hartree']/n:.9f} | {t['xc']/n:.9f} | {t['ewald']/n:.9f} | {t['paw']/n:.9f} | {r['E_Ry']/n:.9f} |")
    f = S.get("fold", {})
    L += ["", f"Fold: |E(R4)/64 - E(R2)/8| = {f.get('d_R4_R2_per_atom', float('nan')):.2e}, |E(R6)/64 - E(R3)/8| = {f.get('d_R6_R3_per_atom', float('nan')):.2e}, difference of differences {f.get('diff_of_diffs', float('nan')):.2e} Ry/atom; "
          f"Gamma-only sampling error {f.get('gamma_sampling_error_Ry_per_atom', float('nan'))*1e3:+.4f} mRy/atom. Primitive vs cubic: {S['cubic_vs_primitive']['d100']:.2e} Ry/atom at ecutrho 100, {S['cubic_vs_primitive']['d200']:.2e} at 200."]
    u = S.get("unfolding") or {}
    if u:
        L += ["", "## Unfolding", "", f"E* = {u['E_star_eV']:.4f} eV; {u['n_supercell']} supercell states vs {u['n_parent']} parent states (with multiplicities {u['multiplicities']}) below E* - {UNFOLD_MARGIN*1e3:.0f} meV; "
              f"max |delta| = {u['max_delta_meV']:.3f} meV at sorted index {u['worst_index']}; {u['n_levels']} distinct parent levels all with the right degeneracy count. "
              f"HOMO/LUMO: R4 {u['homo_lumo_R4'][0]:.4f}/{u['homo_lumo_R4'][1]:.4f} eV, R2 {u['homo_lumo_R2'][0]:.4f}/{u['homo_lumo_R2'][1]:.4f} eV."]
    if S.get("compare_md"):
        L += ["", "## apple-bottom parity (abrun compare; walls are untimed observations, never compared)", ""]
        for name, md in S["compare_md"].items():
            L += [f"### {name}", "", md, ""]
    if S.get("dispatch"):
        L += ["## Dispatch evidence (the library's own counters)", "", "| leg | mode | evidence | build | routine | calls | gpu | cpu | fallbacks | max M | max N | max K | total ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for leg, d in S["dispatch"].items():
            for rt, v in d["routines"].items():
                L.append(f"| {leg} | {d['mode']} | {d['evidence']} ({d['schema']}) | {d['build']} | {rt} | {v['calls']} | {v['gpu_calls']} | {v['cpu_calls']} | {v['fallback_total']} | {v['max_m']} | {v['max_n']} | {v['max_k']} | {v['total_ms']:.0f} |")
        dm = S.get("dims", {})
        if dm:
            L += ["", "Dimension pins: " + "; ".join(f"{k}: {json.dumps({kk: vv for kk, vv in v.items() if kk != 'npw_per_k'})}" for k, v in dm.items())]
    L += ["", "## Files", "", "- results/inputs/*.in (generated), results/pseudo/, results/ref/<leg>/ and results/ab/<leg>/ (abrun run records: input.in, out.txt, err.txt, run.json, summary.json, machine.json, ab_profile.json), results/summary.json, results/si_supercell_folding.png"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
